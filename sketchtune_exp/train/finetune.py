import time
import argparse
import logging
import math
import os
import pickle
import sys
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModelForCausalLM, get_scheduler
import wandb

from sketchtune.tailor_utils import replace_layers, save_pretrained
from utils.log import setup_logging
from utils.config import parse_exp_args_yaml, create_exp_output_dir
from utils.data_utils import (
    DataCollatorForSupervisedDataset,
    SupervisedTranslationDataset,
    verify_translation_type,
)
from utils.model_utils import (
    create_hf_model, 
    load_hf_tokenizer, 
    get_optimizer_grouped_parameters
)
from utils.utils import get_all_reduce_mean, set_random_seed, to_device



LOG = logging.getLogger(__name__)



def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf", 
        type=str, 
        required=True, 
        help="Experiment config file."
    )
    parser.add_argument(
        "--debug",
        action="store_true"
    )
    args = parser.parse_args()

    model_config, train_config, data_config = parse_exp_args_yaml(args.conf)
    
    ## Misc setups
    device = torch.device('cuda')
    set_random_seed(train_config.seed)
    output_dir = create_exp_output_dir(model_config, train_config, data_config)
    wandb.init(project='SketchTune-WMT', name=output_dir.split('/')[-1])
    setup_logging(output_dir, args.debug)
    LOG.info(f"Output dir: {output_dir}")
    LOG.info(f"Configs:\n{model_config}\n{train_config}\n{data_config}")
    LOG.debug(f"{train_config.learning_rate} {type(train_config.learning_rate)}")
    LOG.debug(f"{type(train_config.lr_scheduler_type)}")

    ## Load Tokenizer
    tokenizer = load_hf_tokenizer(
        model_config.model_path,
        fast_tokenizer = True
    )
    tokenizer.model_max_length = train_config.max_seq_len


    ## Load Model 
    model = create_hf_model(
        AutoModelForCausalLM,
        model_config.model_path,
        tokenizer,
    )
    if isinstance(model_config.model_quantizer_path, str):
        for param in model.parameters():
            param.requires_grad = False
        with open(model_config.model_quantizer_path, 'rb') as file:
            quantizers = pickle.load(file)
        replace_layers(model, quantizers)
        model.save_pretrained = type(model.save_pretrained)(save_pretrained, model)

    else:
        raise Exception("Unsupported Quantizer argument.")
    num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOG.info(f"Num Trainable Params: {num}")
    LOG.info("Finished loading model.")
    model = model.to(device)


    ## Load Data 
    assert isinstance(data_config.data_path, str)
    assert ".json" in data_config.data_path, "Only support json data file."
    verify_translation_type(
        data_config.data_path,
        data_config.source_lang,
        data_config.target_lang,
    )
    dataset = SupervisedTranslationDataset(
        data_path = data_config.data_path,
        source_lang = data_config.source_lang,
        target_lang = data_config.target_lang,
        tokenizer = tokenizer,
    )
    if data_config.val_set_size > 0:
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [len(dataset) - data_config.val_set_size, data_config.val_set_size]
        )
        LOG.info(f"Split data set: Val set size: {len(val_dataset)}")
    else:
        train_dataset = dataset
        val_dataset = None   
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    LOG.info(f"Finished loading data. Train set size: {len(train_dataset)}")


    ## Setup Data for Training
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=train_config.per_device_train_batch_size,
        collate_fn=data_collator,
    )
    if val_dataset:
        val_sampler = SequentialSampler(val_dataset)
        val_dataloader = DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=train_config.per_device_eval_batch_size,
            collate_fn=data_collator,
        )
    else:
        val_dataloader = None
    LOG.info("Finished setting up train(val) data loader.")


    ## Setup optimizer
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, 
        train_config.weight_decay, 
        train_config.learning_rate
    )
    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=train_config.learning_rate
    )


    ## Setup scheduler
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / train_config.gradient_accumulation_steps
    )
    lr_scheduler = get_scheduler(
        name=train_config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=train_config.num_warmup_steps,
        num_training_steps=train_config.num_train_epochs * num_update_steps_per_epoch,
    )
    
    LOG.info(">>>>> Start Training >>>>>")
    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            losses = get_all_reduce_mean(losses)
        except:
            LOG.debug("Failed to get all reduced mean loss.")
            pass
        try:
            perplexity = torch.exp(losses).item()
        except OverflowError:
            perplexity = float("inf")
        model.train()
        return perplexity, losses.item()


    total_training_steps = train_config.num_train_epochs * num_update_steps_per_epoch
    current_step_count = 0

    #TODO: Not sure what these are for. 
    eval_step = train_config.eval_step * train_config.gradient_accumulation_steps
    eval_delay = (
        train_config.eval_step
        if isinstance(train_config.eval_delay, int) 
        else int(train_config.eval_delay * total_training_steps)
    )

    wandb.watch(model)
    for epoch in range(train_config.num_train_epochs):
        LOG.info(f"Starting Epoch {epoch+1}.")
        model.train()

        mean_loss = 0.0
        step_loss = 0.0

        start = time.time()
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False) 
            loss = outputs.loss 

            total_loss = loss / train_config.gradient_accumulation_steps 
            total_loss.backward() 

            mean_loss += total_loss.item() 
            step_loss += total_loss.item()

            # Update LR after full batch.
            if ((step + 1) % train_config.gradient_accumulation_steps == 0 
                    or step + 1 == len(train_dataloader)):
                lr = (
                    lr_scheduler.get_last_lr()[1]
                    if len(lr_scheduler.get_last_lr()) > 1
                    else lr_scheduler.get_last_lr()[0]
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    train_config.grad_clip
                )

                optimizer.step() 
                lr_scheduler.step() 
                optimizer.zero_grad()

                # Postprocess
                current_step_count += 1
                end = time.time() 

                LOG.info(
                    f"Batch Loss: {step_loss:.4f} - "
                    f"Mean loss: {mean_loss:4f} - "
                    f"LR: {lr:.8f} - "
                    f"Step: {current_step_count} / {total_training_steps} - "
                    f"Batch time: {end - start:.4f} sec\n"
                )

                metrics = {
                    "loss": step_loss, 
                    "lr": lr,
                    "step": current_step_count,
                    "step_time": float(end - start)
                }
                wandb.log(metrics)
                
                # Reset for next full batch 
                step_loss = 0.0 
                start = time.time() 

            # Perform Eval if necessary 
            if (
                current_step_count % train_config.eval_step == 0 
                and current_step_count >= eval_delay
                and val_dataloader is not None 
            ):
                ppl, val_loss = evaluation(model, val_dataloader)
                LOG.info(
                    f"Validation perplexity: {ppl}, Validation loss: {val_loss}")
                # if val_loss < best_val_loss:
                #     best_val_loss = val_loss
                #     if args.global_rank == 0:
                #         best_model = copy.deepcopy(model).to("cpu")
                #     final_saved_model_index = current_step_count



        LOG.info(
            f"------------\n"
            f"Epoch {epoch+1} / {train_config.num_train_epochs}\n"
            f"Train loss: {mean_loss/len(train_dataloader)}"
            f"------------\n"
        )
        model.save_pretrained(os.path.join(output_dir, f"epoch_{epoch}"))

    flag_file = os.path.join(output_dir, "._OK")
    with open(flag_file, 'w') as f:
        pass 
    LOG.info("FINISHED TRAINING.")


if __name__ == "__main__":
    main()
