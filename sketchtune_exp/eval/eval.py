import argparse
import shutil
import logging
import os
import pickle
import sys
import torch

from transformers import AutoModelForCausalLM, GenerationConfig

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from sketchtune.tailor_utils import replace_layers
from utils.config import create_eval_output_dir, parse_eval_args_yaml
from utils.data_utils import create_test_dataset
from utils.generation_utils import generate_completions
from utils.log import setup_logging
from utils.model_utils import create_hf_model, load_hf_tokenizer
from utils.utils import set_random_seed

DEBUG_TEST_SIZE = 100


LOG = logging.getLogger(__name__)

def setup_model_tokenizer(
    model_path:str,
    model_quantizer_path:str,
    finetuned_quantizer_path:str,
    device
):
    tokenizer = load_hf_tokenizer(
        model_path,
        fast_tokenizer=True
    )
    tokenizer.padding_side = 'left'
    LOG.info(f"Tokenizer padding: {tokenizer.padding_side}")


    model = create_hf_model(
        AutoModelForCausalLM,
        model_path,
        tokenizer
    )
    with open(model_quantizer_path, 'rb') as file:
        quantizers = pickle.load(file) 
    with open(os.path.join(finetuned_quantizer_path, 'quant_grids.pkl'), 'rb') as f:
        quant_grids = pickle.load(f)
    replace_layers(model, quantizers, quant_grids)
    model = model.to(device)
    model.eval()
    return model, tokenizer

def generate(model, tokenizer, prompts, batch_size, device):
    LOG.info("Start generation >>>>>>>>>>")
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        bos_token_id=model.config.bos_token_id,
    )
    model_outputs = [] 
    outputs = generate_completions(
        model=model,
        device=device,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=256,                          
        batch_size=batch_size,
        stop_id_sequences=[[tokenizer.eos_token]],
        verbose=False,
        generation_config = generation_config
    )
    model_outputs.extend(outputs)
    LOG.info("End generation <<<<<<<<<<<<<")

    return model_outputs




def main():
    parser = argparse.ArgumentParser() 
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

    model_config, eval_config, data_config = parse_eval_args_yaml(args.conf)

    # Setup 
    device = torch.device('cuda') 
    set_random_seed(eval_config.seed) 
    output_dir = create_eval_output_dir(model_config, eval_config, data_config)
    shutil.copyfile(args.conf, os.path.join(output_dir, "config.yml"))
    setup_logging(output_dir, args.debug, file_name="eval.log")
    LOG.info(f"Output dir: {output_dir}")
    LOG.info(f"Configs:\n{model_config}\n{eval_config}\n{data_config}")


    model, tokenizer = setup_model_tokenizer(
        model_config.model_path, 
        model_config.model_quantizer_path,
        eval_config.finetuned_quantizer_path,
        device=device
    )


    # prep test data
    if args.debug:
        test_data = create_test_dataset(
            data_config.data_path, 
            data_config.source_lang, 
            data_config.target_lang,
            limit=DEBUG_TEST_SIZE,
        )
        LOG.info(f"Using small test size for debugging, size: ({len(test_data)})")
    else:
        test_data = create_test_dataset(
            data_config.data_path, 
            data_config.source_lang, 
            data_config.target_lang,
        )
        
    model_outputs = generate(
        model,
        tokenizer,
        test_data,
        eval_config.per_device_eval_batch_size,
        device=device
    )

    # Write outputs 
    with open(os.path.join(output_dir, "output.txt"), 'w') as f:
        for line in model_outputs:
            f.write(f"{line.rstrip()}\n")

    LOG.info("Finished writing outputs to file.")

if __name__ == "__main__":
    main()
