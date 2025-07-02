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




LOG = logging.getLogger(__name__)


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


    # Load Tokenizer 
    tokenizer = load_hf_tokenizer(
        model_config.model_path,
        fast_tokenizer=True
    )
    tokenizer.padding_side = 'left'
    LOG.info(f"Tokenizer padding: {tokenizer.padding_side}")

    # Load Model 
    model = create_hf_model(
        AutoModelForCausalLM,
        model_config.model_path,
        tokenizer
    )
    with open(model_config.model_quantizer_path, 'rb') as file:
        quantizers = pickle.load(file) 
    with open(os.path.join(eval_config.finetuned_quantizer_path, 'quant_grids.pkl'), 'rb') as f:
        quant_grids = pickle.load(f)
    replace_layers(model, quantizers, quant_grids)
    model = model.to(device)
    model.eval()

    # prep test data
    test_data = create_test_dataset(
        data_config.data_path, 
        data_config.source_lang, 
        data_config.target_lang
    )
    if args.debug:
        test_data = test_data[:100] if len(test_data) >=100 else test_data
        LOG.info(f"Using small test size for debugging, size: ({len(test_data)})")


    # Create generation 
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
        prompts=test_data,
        max_new_tokens=256,                          
        batch_size=eval_config.per_device_eval_batch_size,
        stop_id_sequences=[[tokenizer.eos_token]],
        verbose=False,
        generation_config = generation_config
    )
    model_outputs.extend(outputs)
    LOG.info("End generation <<<<<<<<<<<<<")

    # Write outputs 
    with open(os.path.join(output_dir, "output.txt"), 'w') as f:
        for line in model_outputs:
            f.write(f"{line.rstrip()}\n")

    LOG.info("Finished writing outputs to file.")

    

if __name__ == "__main__":
    main()
