#!/usr/bin/env python
import logging
import torch
import argparse
import pickle
import sys
from pathlib import Path
from typing import List

from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from modelzip.config import (
    DEF_BATCH_SIZE,
    LANGS_MAP,
    LOG,
    # TRANSLATE_PROMPT,
    USE_CHAT_TEMPLATE,
)
from sketchtune_exp.utils.model_utils import load_hf_tokenizer, create_hf_model
from sketchtune_exp.sketchtune.tailor_utils import replace_layers
from sketchtune_exp.utils.generation_utils import generate_completions

LOG = logging.getLogger(__name__)

TRANSLATE_PROMPT = """<s>Below is a sentence in {source_lang}. Translate it to {target_lang}. 

### Sentence:
{source_sentence}

### Translation:
"""



class LLMWrapper:
    def __init__(
        self,
        model_dir: Path,
        use_chat_template=True,
        prompt_template=TRANSLATE_PROMPT,
        progress_bar=False,
    ):
        self.model_dir = Path(model_dir)
        self.use_chat_template = use_chat_template
        self.prompt_template = prompt_template
        self.progress_bar = progress_bar
        self._tokenizer = None
        self._model = None
        self.device=torch.device("cuda")

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = load_hf_tokenizer(
                "CohereLabs/aya-expanse-8b",
                fast_tokenizer=True
            )
            self._tokenizer.padding_side = 'left'
            LOG.info(f"Tokenizer padding: {self._tokenizer.padding_side}")
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._model = create_hf_model(
                AutoModelForCausalLM,
                "CohereLabs/aya-expanse-8b",
                self.tokenizer
            )
            with open(self.model_dir/"quantizers.pkl", 'rb') as f:
                quantizers = pickle.load(f)
            with open(self.model_dir/"quant_grids.pkl", 'rb') as f:
                quant_grids = pickle.load(f)
            replace_layers(self._model, quantizers, quant_grids)

            self._model = self._model.to(device=self.device)
            self._model.eval()
        return self._model



    def translate_lines(self, pair: str, lines: List[str], batch_size: int = DEF_BATCH_SIZE):
        src, tgt = pair.split("-")
        src_lang = LANGS_MAP[src]
        tgt_lang = LANGS_MAP[tgt]

        prompts = [] 
        for i, line in enumerate(lines):
            prompts.append(
                TRANSLATE_PROMPT.format(
                    source_lang=src_lang,
                    target_lang=tgt_lang,
                    source_sentence=line.strip()
                )
            )
        generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            pad_token_id=self.model.config.pad_token_id,
            eos_token_id=self.model.config.eos_token_id,
            bos_token_id=self.model.config.bos_token_id,
        )
        model_outputs = [] 
        outputs = generate_completions(
            model=self.model,
            device=self.device,
            tokenizer=self.tokenizer,
            prompts=prompts,
            max_new_tokens=self.model.config.max_position_embeddings,                          
            batch_size=batch_size,
            stop_id_sequences=[[self.tokenizer.eos_token]],
            verbose=False,
            generation_config = generation_config
        )
        model_outputs.extend(outputs)
        return model_outputs


def main():
    args = parse_args()
    llm = LLMWrapper(
        args.model,
        use_chat_template=USE_CHAT_TEMPLATE,
        prompt_template=args.prompt,
        progress_bar=args.progress,
    )
    if args.input is sys.stdin:
        LOG.info("Reading from stdin")  # just in case if we forget to pass input via STDIN
    # buffering all inputs into one big maxibatch for sorting based on length,
    # assuming test sets are not too big
    lines = args.input.read().splitlines()
    lines = [x.strip() for x in lines]  # remove empty lines
    assert not any(x == "" for x in lines), "Input file contains empty lines. Please fix them and try again."
    assert len(lines) > 0, "Input file is empty. Please provide some input."
    outputs = llm.translate_lines(args.langs, lines, batch_size=args.batch_size)
    assert len(outputs) == len(
        lines
    ), f"Output length {len(outputs)} does not match input length {len(lines)}"
    args.output.write("\n".join(outputs))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run translation using LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # eval script only sets these two required args. They are positional args for simplicity
    parser.add_argument("langs", help="Lang pairs to evaluate, eg, 'ces-deu")
    parser.add_argument("batch_size", type=int, default=DEF_BATCH_SIZE)

    # this script will/should be placed inside model directory for each model and called run.py,
    # so assume this file's parent dir as model dir
    my_name = Path(__file__).name
    my_dir = Path(__file__).parent
    if my_name == "run.py":
        parser.add_argument(
            "-m",
            "--model",
            type=Path,
            default=my_dir,
            help="Path to model directory that is compatible for HuggingFace Transformers",
        )
    else:
        parser.add_argument(
            "-m",
            "--model",
            type=Path,
            required=True,
            help="Path to model directory that is compatible for HuggingFace Transformers",
        )

    # optional args. Will not be set during evaluation, so make sure the defaults are correct
    parser.add_argument(
        "-i",
        "--input",
        type=argparse.FileType("r", encoding="utf-8", errors="replace"),
        default=sys.stdin,
        help="Input file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w", encoding="utf-8", errors="replace"),
        default=sys.stdout,
        help="Output file",
    )
    parser.add_argument("-pb", "--progress", action="store_true", help="Show progress bar")
    parser.add_argument(
        "-pt",
        "--prompt",
        type=str,
        default=TRANSLATE_PROMPT,
        help="Prompt template for translation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
