import copy
from dataclasses import dataclass
import json
import logging
from typing import Dict, Sequence

import torch
from torch.utils.data import Dataset
import transformers 

from modelzip.config import LANGS_MAP

LOG = logging.getLogger(__name__)

IGNORE_INDEX = -100 

TRANSLATE_PROMPT = """<s>Below is a sentence in {source_lang}. Translate it to {target_lang}. 

### Sentence:
{source_sentence}

### Translation:
"""


def load_json_data(file_path) -> Sequence[Dict]:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return [json.loads(line) for line in file]
        except json.JSONDecodeError as e:
            LOG.error(f"Error decoding JSON: {e}")
            raise e

def verify_translation_type(file_path, source_lang, target_lang):
    if f"{source_lang}-{target_lang}" not in file_path:
        raise Exception(
            f"data_path does not match with source & target lang."
                f"data_path: {file_path},"
                f"source: {source_lang},"
                f"target: {target_lang}"
        )

def get_lang_name(shorthand:str):
    """Return full name of the source_lang.
    
    Args:
        source_lang: a shorthand for the language (e.g. cse, deu...)
    """
    if (shorthand in LANGS_MAP.keys()):
        return LANGS_MAP[shorthand]
    else:
        raise ValueError("Unexpected language type.")

def get_source_sentence(example):
    if "source_sentence" in example: 
        return example["source_sentence"]
    else:
        raise ValueError("Unexpected data format.")

def get_translation(example):
    if "translation" in example:
        return example["translation"]
    else:
        raise ValueError("Unexpected data format.")


def _tokenize_fn(
    strings:Sequence[str], 
    tokenizer: transformers.PreTrainedTokenizer
):
    ids_list = tokenizer(
        strings,
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_attention_mask=False,
    )["input_ids"]

    input_ids = []
    input_ids_lens = []

    for ids in ids_list:
        input_ids.append(torch.tensor(ids))
        input_ids_lens.append(len(ids))

    return dict(
        input_ids=input_ids,
        input_ids_lens=input_ids_lens,
    )

    

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    examples = [s + t for s, t in zip(sources, targets)]
    LOG.debug(f"Example data-------------->\n{examples[0]}")
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)



class SupervisedTranslationDataset(Dataset):
    def __init__(
        self,
        data_path:str,
        source_lang:str,
        target_lang:str,
        tokenizer: transformers.PreTrainedTokenizer,
    ):
        super().__init__()
        LOG.info("Creating dataset...")
        list_data_dict = load_json_data(data_path) 
        sources = [
            TRANSLATE_PROMPT.format_map(
                {
                    "source_lang": get_lang_name(source_lang),
                    "target_lang": get_lang_name(target_lang),
                    "source_sentence": get_source_sentence(example)
                }
            )
            for example in list_data_dict
        ]

        targets = [
            f"{get_translation(example).replace('</s>', '')} {tokenizer.eos_token}"
            for example in list_data_dict
        ]
        
        LOG.info("Tokenizing inputs...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])



@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


