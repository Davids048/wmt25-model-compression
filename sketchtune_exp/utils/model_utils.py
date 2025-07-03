import math
import os
import json
import torch

from transformers import AutoConfig, AutoTokenizer


def get_tokenizer(model_name_or_path, fast_tokenizer=True):
    if "llama" in model_name_or_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer, add_bos_token = False)       # not adding start token 
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.padding_side = 'right'
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer, add_bos_token = False)      # not adding start token 
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
    return tokenizer


def load_hf_tokenizer(model_name_or_path,
                      fast_tokenizer=True,
                      add_special_tokens=None):
    if os.path.exists(model_name_or_path):
        # Locally tokenizer loading has some issue, so we need to force download
        model_json = os.path.join(model_name_or_path, "config.json")
        if os.path.exists(model_json):
            model_json_file = json.load(open(model_json))
            model_name = model_json_file.get("_name_or_path",
                                             model_name_or_path)
            tokenizer = get_tokenizer(model_name,
                                      fast_tokenizer=fast_tokenizer)
        else:
            raise Exception("Model config.json not found.")
    else:
        tokenizer = get_tokenizer(model_name_or_path,
                                  fast_tokenizer=fast_tokenizer)

    
    if add_special_tokens is not None:
        add_special_tokens = [add_special_tokens] if isinstance(add_special_tokens, str) \
            else add_special_tokens
        tokenizer.add_special_tokens(
            {'additional_special_tokens': add_special_tokens})

    return tokenizer


# def configure_dropout(model_config, dropout):
#     if dropout is not None:
#         for key in ('dropout', 'attention_dropout', 'hidden_dropout',
#                     'activation_dropout'):
#             if hasattr(model_config, key):
#                 print(f"Setting model_config.{key} to {dropout}")
#                 setattr(model_config, key, dropout)
#


def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    trained=False,
                    dropout=None):
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    # configure_dropout(model_config, dropout)
    if trained:
        # the weight loading is handled by create critic model
        model = model_class.from_config(model_config)
    else:
       model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            torch_dtype=torch.bfloat16,
            config=model_config)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    model.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    return model


def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    learning_rate,
    no_decay_name_list=[
        "bias", "layer_norm.weight", "layernorm.weight", "norm.weight",
        "ln_f.weight"
    ],
    trainable_name_list=['s2'],
):  
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n.lower() for nd in trainable_name_list))
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n.lower() for nd in trainable_name_list))
            ],
            "weight_decay": weight_decay,
            "lr": learning_rate,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n.lower()
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    non_empty_groups = []
    for group in optimizer_grouped_parameters:
        if group["params"]:
            non_empty_groups.append(group)
    return non_empty_groups


