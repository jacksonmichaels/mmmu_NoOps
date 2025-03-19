import os
import json
import random
import torch
import numpy as np
import yaml
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image

def save_json(path, data):
    path = Path(path)
    parent = path.parent.absolute()
    if not os.path.exists(parent):
        os.makedirs(parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_yaml(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_seed(seed_value):
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_single_sample(sample):
    if "image_path" in sample and os.path.isfile(sample["image_path"]):
        img = Image.open(sample["image_path"]).convert("RGB")
        sample["image"] = img
    return sample

def construct_prompt(sample, config=None):
    question = sample.get("question", "")
    sample["final_input_prompt"] = f"Question: {question}\nAnswer:"
    return sample

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--output_path", type=str, default="output.json")
    parser.add_argument("--config_path", type=str, default="configs/llava1.5.yaml")
    parser.add_argument("--data_path", type=str, default="MMMU/MMMU",
                        help="Hugging Face dataset path for MMMU")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Key in MODEL_DICT_LLMs (e.g. 'llava_one' or 'internvl2_5')")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--access_token", type=str, default=None,
                        help="If model is private, specify your HF token here.")
    parser.add_argument("--noop", type=str, default="none",  help="Define what NoOps to add [text, img, all, none]")
    parser.add_argument("--subject", type=str, default="all")
    parser.add_argument('-w', '--warn_noop', action='store_true')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--store_attention", action='store_true')
    return parser.parse_args()

def set_device_and_seed(seed, device):
    torch.manual_seed(seed)
    if device in ["cuda", "cpu"]:
        return device
    else:
        return "cuda:" + str(device)

def load_config(config_path):
    print(f"Loading config from {config_path} ...")
    config = load_yaml(config_path)
    for k, v in list(config.items()):
        if k != "eval_params" and isinstance(v, list) and len(v) == 1:
            config[k] = v[0]
    return config
