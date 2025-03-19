"""
This script is used to extract hidden states from the model.
"""
#%%
import convenience
import torch
import os
import random

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from dataset.simple_testset import SynthImgCapDataset
import importlib
import json
from dataset.shape_dataset.shape_dataset import COLORS

from model.zoo import MODEL_DICT

# Set grad enabled to False
torch.set_grad_enabled(False)

#%%
from nnsight import NNsight
#%%
from functools import partial
import pickle
#%%
def load_dataset(dataset_name, subsets=None):
    if dataset_name == "shapes":
        from dataset.shape_dataset.shape_dataset import ShapeDataset
        dset = ShapeDataset("dataset/shape_dataset/conflict_set_metadata.json", subset_len=subsets)
    else:
        dset = None
        print("UNKNOWN DATASET")
    print(f"Loaded {len(dset)} samples")
    return dset



def load_model(model_name, device = "cuda"):
    model = processor = call_engine_fn = prompt_format_fn = None
    model_path = MODEL_DICT[model_name]['model_path']
    if model_name.startswith("llava-one"):
        from model.llava_code import prepare_inputs_llava, format_prompt_llava, load_model_llava
        model, processor = load_model_llava(model_path, device)
        call_engine_fn = prepare_inputs_llava
        prompt_format_fn = format_prompt_llava
        
    if model_name == "idefics":
        from transformers import Idefics2Processor, Idefics2ForConditionalGeneration
        from model.idefics_code import call_idefics_engine, format_prompt_for_idefics
        
        call_engine_fn = call_idefics_engine
        prompt_format_fn = format_prompt_for_idefics
        processor = Idefics2Processor.from_pretrained(model_path)
        model = Idefics2ForConditionalGeneration.from_pretrained(model_path)
        model.to(device)
    
    if model_name == "blip-2":
        from model.blip_code import load_model_blip, call_engine_blip, format_prompt_blip
        model, processor = load_model_blip(model_path, device)
        call_engine_fn = call_engine_blip
        prompt_format_fn = format_prompt_blip
        
        
    nnsight_model = NNsight(model)
    call_engine_fn = partial(call_engine_fn, processor=processor, device=device)
    return nnsight_model, processor, call_engine_fn, prompt_format_fn


def extract_attention(dset, model, processor, prepare_input_fn, format_func, out_dir, COLOR_TOKEN_ID_DICT):
    device = "cuda"
    out_samples = {}
    with torch.no_grad():
        for sample in tqdm(dset):
            # Get token id of the color
            text_color_token_id = COLOR_TOKEN_ID_DICT.get(sample['params']['text_color'], None)
            image_color_token_id = COLOR_TOKEN_ID_DICT[sample['params']['image_color']]
            
            # sample = format_func(sample)
            # inputs = prepare_input_fn(inputs, processor, device)
            
            final_input_prompt = [{
                "role": "user",
                "content": sample['text_input']
            }]
            final_input_prompt = processor.apply_chat_template(final_input_prompt, add_generation_prompt=True)
            inputs = processor(
                text=final_input_prompt,
                images=sample['image_input'],
                return_tensors="pt",
                padding=True
            ).to(device)
        
            output = model(**inputs,output_attentions=True)
            logits = output.logits[0, -1] # get the logit of the last token
            attn = torch.stack(output.attentions).detach().cpu()
            
            attn_path_folder = os.path.join(
                "/scratch/workspace/jacksonmicha_umass_edu-attention_scores/attention_scores",
                out_dir, "attention_scores"
            )

            os.makedirs(attn_path_folder, exist_ok=True)
            attn_path = os.path.join(attn_path_folder, sample['id'] + ".pt")
            # async_save(attn, attn_path)


            color_idx, shape_idx = (1560, 1561)
            shape_range = (3, 1488)
            img_attn = attn[:, 0, :, -1,shape_range[0]:shape_range[1]].mean(dim=2)
            color_attn = attn[:, 0, :, -1, color_idx]
            shape_attn = attn[:, 0, :, -1, shape_idx]
            stacked = torch.stack([img_attn, color_attn, shape_attn])
            torch.save(stacked, attn_path)
            
            predicted_token_id = logits.argmax().item()
            predicted_token_str = processor.decode(predicted_token_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            temp = {
                "params": sample['params'],
                "text_input_type": sample.get("text_input_type", "caption"),
                "conflict_type": sample.get("conflict_type", "statement"),
                "predicted_token_id": predicted_token_id,
                "predicted_token_str": predicted_token_str,
                "attn_path": attn_path
            }
            
            if temp["predicted_token_id"] == image_color_token_id:
                temp["predicted_answer_alignment"] = "image"
            elif temp["predicted_token_id"] == text_color_token_id:
                temp["predicted_answer_alignment"] = "text"
                
            out_samples[sample['id']] = temp
            # break
    return out_samples


def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_args():
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=None,help='name of saved json')
    parser.add_argument('--dataset', type=str, default="shapes")
    parser.add_argument('--model', type=str, default="llava_one")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--shuffle',action='store_true')  # on/off flag
    parser.add_argument('--return_logits',action='store_true')  # on/off flag
    parser.add_argument('--reprocess',action='store_true')  # on/off flag
    parser.add_argument('--text_input_type', type=str, default="caption", help="caption, statement")
    parser.add_argument('--conflict_type', type=str, default="direct_alternative", help="aligned, direct_alternative, direct_negation, indirect_negation, indirect_alternative")
    parser.add_argument('--debug',action='store_true')  # on/off flag
    parser.add_argument('--subsets', type=int, default=None)

    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join("dataset/shape_dataset/cache", args.model, f"{args.text_input_type}_{args.conflict_type}")
    os.makedirs("results/"+args.output_dir, exist_ok=True)
    return args

#%%
def main():
    args = get_args()
    assert args.conflict_type in ["direct_aligned", "no_conflict", "direct_alternative", "direct_negation", "indirect_negation", "indirect_alternative"]
    assert args.text_input_type in ["caption", "statement"]
    
    print("Loading Dataset")
    dataset = load_dataset(args.dataset, args.subsets)
    print(dataset[0])
    
    print("Example Datapoint: ")
    print(dataset[0])
        
    print("Loading Model")
    model, processor, prepare_func, prompt_format_fn = load_model(args.model)

    print("Sanity Check the color tokens")
    COLOR_TOKEN_ID_DICT = {}
    for color in COLORS:
        print(f"Color: {color}")
        if args.model == "blip-2":
            color_word = f" {color}"
        else:
            color_word = color
        print(processor.tokenizer(color_word, add_special_tokens=False))
        COLOR_TOKEN_ID_DICT[color] = processor.tokenizer(color_word, add_special_tokens=False)['input_ids'][0]

    
        
    print("formatting prompts")
    print(f"Text input type: {args.text_input_type}")
    print(f"Conflict type: {args.conflict_type}")
    formatted_dataset = []
    for sample in dataset:
        new_sample = prompt_format_fn(sample, text_input_type=args.text_input_type, conflict_type=args.conflict_type)
        formatted_dataset.append(new_sample)
    print("Example formatted sample")
    print(formatted_dataset[0])
    
    print("running model")
    out_samples = extract_attention(formatted_dataset, model, processor, prepare_func, prompt_format_fn, args.output_dir, COLOR_TOKEN_ID_DICT)
    
    json_path = os.path.join("results", args.output_dir, "results.json")
    with open(json_path, 'w') as f:
        json.dump(out_samples,f)
        
    print("results saved to ", json_path)

    
    
    
if __name__ == "__main__":
    main()
