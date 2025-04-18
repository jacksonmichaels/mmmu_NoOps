"""
This script is used to extract hidden states from the model.
"""
#%%
import torch
import os
import random
import re
import ast
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import importlib
import json
from datasets import concatenate_datasets, load_dataset
from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
from PIL import Image
# Set grad enabled to False
torch.set_grad_enabled(False)
from functools import partial
import pickle
import gc
from llava_verbose_attn import *
#%%
def load_noop(data_path, subject, split="validation", noop_root="/jet/home/billyli/mmmu_NoOps/mmmu-noop", noOps='text'):
    from datasets import load_dataset
    # Load your dataset
    ds = load_dataset(data_path, subject, split=split)
    
    if noOps == "text" or noOps == "all":
        # Load the NoOp data from JSON
        noop_file = os.path.join(noop_root, f"mmmu_{split}_noop_insert_sentence.json")
        with open(noop_file, 'r') as f:
            noop_data = json.load(f)
        
    if noOps == "img" or noOps == "all":
        # Load the NoOp data from JSON
        img_noop_file = os.path.join(noop_root, f"val_img_NoOp_metadata.json")
        with open(img_noop_file, 'r') as f:
            img_data = json.load(f)
    
    # Define a function that adds "noop" to each example
    def add_noop(example, noOps, noOp_root="/jet/home/billyli/data_folder/NoOpImgs/mmmu_NoOp_imgs"):
        example_id = example['id']
        if noOps == "text" or noOps == "all":
            # print("adding text noop")
            example['noop'] = noop_data.get(example_id, None)  # fallback if ID not found
        if noOps == "img" or noOps == "all":
            example['noop_imgs'] = True
            img_datum = img_data[example_id]['image_paths']
            image_names = [f"image_{n}" for n in range(1, 8)]
            imgs = [(name, example[name]) for name in image_names if example[name] is not None]
            for img_pair, imgNoOp in zip(imgs, img_datum):
                name, img = img_pair
                noOp_path = os.path.join(noOp_root, example_id, name + ".png") 
                # example[name] = Image.open(imgNoOp + ".png")
                example[name] = Image.open(noOp_path)
                # print("adding img noop: ", name)
        return example

    # Use .map() to apply the function and return a new dataset with the `noop` column
    ds = ds.map(lambda x: add_noop(x, noOps))

    return ds

def prepare_NoOp(data_path, split, all_subs, noOps='text'):
    print(f"Loading MMMU dataset from {data_path} for split={split} ...")
    sub_dataset_list = [load_noop(data_path, subject, split=split, noOps=noOps) for subject in tqdm(all_subs)]
    dataset = concatenate_datasets(sub_dataset_list)
    print("NoOp Dataset loaded:", dataset)
    return dataset

def prepare_dataset(data_path, split, all_subs):
    print(f"Loading MMMU dataset from {data_path} for split={split} ...")
    sub_dataset_list = [load_dataset(data_path, subject, split=split) for subject in tqdm(all_subs)]
    dataset = concatenate_datasets(sub_dataset_list)
    print("Dataset loaded:", dataset)
    return dataset

def load_dataset(noop='all'):
    all_subs = [
    'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art',
    'Art_Theory','Basic_Medical_Science','Biology','Chemistry','Clinical_Medicine',
    'Computer_Science','Design','Diagnostics_and_Laboratory_Medicine','Economics',
    'Electronics','Energy_and_Power','Finance','Geography','History','Literature',
    'Manage','Marketing','Materials','Math','Mechanical_Engineering','Music',
    'Pharmacy','Physics','Psychology','Public_Health','Sociology'
    ]
    # all_subs = ['Accounting', 'Art']
    data_path = "MMMU/MMMU"
    split = "validation"
    subs = all_subs
    if noop != "none":
        dataset = prepare_NoOp(data_path, split, subs, noop)
    else:
        dataset = prepare_dataset(data_path, split, subs)
    return dataset

def load_model(model_name, device="cuda:0"):
    model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager"
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    model.eval()
    return processor, model


def find_target_segments(x, target=151646):
    """
    Returns a tensor of shape [n, 2], where each row [start, end]
    indicates that x[start:end] is a contiguous block with value `target`.
    """
    # Create a boolean mask where elements equal target.
    mask = (x == target)
    
    # If no elements match the target, return an empty tensor.
    if not mask.any():
        return torch.empty((0, 2), dtype=torch.int64, device=x.device)
    
    # Convert the boolean mask to integers (0 or 1) to compute differences.
    mask_int = mask.to(torch.int)
    # Compute differences between consecutive elements.
    diff = mask_int[1:] - mask_int[:-1]
    
    # A diff value of 1 indicates a transition from False to True (start of a segment)
    # A diff value of -1 indicates a transition from True to False (end of a segment)
    start_indices = torch.nonzero(diff == 1).flatten() + 1
    end_indices = torch.nonzero(diff == -1).flatten() + 1
    
    # Edge case: if the first element is True, then a segment starts at index 0.
    if mask[0]:
        start_indices = torch.cat((torch.tensor([0], device=x.device), start_indices))
    
    # Edge case: if the last element is True, then a segment continues to the end.
    if mask[-1]:
        end_indices = torch.cat((end_indices, torch.tensor([len(x)], device=x.device)))
    
    # Stack start and end indices into a [n,2] tensor.
    segments = torch.stack((start_indices, end_indices), dim=1)
    return segments

def find_inserted_section_indices(A, B):
    # A and B are 1D tensors.
    lenA = A.size(0)
    lenB = B.size(0)
    min_len = min(lenA, lenB)
    
    # Compute prefix equality for the first min_len elements.
    prefix_eq = (A[:min_len] == B[:min_len])
    # Find first index where they differ.
    if prefix_eq.all():
        prefix_match_length = min_len
    else:
        # nonzero returns a 2D tensor; we take the first index.
        prefix_match_length = torch.nonzero(~prefix_eq, as_tuple=False)[0].item()
    
    # Compute suffix equality by reversing the tensors.
    revA = A.flip(0)
    revB = B.flip(0)
    min_len_rev = min(lenA, lenB)
    suffix_eq = (revA[:min_len_rev] == revB[:min_len_rev])
    if suffix_eq.all():
        suffix_match_length = min_len_rev
    else:
        suffix_match_length = torch.nonzero(~suffix_eq, as_tuple=False)[0].item()
    
    # The inserted section in B is located between the matching prefix and the matching suffix.
    start_index = prefix_match_length
    end_index = lenB - suffix_match_length
    return torch.tensor([start_index, end_index])

def sample_to_model_inputs(sample, question, processor):
    for i in range(len(sample['images'])):
        if isinstance(sample['images'][i], str):
            sample['images'][i] = sample[
                sample['images'][i].replace(" ", "_").replace("<", "").replace(">","")
            ]
    # response = [{"role": "user", "content": question}]
    response = question

    prompt = processor.apply_chat_template(response, add_generation_prompt=True)

    inputs = processor(
        text=prompt,
        images=sample['images'],
        return_tensors="pt",
        padding=True
    )
    return inputs

def final_selection_mask(B, inserted_indices, target=151646):
    """
    Given a tensor B, inserted_indices (a tuple (start, end) of the inserted section),
    and a target integer value, return a boolean mask selecting tokens that are neither:
      - equal to the target, nor
      - in the inserted section.
      
    Parameters:
      B (torch.Tensor): 1D tensor of tokens.
      inserted_indices (tuple): A tuple (start, end) indicating the inserted segment indices.
      target (int): The target value to exclude.
      
    Returns:
      torch.BoolTensor: A boolean tensor of the same length as B.
    """
    ins_start, ins_end = inserted_indices
    # Create an index tensor [0, 1, ..., len(B)-1]
    indices = torch.arange(B.size(0), device=B.device)
    
    # Build the mask:
    #   - B != target ensures tokens are not equal to the target.
    #   - (indices < ins_start) | (indices >= ins_end) ensures the token is not in the inserted section.
    mask = (B != target) & ((indices < ins_start) | (indices >= ins_end))
    del indices
    return mask

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
    parser.add_argument('--noop', type=str, default="text")
    parser.add_argument('--model', type=str, default="llava_one")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--shuffle',action='store_true')  # on/off flag
    parser.add_argument('--return_logits',action='store_true')  # on/off flag
    parser.add_argument('--reprocess',action='store_true')  # on/off flag
    parser.add_argument('--debug',action='store_true')  # on/off flag
    parser.add_argument('--subsets', type=int, default=None)
    parser.add_argument('--store_all',action='store_true')  # on/off flag

    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join("dataset/shape_dataset/cache", args.model)
    os.makedirs("results/"+args.output_dir, exist_ok=True)
    return args
    
def process_single_sample(sample):
    if "image_path" in sample and os.path.isfile(sample["image_path"]):
        img = Image.open(sample["image_path"]).convert("RGB")
        sample["image"] = img
    return sample


def string_to_dict(string):
    content = []
    image_placeholder = {"type": "image"}
    image_tokens = []
    pattern = r"<image (?:10|[0-9])>"
    matches = re.findall(pattern, string)
    image_tokens.extend(matches)
    split_question = re.split(pattern, string)
    for idx, segment in enumerate(split_question):
        if segment.strip():
            content.append({"type": "text", "text": segment})
        if idx < len(split_question) - 1:
            content.append(image_placeholder)
    return content, image_tokens
    
def construct_prompt_l1(sample):
    if "noop" in sample.keys():
        question = sample.get("noop", "")
    else:
        question = sample.get("question", "")

    for idx in range(1,8):
        name = f"image_{idx}"
        images = []
        if name in sample and sample[name] is not None:
            images.append(sample[name])
            
    sample["images"] = []
    conversation = []
    image_placeholder = {"type": "image"}

    pattern = r"<image (?:10|[0-9])>"
    
    question = sample.get("question", "")
    split_question = re.split(pattern, question)
    if sample['question_type'] == "multiple-choice":
        instruction = ("you are an advanced logical AI who is tasked to answer all questions accurately and briefly. "
                       "For multiple choice questions, please only reply with a single letter for your answer choice.")
    else:
        instruction = ("you are an advanced logical AI who is tasked to answer all questions accurately and briefly. "
                       "For open questions, reply as succinctly as possible with the final answer only.")

    if sample['noop_warning']:
        warn_text = "This might be a trick question designed to confuse LLMs with additional information. Look for irrelevant information or distractors in the question:"
        instruction += "\n" + warn_text

    content = [{"type": "text", "text": instruction}]
    question_dict, question_images = string_to_dict(question)
    content += question_dict
    sample["images"] += question_images

    if sample['question_type'] == "multiple-choice":
        answers = ast.literal_eval(sample['options'])
        ops = 'ABCDEFGHIJKL'
        content.append({"type": "text", "text": "Your options are: "})
        for i, ans in enumerate(answers):
            content.append({"type": "text", "text": ops[i] + ") "})
            answer_dict, answer_imgs = string_to_dict(ans)
            sample["images"] += answer_imgs
            content += answer_dict

    response = {
        "role": "user",
        "content": content
    }
    sample["final_input_prompt_base"] = [response]

    
    noop_sample = sample.copy()
    question = noop_sample.get("noop", "")
    split_question = re.split(pattern, question)
    if noop_sample['question_type'] == "multiple-choice":
        instruction = ("you are an advanced logical AI who is tasked to answer all questions accurately and briefly. "
                       "For multiple choice questions, please only reply with a single letter for your answer choice.")
        # instruction = ("you are an advanced logical AI who is tasked to answer all questions accurately. "
        #                "please fully explain your reasoning steps behind your answer and end your reply with your final choice in brackets.")
    else:
        instruction = ("you are an advanced logical AI who is tasked to answer all questions accurately and briefly. "
                       "For open questions, reply as succinctly as possible with the final answer only.")

    if noop_sample['noop_warning']:
        warn_text = "This might be a trick question designed to confuse LLMs with additional information. Look for irrelevant information or distractors in the question:"
        instruction += "\n" + warn_text

    content = [{"type": "text", "text": instruction}]
    question_dict, question_images = string_to_dict(question)
    content += question_dict
    # sample["images"] += question_images

    if noop_sample['question_type'] == "multiple-choice":
        answers = ast.literal_eval(noop_sample['options'])
        ops = 'ABCDEFGHIJKL'
        content.append({"type": "text", "text": "Your options are: "})
        for i, ans in enumerate(answers):
            content.append({"type": "text", "text": ops[i] + ") "})
            answer_dict, answer_imgs = string_to_dict(ans)
            # sample["images"] += answer_imgs
            content += answer_dict

    response = {
        "role": "user",
        "content": content
    }
    sample["final_input_prompt_noop"] = [response]
    return sample
    
def extract_attention(dset, model, processor, out_dir, store_all=False, layers=None, device="cuda:0"):
    out_samples = {}
    layers = model.language_model.model.layers

    with torch.no_grad():
        with tqdm(dset) as pbar:
            for sample in pbar:
                try:
                    # Prepare inputs
                    base = sample_to_model_inputs(sample, sample['final_input_prompt_base'], processor)
                    noop = sample_to_model_inputs(sample, sample['final_input_prompt_noop'], processor)
                    noop_indices = find_inserted_section_indices(base.input_ids[0], noop.input_ids[0])
                    img_segments = find_target_segments(noop.input_ids[0])
                    final_mask = final_selection_mask(noop.input_ids[0], noop_indices)

                    # Set up attention context
                    for layer in layers:
                        layer.self_attn.noop_indices = noop_indices
                        layer.self_attn.img_segments = img_segments
                        layer.self_attn.not_noop_mask = final_mask

                    # Memory before inference
                    mem_before = torch.cuda.memory_allocated(device) / (1024**2)  # MB

                    # Run inference
                    output = model(**noop.to(device), output_attentions=False)

                    # Memory after inference
                    mem_after = torch.cuda.memory_allocated(device) / (1024**2)  # MB

                    # Update progress bar
                    pbar.set_description(f"Mem: {mem_before:.1f}MB → {mem_after:.1f}MB")

                    # Process outputs
                    logits = output.logits.detach().cpu()[0, -1]
                    top_logit = torch.argmax(logits)
                    attn = torch.stack([layer.self_attn.attn.detach().cpu() for layer in layers])

                    attn_path_folder = os.path.join(out_dir, "attention_scores")
                    os.makedirs(attn_path_folder, exist_ok=True)
                    attn_path = os.path.join(attn_path_folder, sample['id'] + ".pt")

                    if not store_all:
                        torch.save(attn, attn_path)

                    predicted_token_id = logits.argmax().item()
                    predicted_token_str = processor.decode(predicted_token_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    pred_ans = processor.batch_decode([top_logit])

                    temp = {
                        "id": sample['id'],
                        "predicted_ans": pred_ans,
                        "real_ans": sample['answer'],
                        "attn_path": attn_path,
                        "is_correct": pred_ans == sample['answer']
                    }

                    out_samples[sample['id']] = temp

                except Exception as e:
                    print(f"\n[ERROR] Sample ID: {sample.get('id', 'UNKNOWN')}")
                    print(f"Sample content: {sample}")
                    print("Exception Traceback:")
                    output, attn, logits = None
                    traceback.print_exc()

                finally:
                    # del base, noop, output, attn, logits
                    # gc.collect()
                    # torch.cuda.empty_cache()

    return out_samples
#%%
def main():
    args = get_args()
    device = "cuda:0"
    # assert args.conflict_type in ["direct_aligned", "no_conflict", "direct_alternative", "direct_negation", "indirect_negation", "indirect_alternative"]
    # assert args.text_input_type in ["caption", "statement"]
    
    print("Loading Dataset")
    print(args.noop)
    dataset = load_dataset(args.noop)
    
    print("Example Datapoint: ")
    print(dataset[0])
         
    print("Loading Model")
    processor, model = load_model(args.model, device)
    layers = model.language_model.model.layers
    for idx, layer in enumerate(layers):
        # funcType = type(layer.self_attn.forward)
        # print(layer)
        # print(layer.self_attn)
        layer.self_attn.forward = types.MethodType(custom_Qwen2Attention_forward, layer.self_attn)
        layer.self_attn.attn = None
        
    print("formatting prompts")
    formatted_dataset = []
    for sample in dataset:
        sample['noop_warning'] = False
        new_sample = process_single_sample(sample)
        new_sample = construct_prompt_l1(new_sample)
        formatted_dataset.append(new_sample)
    print("Example formatted sample")
    print(formatted_dataset[0])
    
    print("running model")
    
    
    out_samples = extract_attention(
        formatted_dataset, 
        model, 
        processor, 
        args.output_dir, 
    )
    
    json_path = os.path.join("results", args.output_dir, "results.json")
    with open(json_path, 'w') as f:
        json.dump(out_samples,f)
        
    print("results saved to ", json_path)

    
    
    
if __name__ == "__main__":
    main()
