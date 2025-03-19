#!/usr/bin/env python3

import sys
import os
import json
import torch
import random
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image
import re
import ast
from pathlib import Path
# Transformers imports
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
)
from transformers import LlavaNextForConditionalGeneration  # for "LLAVA" model_type
from transformers import LlavaOnevisionForConditionalGeneration
from transformers import AutoModelForCausalLM
# --------------------------------------------------------------------------------
# 1) MODEL DICTIONARY (INCLUDING LLaVA-Next Interleave Qwen-7B)
# --------------------------------------------------------------------------------
CACHE_DIR_BASE = "/u/li19/data_folder/model_cache"

MODEL_DICT_LLMs = {
    "idefics": {
        "model_id": "HuggingFaceM4/idefics2-8b-base",
        "cache_dir": CACHE_DIR_BASE,
        "model_type": "AutoModelForVision2Seq",
    },
    "llava_next": {
        "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "cache_dir": CACHE_DIR_BASE,
        "model_type": "LLAVA",
    },
    "llava_one": {
        "model_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "cache_dir": CACHE_DIR_BASE,
        "model_type": "llava_one",
    },
    # Additional existing or custom models can go here...
}

# --------------------------------------------------------------------------------
# 2) LOAD MMMU DATASET
# --------------------------------------------------------------------------------
# We map short subject names to full subject strings. Adjust as needed for MMMU.
CAT_SHORT2LONG = {
    "math": "Mathematics",
    "science": "Science",
    "history": "History",
    "chemistry": "Chemistry",
    "biology": "Biology",
    # etc. Fill in all relevant MMMU subjects
}

all_subs = ['Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science', 'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology']

def load_dataset(data_path, subject, split="validation"):
    """
    Loads the MMMU dataset from Hugging Face by subject and split.
    Adjust if your MMMU dataset has different naming.
    """
    # Example: data_path="MMMU/MMMU" -> load_dataset("MMMU/MMMU", subject=..., split=...)
    from datasets import load_dataset
    ds = load_dataset(data_path, subject, split=split)
    return ds

def load_noop(data_path, subject, split= "validation", noop_root = "/u/li19/MMMU/mmmu-noop"):
    with open(os.path.join(noop_root, f"mmmu_{split}_noop.json"), 'r') as f:
        noop_data = json.load(f)
    ds = load_dataset(data_path, subject, split)
    for row in ds:
        noop = noop_data[row['id']]
        row['noop'] = noop
    return ds
        
        
def concatenate_datasets(sub_dataset_list):
    """
    Concatenate multiple subject splits into a single dataset.
    """
    from datasets import concatenate_datasets
    combined = concatenate_datasets(sub_dataset_list)
    return combined

# --------------------------------------------------------------------------------
# 3) HELPER FUNCTIONS
# --------------------------------------------------------------------------------
def save_json(path, data):
    path = Path(path)
    parent = path.parent.absolute()
    if not os.path.exists(parent):
        os.makedirs(parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_yaml(config_path):
    import yaml
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
    """
    Perform any necessary conversions for an MMMU example.
    If an image is referenced in the sample, open it here.
    """
    # Example: if 'image_path' is in the sample, load the actual image
    if "image_path" in sample and os.path.isfile(sample["image_path"]):
        img = Image.open(sample["image_path"]).convert("RGB")
        sample["image"] = img
    return sample

def construct_prompt(sample, config):
    """
    Build final_input_prompt using the sample's question.
    Adjust this logic to match your MMMU usage.
    """
    question = sample.get("question", "")
    # If there's some config-driven format, you can incorporate it here
    # For now, we do a simple "Q: ... A:" style
    sample["final_input_prompt"] = f"Question: {question}\nAnswer:"
    return sample

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--output_path", type=str, default="output.json")
    parser.add_argument("--config_path", type=str, default="configs/llava1.5.yaml")
    parser.add_argument(
        "--data_path",
        type=str,
        default="MMMU/MMMU",
        help="Hugging Face dataset path for MMMU"
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="Key in MODEL_DICT_LLMs (e.g. 'llava_next_interleave_qwen_7b')")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--access_token", type=str, default=None,
                        help="If model is private, specify your HF token here.")
    parser.add_argument("--noop", action='store_true')
    parser.add_argument("--subject", type=str, default="all")
    return parser.parse_args()

def set_device_and_seed(seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    return device

def load_config(config_path):
    print(f"Loading config from {config_path} ...")
    config = load_yaml(config_path)
    for k, v in list(config.items()):
        if k != "eval_params" and isinstance(v, list) and len(v) == 1:
            config[k] = v[0]
    return config

def prepare_NoOp(data_path, split, all_subs):
    print(f"Loading MMMU dataset from {data_path} for split={split} ...")
    sub_dataset_list = [load_noop(data_path, subject, split=split) for subject in all_subs]
    dataset = concatenate_datasets(sub_dataset_list)
    print("NoOp Dataset loaded:", dataset)
    return dataset

def prepare_dataset(data_path, split, all_subs):
    print(f"Loading MMMU dataset from {data_path} for split={split} ...")
    sub_dataset_list = [load_dataset(data_path, subject, split=split) for subject in all_subs]
    dataset = concatenate_datasets(sub_dataset_list)
    print("Dataset loaded:", dataset)
    return dataset

def load_model_and_tokenizer(args, device):
    if args.model_name not in MODEL_DICT_LLMs.keys():
        raise ValueError(f"Model '{args.model_name}' not found in MODEL_DICT_LLMs.")

    model_info = MODEL_DICT_LLMs[args.model_name]
    model_id = model_info["model_id"]
    cache_dir = model_info["cache_dir"]
    model_type = model_info.get("model_type", "AutoModel")

    print(f"Loading model '{args.model_name}' from '{model_id}' ...")

    if model_type == "LLAVA":
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            use_auth_token=args.access_token,
            trust_remote_code=True
        ).to(device)

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            use_auth_token=args.access_token,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(
            model_id,
            use_auth_token=args.access_token,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
    elif model_type == "llava_one":
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
                use_auth_token=args.access_token,
                trust_remote_code=True,  # Qwen-based often needs this
            ).to(device)
        tokenizer = None
        processor = AutoProcessor.from_pretrained(
            model_id,
            use_auth_token=args.access_token,
            cache_dir=cache_dir,
            trust_remote_code=True
        )

    else:
        tokenizer, processor, model = handle_model_types(model_type, model_id, cache_dir, args, device)

    print("Model loaded successfully!")
    return model, tokenizer, processor

def handle_model_types(model_type, model_id, cache_dir, args, device):
    if model_type == "AutoModelForCausalLM":
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_auth_token=args.access_token, cache_dir=cache_dir
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, use_auth_token=args.access_token, cache_dir=cache_dir
        ).to(device)
        processor = None
    else:
        # Fallback
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_auth_token=args.access_token, cache_dir=cache_dir
        )
        model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, use_auth_token=args.access_token, cache_dir=cache_dir
        ).to(device)
        processor = None
    return tokenizer, processor, model
    
def string_to_dict(string):
    content = []
    image_placeholder = {"type": "image"}
    image_tokens = []  # To store image tokens in order

    # Regular expression to match <image_n> where n is 0 to 10
    pattern = r"<image (?:10|[0-9])>"

    # Find all image tokens in the string
    matches = re.findall(pattern, string)
    image_tokens.extend(matches)

    # Split the string by image tokens
    split_question = re.split(pattern, string)

    for idx, segment in enumerate(split_question):
        if segment.strip():  # Add text segment if not empty
            content.append({"type": "text", "text": segment})
        if idx < len(split_question) - 1:
            content.append(image_placeholder)

    return content, image_tokens

def construct_prompt_l1(sample):
    """
    Build final_input_prompt using the sample's question.
    Adjust this logic to match your MMMU usage.
    """
    question = sample.get("question", "")
    image_names = [f"image_{idx}" for idx in range(1,8)]
    
    images = [sample[name] for name in image_names 
              if sample[name] is not None 
              and (name.replace("_", " ") in sample["question"] or name.replace("_", " ") in sample["options"])]

    sample["images"] = []
    # Build conversation history for multi-modal prompts
    conversation = []
    image_placeholder = {"type": "image"}
    
    # Regular expression to match <image_n> where n is 0 to 10
    pattern = r"<image (?:10|[0-9])>"
    # pattern = "r<www>"
    # Split the string
    split_question = re.split(pattern, question)
    
    if sample['question_type'] == "multiple-choice":
        instruction = "you are an advanced logical AI who is tasked to answer all questions accurately and briefly. For the following multiple choice questions please only reply with a single letter to show your answer choice, more than a single letter responses will be considered incorrect"
    else:
        instruction = "you are an advanced logical AI who is tasked to answer all questions accurately and briefly. For the following questions answer as briefly as possible. A single term or integer if possible. Do not explain yourself, do not include calculations only include your final answer to the question. Overly verbose answers will be marked as incorrect."
        
        option_text = f""
    
    content = [
    {"type": "text", "text": instruction}
    ]
    
    
    # print(sample["question"])
    question_dict, question_images = string_to_dict(sample["question"])
    # print(question_dict)
    content += question_dict
    sample["images"] += question_images
    
    if sample['question_type'] == "multiple-choice":
        answers = ast.literal_eval(sample['options'])
        ops = 'ABCDEFGHIJKL'
        
        content.append( {"type": "text", "text": "Your options are: "})
        for i, ans in enumerate(answers):
            content.append( {"type": "text", "text": ops[i] + ") "})
            answer_dict, answer_imgs = string_to_dict(ans)
            sample["images"] += answer_imgs
            content += answer_dict
    else:
        option_text = " "        
        
    response = {
        "role": "user",
        "content": content
    }
    
    sample["final_input_prompt"] = [response]

    return sample

    

def parse_llava_1v(response):
    text = response[0]
    ans = text.split("\n")[-1]
    return ans

def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Basic multiple choice parsing:
    We check if any choice string is in the response.
    If none matched, return the full response or a fallback.
    """
    resp_lower = response.lower()
    for choice in all_choices:
        if choice.lower() in resp_lower:
            return choice
    # If no direct match, fallback:
    return response

def hf_image_processor(image):
    """
    If needed for certain model pipelines, you could do extra image processing here.
    For standard HF usage, we typically use AutoProcessor directly in call_hf_model_engine.
    """
    return image

# --------------------------------------------------------------------------------
# 4) THE INFERENCE ENGINE
# --------------------------------------------------------------------------------
def call_llava_one_engine(args, sample, model, processor):
    """
    Takes a sample with 'final_input_prompt', possibly an 'image', and
    runs inference using the LlavaOnevisionForConditionalGeneration model.
    """
    device = next(model.parameters()).device
    prompt_text = sample.get("final_input_prompt")
    prompt = processor.apply_chat_template(prompt_text, add_generation_prompt=True)

    for i in range(len(sample['images'])):
        if isinstance(sample['images'][i], str):
            sample['images'][i] = sample[sample['images'][i].replace(" ", "_").replace("<", "").replace(">","")]
    
    # Prepare inputs using the processor
    inputs = processor(
        text=[prompt],
        images=sample['images'],
        return_tensors="pt",
        padding=True
    )
        
    for k in inputs:
        inputs[k] = inputs[k].to(device)

    with torch.no_grad():
        # We can simply feed images in the order they have to be used in the text prompt
        generate_ids = model.generate(**inputs, max_new_tokens=300, pad_token_id=processor.tokenizer.eos_token_id)
        responses = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return responses


def run_inference_old(dataset, model, processor, args):
    samples = []
    print("Processing Dataset")
    for sample in tqdm(dataset):
        sample = process_single_sample(sample)
        sample = construct_prompt_l1(sample)
        samples.append(sample)

    out_samples = {}
    model.eval()
    print("Running inference on MMMU dataset...")
    for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
        try:
            response = call_llava_one_engine(args, sample, model, processor)
            pred_ans = parse_llava_1v(response)
            if i % 250 == 0:
                print(sample)
                print(response)
                print("PRED: ", pred_ans, " REAL: ", sample["answer"])

            sample_id = sample.get("id", str(hash(sample.get("question", ""))))
            out_samples[sample_id] = pred_ans
        except Exception as e:
            print("Failed on sample: \n ============== \n", sample)
            print(e)
            print(traceback.format_exc())
            break
    return out_samples

subjects = []

def run_inference(dataset, model, processor, args):
    samples = []
    print("Processing Dataset")
    for sample in tqdm(dataset):
        sample = process_single_sample(sample)
        sample = construct_prompt_l1(sample)
        samples.append(sample)

    out_samples_eval_only = {}
    out_samples_parse_and_eval = {}
    model.eval()
    print("Running inference on MMMU dataset...")
    for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
        try:
            response = call_llava_one_engine(args, sample, model, processor)
            pred_ans = parse_llava_1v(response)

            # Evaluation-only mode output
            sample_id = sample.get("id", str(hash(sample.get("question", ""))))
            out_samples_eval_only[sample_id] = pred_ans

            # Parse-and-evaluation mode output
            parsed_sample = {
                "id": sample_id,
                "question_type": sample["question_type"],
                "answer": sample.get("answer", ""),  # Ground-truth answer
                "response": pred_ans,  # Predicted answer
            }

            if sample["question_type"] == "multiple-choice":
                parsed_sample["all_choices"] = ["A", "B", "C", "D"]  # Example choices
                parsed_sample["index2ans"] = sample.get("index2ans", {})  # Choices with full answers

            # Organize by subject (if applicable)
            # subject = sample.get("subject", "General")
            subject = "_".join(sample.get("id").split("_")[1:-1])
            if subject not in subjects:
                print("NEW SUBJECT: ", subject)
                subjects.append(subject)
            if subject not in out_samples_parse_and_eval:
                out_samples_parse_and_eval[subject] = []
            out_samples_parse_and_eval[subject].append(parsed_sample)

            if i % 250 == 0:
                print(sample)
                # print(response)
                print("PRED: ", pred_ans, " REAL: ", sample["answer"])

        except Exception as e:
            print("Failed on sample: \n ============== \n", sample)
            print(e)
            print(traceback.format_exc())
            break

    return out_samples_eval_only, out_samples_parse_and_eval

def save_results(output_path, eval_only_results, parse_and_eval_results):
    # Save evaluation-only results
    eval_only_path = os.path.join(output_path, "eval_only_output.json")
    save_json(eval_only_path, eval_only_results)
    print(f"Evaluation-only results saved to {eval_only_path}")

    # Save parse-and-evaluation results
    parse_and_eval_path = os.path.join(output_path, "parse_and_eval")
    os.makedirs(parse_and_eval_path, exist_ok=True)
    for subject, subject_data in parse_and_eval_results.items():
        subject_path = os.path.join(parse_and_eval_path, subject)
        os.makedirs(subject_path, exist_ok=True)
        output_file = os.path.join(subject_path, "output.json")
        save_json(output_file, subject_data)
    print(f"Parse-and-evaluation results saved to {parse_and_eval_path}")


# --------------------------------------------------------------------------------
# Main Function
# --------------------------------------------------------------------------------

def main():
    args = parse_arguments()
    if args.subject == "all":
        subs = all_subs
    else:
        subs = [args.subject]
    device = set_device_and_seed(args.seed)
    args.config = load_config(args.config_path)
    if args.noop:
        dataset = prepare_NoOp(args.data_path, args.split, subs)
    else:
        dataset = prepare_dataset(args.data_path, args.split, subs)
    model, tokenizer, processor = load_model_and_tokenizer(args, device)
    eval_only_results, parse_and_eval_results = run_inference(dataset, model, processor, args)
    save_results(args.output_path, eval_only_results, parse_and_eval_results)


if __name__ == "__main__":
    main()

