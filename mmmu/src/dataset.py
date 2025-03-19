import os
import json
from datasets import load_dataset, concatenate_datasets
from PIL import Image

def load_dataset_mmmu(data_path, subject, split="validation"):
    ds = load_dataset(data_path, subject, split=split)
    return ds

def load_noop(data_path, subject, split="validation", noop_root="/u/li19/MMMU/mmmu-noop", noOps='text'):
    ds = load_dataset(data_path, subject, split=split)
    
    if noOps in ["text", "all"]:
        noop_file = os.path.join(noop_root, f"mmmu_{split}_noop_insert_sentence.json")
        with open(noop_file, 'r') as f:
            noop_data = json.load(f)
    if noOps in ["img", "all"]:
        img_noop_file = os.path.join(noop_root, f"val_img_NoOp_metadata.json")
        with open(img_noop_file, 'r') as f:
            img_data = json.load(f)
    
    def add_noop(example, noOps):
        example_id = example['id']
        if noOps in ["text", "all"]:
            example['noop'] = noop_data.get(example_id, None)
        if noOps in ["img", "all"]:
            example['noop_imgs'] = True
            img_datum = img_data[example_id]['image_paths']
            image_names = [f"image_{n}" for n in range(1, 8)]
            imgs = [(name, example[name]) for name in image_names if example[name] is not None]
            for img_pair, imgNoOp in zip(imgs, img_datum):
                name, _ = img_pair
                example[name] = Image.open(imgNoOp + ".png")
        return example

    ds = ds.map(lambda x: add_noop(x, noOps))
    return ds

def concatenate_datasets_list(sub_dataset_list):
    return concatenate_datasets(sub_dataset_list)

def prepare_NoOp(data_path, split, subjects, noOps='text'):
    print(f"Loading MMMU dataset from {data_path} for split={split} with NoOp...")
    sub_dataset_list = [load_noop(data_path, subject, split=split, noOps=noOps) for subject in subjects]
    dataset = concatenate_datasets_list(sub_dataset_list)
    print("NoOp Dataset loaded.")
    return dataset

def prepare_dataset(data_path, split, subjects):
    print(f"Loading MMMU dataset from {data_path} for split={split} ...")
    sub_dataset_list = [load_dataset_mmmu(data_path, subject, split=split) for subject in subjects]
    dataset = concatenate_datasets_list(sub_dataset_list)
    print("Dataset loaded.")
    return dataset
