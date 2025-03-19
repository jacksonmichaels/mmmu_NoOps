from random import random
import torch
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=12):
    # image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=False, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values
    

def call_llava_engine_df(args, sample, model, tokenizer=None, processor=None):
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle

    def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

    def deal_with_prompt(input_text, mm_use_im_start_end):
        qs = input_text
        if mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        return qs

    prompt = sample['final_input_prompt']
    prompt = deal_with_prompt(prompt, model.config.mm_use_im_start_end)
    conv = conv_templates['vicuna_v1'].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    image = sample['image']
    if image is not None:
        output_ids = model.generate(
            input_ids,
            images=image.unsqueeze(0).half().cuda(),
            do_sample=True,
            temperature=1,
            top_p=None,
            num_beams=5,
            max_new_tokens=128,
            use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        response = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    else:  # multiple images actually
        if sample['question_type'] == 'multiple-choice':
            all_choices = sample['all_choices']
            response = random.choice(all_choices)
        else:
            response = 'INVALID GENERATION FOR MULTIPLE IMAGE INPUTS'

    return response


def llava_image_processor(raw_image, vis_processors=None):
    image_tensor = vis_processors.preprocess(raw_image, return_tensors='pt')['pixel_values'][0]
    return image_tensor

# utils/model_utils.py

def call_hf_model_engine(args, sample, model, tokenizer, processor):
    keys = list(sample.keys())
    # for key in keys:
    #     print(key)
    #     print(sample[key])
    #     print("=========")
    device = next(model.parameters()).device
    # Prepare the inputs
    inputs = tokenizer(
        sample["final_input_prompt"],
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)
    if sample.get("image") is not None:
#         # If the model supports image inputs, include the processed image
#         inputs["pixel_values"] = sample["image"].unsqueeze(0)
#         inputs["image_sizes"] = torch.tensor([
#             inputs["pixel_values"].shape[-2], 
#             inputs["pixel_values"].shape[-1]
#         ])
        
#         # print("IMAGE: ", sample["image"].shape)
#         image = sample["image"]
#         sample["image"] = (image - image.min()) / (image.max() - image.min())

        inputs = processor(
            text=sample["final_input_prompt"], 
            images=sample["image"].shape, 
            return_tensors="pt")
        inputs["input_ids"] = inputs["input_ids"].to(device)
        inputs["attention_mask"] = inputs["attention_mask"].to(device)
        inputs["pixel_values"] = inputs["pixel_values"].to(device)
        inputs["image_sizes"] = inputs["image_sizes"].to(device) 
        print(inputs["input_ids"].shape)
        print(inputs["attention_mask"].shape)
        print(inputs["pixel_values"].shape)
        print(inputs["image_sizes"].shape)
    
    
    outputs = model.generate(input_ids = inputs['input_ids'].cuda(),
                              attention_mask = inputs['attention_mask'].cuda(),
                              pixel_values = inputs['pixel_values'].cuda(),
                              image_sizes = inputs['image_sizes'].cuda(),
                              max_new_tokens=256)
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# utils/model_utils.py

def hf_image_processor(image, tokenizer=None):
    from transformers import AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained(tokenizer.name_or_path)
    return processor(images=image, return_tensors="pt")["pixel_values"][0]

# utils/model_utils.py


def llava_image_processor(raw_image, processor):
    """
    Process images for the LLaVA model.
    
    Args:
        raw_image: Input image
        processor: LLaVA image processor
    
    Returns:
        torch.Tensor: Processed image tensor (pixel values)
    """
    if raw_image is None:
        return None
        
    # Process the image without accessing 'text'
    try:
        image_tensor = processor.image_processor(raw_image, return_tensors='pt')
        return image_tensor['pixel_values']
    except Exception as e:
        print(f"Error processing image: {e}")
        return None