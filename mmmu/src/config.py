import os

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
    "llava_one_0.5b": {
        "model_id": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        "cache_dir": CACHE_DIR_BASE,
        "model_type": "llava_one",
    },
    "internvl2_5": {
        "model_id": "OpenGVLab/InternVL2_5-8B",
        "cache_dir": CACHE_DIR_BASE,
        "model_type": "internvl2_5",
    },
    # Additional models...
}

CAT_SHORT2LONG = {
    "math": "Mathematics",
    "science": "Science",
    "history": "History",
    "chemistry": "Chemistry",
    "biology": "Biology",
    # etc.
}

all_subs = [
    'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art',
    'Art_Theory','Basic_Medical_Science','Biology','Chemistry','Clinical_Medicine',
    'Computer_Science','Design','Diagnostics_and_Laboratory_Medicine','Economics',
    'Electronics','Energy_and_Power','Finance','Geography','History','Literature',
    'Manage','Marketing','Materials','Math','Mechanical_Engineering','Music',
    'Pharmacy','Physics','Psychology','Public_Health','Sociology'
]
