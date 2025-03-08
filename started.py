
import torch
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download
from utils.train_utils import create_pretrained_tokenizer
from sample_imagenet import *
from datasets import load_dataset
from torch.utils.data import DataLoader
import random
import os

# rar_model_size = ["rar_b", "rar_l", "rar_xl", "rar_xxl"][0]

# hf_hub_download(repo_id="fun-research/TiTok", filename=f"maskgit-vqgan-imagenet-f16-256.bin", local_dir="./")

# hf_hub_download(repo_id="yucornetto/RAR", filename=f"{rar_model_size}.bin", local_dir="./")


imagenet_datapath = './data/1k-imagenet'

os.makedirs(imagenet_datapath, exist_ok=True)

dataset = load_dataset("evanarlian/imagenet_1k_resized_256", split="train")

random.seed(1)

sampled_indices = random.sample(range(len(dataset)), 1000)

for i, idx in enumerate(sampled_indices):
    example = dataset[idx]
    image = example["image"]
    label = example["label"]

    image.save(os.path.join(imagenet_datapath, f"image_{i}_label_{label}.jpg"))

    print(f"Saved image {i + 1}/1000: image_{i}_label_{label}.jpg")

print(f"All 1000 images saved to {imagenet_datapath}")

dataloader = sample_imagenet(imagenet_datapath=imagenet_datapath, save_path="data/imagenet_subset.pkl", seed=1)


