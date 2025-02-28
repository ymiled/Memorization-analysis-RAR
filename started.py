
import torch
from PIL import Image
import numpy as np
import demo_util
from huggingface_hub import hf_hub_download
from utils.train_utils import create_pretrained_tokenizer
from sample_imagenet import sample_imagenet

rar_model_size = ["rar_b", "rar_l", "rar_xl", "rar_xxl"][3]

hf_hub_download(repo_id="fun-research/TiTok", filename=f"maskgit-vqgan-imagenet-f16-256.bin", local_dir="./")

hf_hub_download(repo_id="yucornetto/RAR", filename=f"{rar_model_size}.bin", local_dir="./")

# imagenet_datapath = './data/tiny-imagenet-200'
# dataloader = sample_imagenet(imagenet_datapath, percent=0.0015)