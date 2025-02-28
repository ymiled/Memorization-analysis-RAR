import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose, Normalize, ToPILImage, CenterCrop, Resize
from TinyImagenet import TinyImageNet
from collections import defaultdict
import random
from torchvision import transforms
from tqdm import tqdm
import pickle

def sample_imagenet(imagenet_datapath, percent=0.01):
    save_path = "data/imagenet_subset.pkl"
    
    # resizing images to 256x256 to be compatible with the VQ tokenizer : 
    resize_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    IMAGENET_Dataset = TinyImageNet(
        imagenet_datapath, split="train", transform=resize_transform
    )

    # Selecting only 0.1% of the dataset
    
    total_indices = list(range(len(IMAGENET_Dataset)))
    nb_samples = int(percent * len(total_indices))
    subset_indices = random.sample(total_indices, nb_samples)

    subset = torch.utils.data.Subset(IMAGENET_Dataset, subset_indices)
    dataloader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=True, num_workers=7)
    
    print(f"Subset size: {len(subset)} samples")
    
    with open(save_path, 'wb') as f:
        pickle.dump(dataloader, f)
    return dataloader


def load_dataloader(save_path="data/imagenet_subset.pkl"):
    with open(save_path, 'rb') as f:
        dataloader = pickle.load(f)
        print(f"DataLoader loaded from {save_path}")
    return dataloader


