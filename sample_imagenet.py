import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose, Normalize, ToPILImage, CenterCrop, Resize
from Imagenet import Imagenet
import random
from torchvision import transforms
import pickle
import torch


def sample_imagenet(imagenet_datapath, save_path="data/imagenet_subset.pkl", seed=1):
    random.seed(seed)
    torch.manual_seed(seed)

    # Resizing images to 256x256 for VQ tokenizer compatibility
    resize_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    IMAGENET_Dataset = Imagenet(
        imagenet_datapath, transform=resize_transform
    )

    total_indices = list(range(len(IMAGENET_Dataset)))
    nb_samples = int(len(total_indices))
    subset_indices = random.sample(total_indices, nb_samples)

    subset = torch.utils.data.Subset(IMAGENET_Dataset, subset_indices)
    dataloader = torch.utils.data.DataLoader(subset, batch_size=1, num_workers=7)
    
    print(f"Subset size: {len(subset)} samples")
    
    with open(save_path, 'wb') as f:
        pickle.dump(dataloader, f)
    return dataloader


def load_dataloader(save_path="data/imagenet_subset.pkl"):
    with open(save_path, 'rb') as f:
        dataloader = pickle.load(f)
        print(f"DataLoader loaded from {save_path}")
    return dataloader
