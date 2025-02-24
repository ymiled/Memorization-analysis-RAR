import scipy.io
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision
from torchvision import models, transforms
from torchvision.transforms import ToTensor, Compose, Normalize, ToPILImage, CenterCrop, Resize
from tqdm import tqdm
from PIL import Image
import numpy as np
import demo_util
from utils.train_utils import create_pretrained_tokenizer
import torch.nn as nn
import torchvision.models._utils as utils
import plotly.express as px
import plotly.graph_objects as go

datapath = './data'
s = 1
color_jitter = transforms.ColorJitter(
        0.9 * s, 0.9 * s, 0.9 * s, 0.1 * s)
flip = transforms.RandomHorizontalFlip()
Aug = transforms.Compose(
    [
    transforms.RandomResizedCrop(size=256),
    transforms.RandomApply([flip], p=0.5),
    transforms.RandomApply([color_jitter], p=0.9),
    transforms.RandomGrayscale(p=0.1)
    ])
data_transforms = transforms.Compose(
            [
                ToTensor(),
                Normalize(0.5, 0.5)
            ])

# resizing the image to 256x256 to be compatible with the VQ tokenizer : 
resize_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Image_net = torchvision.datasets.ImageNet(
#     datapath, download=True, transform=resize_transform
# )


CIFAR_10_Dataset = torchvision.datasets.CIFAR10(
    datapath, train=True, download=True, transform=resize_transform
)



sublist = list(range(0, 2, 1))
subset = torch.utils.data.Subset(CIFAR_10_Dataset, sublist)
dataloader = torch.utils.data.DataLoader(subset, 1, shuffle=False, num_workers=2)

cifar_to_imagenet = {
    6: 30,
    9: 555
}



rar_model_size = ["rar_b", "rar_l", "rar_xl", "rar_xxl"][3]
config = demo_util.get_config("configs/training/generator/rar.yaml")
config.experiment.generator_checkpoint = f"{rar_model_size}.bin"
config.model.generator.hidden_size = {"rar_b": 768, "rar_l": 1024, "rar_xl": 1280, "rar_xxl": 1408}[rar_model_size]
config.model.generator.num_hidden_layers = {"rar_b": 24, "rar_l": 24, "rar_xl": 32, "rar_xxl": 40}[rar_model_size]
config.model.generator.num_attention_heads = 16
config.model.generator.intermediate_size = {"rar_b": 3072, "rar_l": 4096, "rar_xl": 5120, "rar_xxl": 6144}[rar_model_size]


device = "cuda" if torch.cuda.is_available() else "cpu"

# maskgit-vq as tokenizer
tokenizer = create_pretrained_tokenizer(config)
model = demo_util.get_rar_generator(config)
tokenizer.to(device)
model.to(device)


if __name__ == '__main__':
    nb_layers = 42
    max_selectivity_per_layer = []
    for layer in range(nb_layers):
        final1 = []      
        for img, label in tqdm(iter(dataloader)):
            final = []        
            img = img.to(device)
            label = torch.tensor([cifar_to_imagenet[label.item()]]).to(device)
            for j in range(10):
                with torch.no_grad():
                    tokens = tokenizer.encode(Aug(img))
                    _, intermediates = model(tokens, condition=label)
                    out = intermediates[layer]
                    activations = np.mean(out.reshape(out.size(1), out.size(2)).cpu().detach().numpy(), axis=1)
                    # we take the absolute value of the activations because some of them like Gelu have negative values
                    # and we are interested in the magnitude of the activations
                    activations = np.abs(activations)
                    final.append(activations)
                
            out1 = np.mean(np.array(final), axis=0)

            final1.append(out1)
        
        finalout = np.array(final1)
        maxout = np.max(finalout, axis=0)
        medianout = np.median(np.sort(finalout, axis=0)[0:-1], axis=0)
        selectivity = (maxout - medianout) / (maxout + medianout)
        
        print(f"maximum selectivity for layer {layer}: {np.max(selectivity)}, which corresponds to neuron {np.argmax(selectivity)}")
        max_selectivity_per_layer.append(np.max(selectivity))
    
    
