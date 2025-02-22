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
from huggingface_hub import hf_hub_download
from utils.train_utils import create_pretrained_tokenizer
import torch.nn as nn
import torchvision.models._utils as utils
from wrapper import ModuleListWrapper

# image = Image.open("assets/rar_generated_226.png").convert("RGB") 
# image_array = np.array(image)
# scipy.io.savemat("test_image.mat", {"test_image": image_array})
# print("PNG successfully converted to MAT file.")


# data_in = h5py.File('./test_image.mat', 'r')
# b = np.array(data_in['ans'],dtype='int').reshape(300)


datapath = './data'
s = 1
color_jitter = transforms.ColorJitter(
        0.9 * s, 0.9 * s, 0.9 * s, 0.1 * s)
flip = transforms.RandomHorizontalFlip()
Aug = transforms.Compose(
    [
    transforms.RandomResizedCrop(size=32),
    transforms.RandomApply([flip], p=0.5),
    transforms.RandomApply([color_jitter], p=0.9),
    transforms.RandomGrayscale(p=0.1)
    ])
data_transforms = transforms.Compose(
            [
                ToTensor(),
                Normalize(0.5, 0.5)
            ])
CIFAR_10_Dataset = torchvision.datasets.CIFAR10(datapath, train=True, download=False,
                                                 transform=data_transforms)

sublist = list(range(0, 2, 1))
subset = torch.utils.data.Subset(CIFAR_10_Dataset, sublist)
dataloader = torch.utils.data.DataLoader(subset, 1, shuffle=False, num_workers=2)


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
generator = demo_util.get_rar_generator(config)
tokenizer.to(device)
generator.to(device)



model = generator
model.load_state_dict(torch.load('./rar_xxl.bin', map_location=device))
print("The model is: ", model)
    

new_m = torchvision.models._utils.IntermediateLayerGetter(model, {'lm_head': 'feat1'})
# new_m = model.blocks[15].norm1
new_m = new_m.to(device)


model.blocks = ModuleListWrapper(model.blocks)

new_m = utils.IntermediateLayerGetter(model, {'blocks': 'feat1'})


print("new_m: ", new_m)

print('-------------------------------------------')

final1 = []

if __name__ == '__main__':
    for img, label in tqdm(iter(dataloader)):
        img = img.to(device)
        final = []

        for j in range(10):           
            out = new_m(Aug(img)) # to extract the output of a particular layer in the model
            for k, v in out.items():
                # v is the activation tensor, k is the layer name (here it is 'feat1')
                my = np.mean(v.reshape(256, 4).cpu().detach().numpy(), axis=1) # 256 neurons, and for each neuron 4 values (sub activation values)
                final.append(my)
        out1 = np.mean(np.array(final), axis=0)
        final1.append(out1)

    finalout = np.array(final1)
    maxout = np.max(finalout, axis=0)
    medianout = np.median(np.sort(finalout, axis=0)[0:-1], axis=0)
    selectivity = (maxout - medianout)/(maxout + medianout)
    scipy.io.savemat('./data/selectivity_unit.mat', {'selectivity': selectivity})
    
    print("filnalout = ", finalout)
    print("maxout = ", maxout)
    print("medianout = ", medianout)
    print("selectivity = ", selectivity)