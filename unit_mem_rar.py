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
from sample_imagenet import load_dataloader

s = 1
color_jitter = transforms.ColorJitter(
        0.9 * s, 0.9 * s, 0.9 * s, 0.1 * s
    )

flip = transforms.RandomHorizontalFlip()

# Aug = transforms.Compose(
#         [
#         transforms.RandomResizedCrop(size=256),
#         transforms.RandomApply([flip], p=0.5),
#         transforms.RandomApply([color_jitter], p=0.9),
#         transforms.RandomGrayscale(p=0.1)
#         ]
#     )

Aug = transforms.Compose([
    transforms.CenterCrop(size=256),  
    transforms.RandomHorizontalFlip(p=0.5),
])

imagenet_datapath = './data/1k-imagenet'
dataloader = load_dataloader(save_path="data/imagenet_subset.pkl")

rar_model_size = ["rar_b", "rar_l", "rar_xl", "rar_xxl"][0]
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


def compute_utilities(layer): 
    final1 = []    
    for img, label in tqdm(iter(dataloader)):
        final = []        
        img = img.to(device)
        label = label.to(device)
        for j in range(10):
            with torch.no_grad():
                tokens = tokenizer.encode(Aug(img))
                _, intermediates = model(tokens, condition=label, return_intermediates=True, stop_at=layer)
                if (len(intermediates) != layer + 1):
                    print(f"Layer {layer} not found in the model")
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
    maxposition = np.argmax(finalout, axis=0)  # index of data points with max activation for each unit in this layer
    medianout = np.median(np.sort(finalout, axis=0)[0:-1], axis=0)
    selectivity = (maxout - medianout) / (maxout + medianout)
    
    with open(f'results/{rar_model_size}_maxposition_per_layer.txt', 'a') as f_maxposition:
        f_maxposition.write(f"Max position for layer {layer}:\n")
        np.savetxt(f_maxposition, maxposition, delimiter=',')
        f_maxposition.write("\n\n")

    with open(f'results/{rar_model_size}_selectivities.txt', 'a') as f_selectivities:
        f_selectivities.write(f"Selectivity for layer {layer}:\n")
        np.savetxt(f_selectivities, selectivity, delimiter=',')
        f_selectivities.write("\n\n")
    
    return finalout, maxout, maxposition, medianout, selectivity



def top_memorizing_neurons(nb_layers, total_neurons, top_percent, layer=None):
    top_mem_neurons = []
    maxposition_per_layer = []
    if layer is not None:
        layers = [layer]
    else:
        layers = range(nb_layers)
        
    for layer in layers:
        finalout, maxout, maxposition, medianout, selectivity = compute_utilities(layer)
        maxposition_per_layer.append(maxposition)
        # print(maxposition_layer)
        
        for neuron_idx, selec in enumerate(selectivity):
            candidate = (selec, layer, neuron_idx)

            if len(top_mem_neurons) < top_percent * total_neurons:  
                top_mem_neurons.append(candidate)  
                top_mem_neurons.sort(reverse=True, key=lambda x:x[0])  # we keep the list sorted
            elif top_mem_neurons[-1][0]:
                top_mem_neurons.append(candidate)
                top_mem_neurons.sort(reverse=True, key=lambda x:x[0])  
                top_mem_neurons.pop()
        
        print(f"maximum selectivity for layer {layer}: {np.max(selectivity)}, which corresponds to neuron {np.argmax(selectivity)}")

    
    with open(f'results/{rar_model_size}_highest_memorizing_units_unitmem_order.txt', 'a') as f:
        f.write("Top memorizing neurons:\n")
        np.savetxt(f, sorted(top_mem_neurons, reverse=True, key=lambda x:x[0]), delimiter=',')
        f.write("\n\n")
    
    with open(f'results/{rar_model_size}_highest_memorizing_units_layer_order.txt', 'a') as f:
        f.write("Top memorizing neurons:\n")
        np.savetxt(f, sorted(top_mem_neurons, reverse=True, key=lambda x:x[1]), delimiter=',')
        f.write("\n\n")        

    
    return top_mem_neurons


if __name__ == '__main__':
    nb_layers = config.model.generator.num_hidden_layers * 2 + 2
    total_neurons = 12898 # b : 12898 xl: 17026 xxl : 21154
    top_mem_neurons = top_memorizing_neurons(nb_layers, total_neurons, 0.1)
    print("top 10% memorizing neurons (unitmem decreasing order):", sorted(top_mem_neurons, reverse=True, key=lambda x:x[0]))
    print("top 10% memorizing neurons (layer decreasing order) :", sorted(top_mem_neurons, reverse=True, key=lambda x:x[1]))

    # nb_layers = config.model.generator.num_hidden_layers * 2 + 2
    # total_neurons = 0
    # for l in range(nb_layers):
    #     img, label = next(iter(dataloader))
    #     img = img.to(device)
    #     label = label.to(device)
    #     for j in range(1):
    #         with torch.no_grad():
    #             tokens = tokenizer.encode(Aug(img))
    #             _, intermediates = model(tokens, condition=label, return_intermediates=True, stop_at=l)
    #             out = intermediates[l]
    #             total_neurons += out.size(1)
    #             print(total_neurons)
    # print(total_neurons)
    
    # selectivities = compute_utilities(layer=35)[4]
    # fig = px.histogram(x=selectivities, nbins=10)
    # fig.update_layout(
    #     xaxis_title="Selectivity",
    #     yaxis_title="Frequency",
    #     plot_bgcolor="white",  
    #     paper_bgcolor="white", 
    #     font=dict(  # Set global font size
    #     size=50  # Increase font size
    #     ),
    #     xaxis=dict(
    #         title_font=dict(size=50), 
    #         tickfont=dict(size=45)     
    #     ),
    #     yaxis=dict(
    #         title_font=dict(size=50),  
    #         tickfont=dict(size=45)    
    #     )
    # )
    # fig.show()