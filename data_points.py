import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from sample_imagenet import load_dataloader
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

file_path = "results/rar_b_highest_memorizing_units_layer_order.txt"
data = np.loadtxt(file_path, delimiter=",", skiprows=1)

memorization_scores = data[:, 0]
layer_indices = data[:, 1].astype(int)
neuron_indices = data[:, 2].astype(int)

df = pd.DataFrame({
    "Neuron Index": neuron_indices,
    "Layer": layer_indices,
    "Memorization Score": memorization_scores
})


top_10_memorizing_units = neuron_indices
top_10_memorizing_units_layers_idx = layer_indices


def extract_data_from_txt(file_path):
    data = {}
    current_layer = -1
    layer_data = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()  
            if line.startswith("Max position for layer"):
                current_layer = int(line.split()[-1].strip(":"))
                if current_layer not in data:
                    data[current_layer] = []
    return data


file_path = "results/rar_xxl_maxposition_per_layer.txt"
data_images = extract_data_from_txt(file_path)



memorized_data_points = []

for i in range(len(top_10_memorizing_units)):
    unit = top_10_memorizing_units[i]
    layer_idx = layer_indices[i]

    data_point = int(data_images[layer_idx][unit])

    memorized_data_points.append(data_point)


dataloader_imagenet = load_dataloader(save_path="data/imagenet_subset.pkl")

labels_map = {}
i = 0
for img, label in dataloader_imagenet:
    labels_map[i] = int(label)  
    i += 1
    
data_point_counts = Counter(memorized_data_points)

for data_point, count in sorted(data_point_counts.items(), key=lambda x: x[1], reverse=True):
    if data_point in labels_map:
        print(f"Data Point {data_point}: {count} times, label: {labels_map[data_point]}")
    else:
        print(f"Data Point {data_point}: {count} times, Image: Not found")






device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.eval().to(device)

features = []
image_paths = []

with torch.no_grad():
    for imgs, _ in dataloader_imagenet:
        imgs = imgs.to(device)
        feats = model(imgs).squeeze() 
        features.append(feats.cpu().numpy())

features = np.vstack(features) 
features = features.reshape(features.shape[0], -1) 


# k-NN to detect outliers (rare images)
knn = NearestNeighbors(n_neighbors=10)
knn.fit(features)

distances, _ = knn.kneighbors(features)
mean_distances = distances[:, 1:].mean(axis=1)  

outlier_indices = np.argsort(mean_distances)[-20:]

print(outlier_indices)


top_most_common = data_point_counts.most_common(10)
top_most_common_indices = [idx for idx, _ in top_most_common]
# see if the top most common data points are also outliers
outliers_in_top_common = set(top_most_common_indices).intersection(outlier_indices)
print("outliers in top common: ", outliers_in_top_common)


# Example data (replace with real counts)
data = {
    "Data point": [data_point for data_point, _ in sorted(data_point_counts.items(), key=lambda x: x[1], reverse=True)],
    "Frequency": [count for _, count in sorted(data_point_counts.items(), key=lambda x: x[1], reverse=True)]
}

df = pd.DataFrame(data)

fig = px.bar(df, x="Data point", y="Frequency", color="Frequency", 
             color_continuous_scale="viridis")

fig.update_layout(
    xaxis=dict(title_font=dict(size=20), tickfont=dict(size=30)),  
    yaxis=dict(title_font=dict(size=20), tickfont=dict(size=30)),  
    coloraxis_colorbar=dict(title_font=dict(size=30)),
    font=dict(family="Arial", size=30) 
)


fig.show()