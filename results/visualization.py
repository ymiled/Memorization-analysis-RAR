import numpy as np
import pandas as pd
import plotly.graph_objects as go


layers = [i for i in range(24*2+2)]

file_paths = [
    "rar_b_highest_memorizing_units_layer_order.txt",
    "rar_xl_highest_memorizing_units_layer_order.txt",
    "rar_xxl_highest_memorizing_units_layer_order.txt"
]

file_path = file_paths[0] 
data = np.loadtxt(file_path, delimiter=",", skiprows=1)

memorization_scores = data[:, 0]
layer_indices = data[:, 1].astype(int)
neuron_indices = data[:, 2].astype(int)

max_layer_index = len(layers) - 1
layer_indices = np.clip(layer_indices, 0, max_layer_index)

df = pd.DataFrame({
    "Neuron Index": neuron_indices,
    "Layer": [layers[i] for i in layer_indices],
    "Memorization Score": memorization_scores
})

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["Layer"],  
    y=df["Neuron Index"],  
    mode="markers",  
    marker=dict(
        size=8,  
        color=df["Memorization Score"],  
        colorscale="RdBu",  
        colorbar=dict(title="Memorization Score"),  
        showscale=True,  
        opacity=0.7,  
    ),
    showlegend=False  
))

fig.update_layout(
    xaxis_title="Layer",
    yaxis_title="Neuron Index",
    font=dict(family="Arial, sans-serif", size=23),  
    coloraxis_colorbar=dict(
        title="Memorization Score",
        titlefont=dict(size=23),  
        tickfont=dict(size=21)  
    ),
    template="plotly_white",
    margin=dict(l=80, r=80, t=100, b=80),
    height=700,  
    width=900,
)


fig.show()








