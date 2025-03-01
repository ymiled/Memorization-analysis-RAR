import numpy as np
import pandas as pd

def categorize_layer(layer):
    if layer < 15:
        return "Early Layer"
    elif layer < 32:
        return "Mid Layer"
    else:
        return "Final Layer"

layer_order = ["Early Layer", "Mid Layer", "Final Layer"]

file_path = "rar_xxl_highest_memorizing_units_layer_order.txt"
data = np.loadtxt(file_path, delimiter=",", skiprows=1)

memorization_scores = data[:, 0]  
neuron_indices = data[:, 2].astype(int)  
layer_indices = data[:, 1].astype(int)  

df = pd.DataFrame({
    "Neuron Index": neuron_indices,
    "Layer": layer_indices,
    "Memorization Score": memorization_scores
})

neurons_per_layer_count = df.groupby("Layer")["Neuron Index"].nunique()
total_top_neurons = len(df)
percentage_per_layer = (neurons_per_layer_count / total_top_neurons) * 100
result = pd.DataFrame({
    "Layer_Type": neurons_per_layer_count.index,
    "Number of Neurons": neurons_per_layer_count.values,
    "Percentage_of_Top_Memorizing_Neurons": percentage_per_layer.values
})

result = result.reset_index(drop=True)  # Resetting index for cleaner display
result = result.sort_values(by="Percentage_of_Top_Memorizing_Neurons", ascending=False).reset_index(drop=True)

result = result.assign(Layer_Type = lambda x: np.where(x.Layer_Type < 15, "First Layer", np.where(x.Layer_Type < 32, "Mid Layer", "Final Layer")))
result = result.groupby("Layer_Type").sum().reset_index()
result = result.assign(Percentage_of_Top_Memorizing_Neurons = lambda x: x.Percentage_of_Top_Memorizing_Neurons / x.Percentage_of_Top_Memorizing_Neurons.sum() * 100)
result = result.sort_values(by="Percentage_of_Top_Memorizing_Neurons", ascending=False).reset_index(drop=True)

print(result)


# ------------------------------------------------------------------------------------------------------------------------------ #


file_path = "rar_xxl_highest_memorizing_units_layer_order.txt"
data = np.loadtxt(file_path, delimiter=",", skiprows=1)


memorization_scores = data[:, 0]
layer_indices = data[:, 1].astype(int)
neuron_indices = data[:, 2].astype(int)

# Create DataFrame
df = pd.DataFrame({
    "Neuron Memorization Score": memorization_scores,
    "Layer": layer_indices,
    "Neuron Index": neuron_indices
})

max_memorization_per_layer = df.groupby("Layer")["Neuron Memorization Score"].max().reset_index()
max_memorization_df = df.merge(max_memorization_per_layer, on=["Layer", "Neuron Memorization Score"])
max_memorization_df["Layer_Type"] = max_memorization_df["Layer"].apply(categorize_layer)
sorted_df = max_memorization_df.groupby("Layer_Type", as_index=False).max(numeric_only=True)

sorted_df["Layer_Type"] = pd.Categorical(sorted_df["Layer_Type"], categories=layer_order, ordered=True)
sorted_df = sorted_df.sort_values(by="Layer_Type").reset_index(drop=True)

print(sorted_df[["Layer_Type", "Neuron Index", "Neuron Memorization Score"]])
