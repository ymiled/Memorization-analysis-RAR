# Memorization-analysis-RAR

This is the repository for an implementation and analysis of memorization in MAR models

# Description of the code:

1) This script downloads the necessary pretrained models for the RAR generator and MaskGIT-VQ tokenizer. The script is a preliminary step for working with the RAR generator on ImageNet data.
  
2)  unit_mem_rar: This analyzes neuron selectivity in the RAR model using activations from intermediate layers. It processes images from the Tiny-ImageNet dataset, applies data augmentation, and computes selectivity metrics for neurons. The results help identify the most memorizing neurons, which are saved for further analysis. This also gives stores the data points that are responsible for the UnitMem score of a neuron. Make sure to have model_r and in the same folder.

3) TinyImagenet.py, sample_imagenet.py: utilities for the Tiny ImageNet dataset

4) results: folder with the results obtained related to memorization in neurons and data points that caused it.
  
5) data_points.py, results/visualization.py, results/memorization_comp_per_layer.py are utilities to visualize results. 
