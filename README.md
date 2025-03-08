# Memorization-analysis-RAR

This is the repository for an implementation and analysis of memorization in MAR models
We clone the code from https://github.com/bytedance/1d-tokenizer , and we adapt the code from https://github.com/sprintml/LocalizingMemorizationInSSL to this purpose.

# Description of the code:

1) started.py: this script downloads the necessary pretrained models for the RAR generator, MaskGIT-VQ tokenizer and a sample from the Imagenet-1k-256 dataset. The script is a preliminary step for working with the RAR generator on ImageNet data.
  
2)  unit_mem_rar: This analyzes unit selectivity in the RAR model using activations from intermediate layers. It processes images from the sample from the Imagenet-1k-256 dataset, applies the data augmentation used during training, and computes selectivity metrics for each unit. The results help identify the most memorizing neurons, which are saved for further analysis. This also stores the data points that are responsible for the UnitMem score of a unit. Make sure to have model_xxx.bin and maskgit-vqgan-imagenet-f16-256.bin in the same folder.

3) Imagenet.py, sample_imagenet.py: utilities for the ImageNet dataset

4) results: folder with the results obtained related to memorization in neurons and data points that caused it.
  
5) data_points.py, results/visualization.py, results/memorization_comp_per_layer.py are utilities to visualize results. 
