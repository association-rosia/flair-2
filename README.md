# FLAIR #2 

<img src="assets/bandeau.jpg">

# Getting started
Create the conda environment
```bash
conda env create -f environment.yml
```

Activate the conda environment
```bash
conda activate flair-2-env
```

Install PyTorch and CUDA libraries 
```bash
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```