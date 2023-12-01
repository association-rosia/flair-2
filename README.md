# 🛰️ FLAIR #2

<img src="assets/bandeau.jpg">

This project was made possible by our compute partners [2CRSI](https://2crsi.com/)
and [NVIDIA](https://www.nvidia.com/).

## 🖼️ Result example <a name="result-example"></a>

Aerial input image | Multi-class label | Multi-class pred
:--------------------:|:--------------------:|:--------------------:|
![](assets/aerial.png) | ![](assets/label.png) | ![](assets/pred.png)

View more results on the [WandB project](https://wandb.ai/association-rosia/flair-2).

## 🏛️ Model architecture <a name="model-architecture"></a>

<img src="assets/model-architecture.jpg">

## 🏁 Getting started <a name="start"></a>

1 - Create the conda environment:

```bash
conda env create -f environment.yml
```

2 - Activate the conda environment:

```bash
conda activate flair-2-env
```

3 - Install PyTorch and CUDA libraries:

```bash
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

## ⚙️ Train a model <a name="train"></a>

1 - Move to the project folder:

```bash
cd flair-2
```

2 - Activate the conda environment:

```bash
conda activate flair-2-env
```

3 - Launch the model training in background:

```bash
nohup python src/models/train_model.py <hyperparams args> </dev/null &>/dev/null &
```

(Bonus) - Kill the background process:

```bash
pkill -f 'python src/models/train_model.py'
```

## 📝 Create a submission <a name="submission"></a>

1 - Move to the project folder:

```bash
cd flair-2
```

3 - Activate the conda environment:

```bash
conda activate flair-2-env
```

4 - Launch the model inference:

```bash
python src/models/predict_model.py -n {model.ckpt}
```

## 🔬 References <a name="references"></a>

Chen, Liang-Chieh, et al. « Rethinking Atrous Convolution for Semantic Image Segmentation ». arXiv.Org, 17 juin 2017, https://arxiv.org/abs/1706.05587v3.

Garioud, Anatol, et al. « FLAIR #2: Textural and Temporal Information for Semantic Segmentation from Multi-Source Optical Imagery ». arXiv.Org, 23 mai 2023, https://doi.org/10.13140/RG.2.2.30938.93128/1.

Iakubovskii, Pavel. Segmentation Models Pytorch. GitHub, 2019, https://github.com/qubvel/segmentation_models.pytorch.

Xie, Enze, et al. « SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers ». arXiv.Org, 31 mai 2021, https://arxiv.org/abs/2105.15203v3.

## 📝 Citing

```
@misc{RebergaUrgell:2023,
  Author = {Louis Reberga and Baptiste Urgell},
  Title = {Flair #2},
  Year = {2023},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/association-rosia/flair-2}}
}
```

## 🛡️ License

Project is distributed under [MIT License](https://github.com/association-rosia/flair-2/blob/main/LICENSE)

## 👨🏻‍💻 Contributors <a name="contributors"></a>

Louis
REBERGA <a href="https://twitter.com/rbrgAlou"><img src="https://abs.twimg.com/favicons/twitter.3.ico" width="18px"/></a> <a href="https://www.linkedin.com/in/louisreberga/"><img src="https://static.licdn.com/sc/h/akt4ae504epesldzj74dzred8" width="18px"/></a> <a href="louis.reberga@gmail.com"><img src="https://www.google.com/a/cpanel/aqsone.com/images/favicon.ico" width="18px"/></a>

Baptiste
URGELL <a href="https://twitter.com/Baptiste2108"><img src="https://abs.twimg.com/favicons/twitter.3.ico" width="18px"/></a> <a href="https://www.linkedin.com/in/baptiste-urgell/"><img src="https://static.licdn.com/sc/h/akt4ae504epesldzj74dzred8" width="18px"/></a> <a href="baptiste.u@gmail.com"><img src="https://www.google.com/a/cpanel/aqsone.com/images/favicon.ico" width="18px"/></a> 
