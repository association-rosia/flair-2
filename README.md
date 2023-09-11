# 🛰️ FLAIR #2

<img src="assets/bandeau.jpg">

This project was made possible by our compute partners [2CRSI](https://2crsi.com/)
and [NVIDIA](https://www.nvidia.com/).

## 📋 Table of content

1. [🏁 Getting started](#start)
2. [⚙️ Train a model](#train)
3. [📝 Create a submission](#models)
4. [🔬 References](#references)
5. [📝 Citing](#citing)
6. [🛡️ License](#license)
7. [👨🏻‍💻 Contributors](contributors)

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
nohup python src/models/train_model.py \
    --arch DeepLabV3Plus \
    --encoder_name tu-tf_efficientnetv2_s \
    --learning_rate 0.001 \
    --sen_size 40 \
    --sen_temp_size 3 \
    --sen_temp_reduc median \
    --sen_list_bands 2 3 4 5 6 7 8 8a 11 12 \
    --prob_cover 10 \
    --batch_size 26 \
    --use_augmentation True \
    --class_weights 0.07451054458054185 0.07123414669165881 0.06501057431176234 0.10243128536707254 0.0751622868386753 0.060451925970421205 0.057084409075513015 0.0712831075581589 0.08115403779097626 0.05767359681290979 0.05792606455080904 0.0952665140613815 0.1308115063901194 \
    --seed 42 \
    --dry False \
 </dev/null &>/dev/null &
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
python src/models/predict_model.py -n {model checkpoint name}
```

## 🔬 References <a name="references"></a>

## 📝 Citing <a name="citing"></a>

## 🛡️ License <a name="license"></a>

Project is distributed under [MIT License](https://github.com/association-rosia/flair-2/blob/main/LICENSE)

## 👨🏻‍💻 Contributors <a name="contributors"></a>

Louis
REBERGA <a href="https://twitter.com/rbrgAlou"><img src="https://abs.twimg.com/favicons/twitter.3.ico" width="18px"/></a> <a href="https://www.linkedin.com/in/louisreberga/"><img src="https://static.licdn.com/sc/h/akt4ae504epesldzj74dzred8" width="18px"/></a> <a href="louis.reberga@gmail.com"><img src="https://www.google.com/a/cpanel/aqsone.com/images/favicon.ico" width="18px"/></a>

Baptiste
URGELL <a href="https://twitter.com/Baptiste2108"><img src="https://abs.twimg.com/favicons/twitter.3.ico" width="18px"/></a> <a href="https://www.linkedin.com/in/baptiste-urgell/"><img src="https://static.licdn.com/sc/h/akt4ae504epesldzj74dzred8" width="18px"/></a> <a href="baptiste.u@gmail.com"><img src="https://www.google.com/a/cpanel/aqsone.com/images/favicon.ico" width="18px"/></a> 
