# 🛰️ FLAIR #2

<img src="assets/bandeau.jpg">

This project was made possible by our compute partners [2CRSI](https://2crsi.com/) and [NVIDIA](https://www.nvidia.com/).

## 🏁 Getting started

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

## ⚙️ Train a model 

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
nohup python src/models/train_model.py </dev/null &>/dev/null &
```

(Bonus) - Kill the background process:
```bash
pkill -f 'python src/models/train_model.py'
```

## 📝 Create a submission

1 - Edit `run_name` in [predict_model.py](src%2Fmodels%2Fpredict_model.py).

2 - Move to the project folder:
```bash
cd flair-2
```

3 - Activate the conda environment:
```bash
conda activate flair-2-env
```

4 - Launch the model inference:
```bash
python src/models/predict_model.py
```

## Contributors

Louis REBERGA <a href="https://twitter.com/rbrgAlou"><img src="https://abs.twimg.com/favicons/twitter.3.ico" width="18px"/></a> <a href="https://www.linkedin.com/in/louisreberga/"><img src="https://static.licdn.com/sc/h/akt4ae504epesldzj74dzred8" width="18px"/></a> <a href="louis.reberga@gmail.com"><img src="https://www.google.com/a/cpanel/aqsone.com/images/favicon.ico" width="18px"/></a>

Baptiste URGELL <a href="https://twitter.com/Baptiste2108"><img src="https://abs.twimg.com/favicons/twitter.3.ico" width="18px"/></a> <a href="https://www.linkedin.com/in/baptiste-urgell/"><img src="https://static.licdn.com/sc/h/akt4ae504epesldzj74dzred8" width="18px"/></a> <a href="baptiste.u@gmail.com"><img src="https://www.google.com/a/cpanel/aqsone.com/images/favicon.ico" width="18px"/></a> 


