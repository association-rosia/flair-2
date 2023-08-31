# FLAIR #2

<img src="assets/bandeau.jpg">

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

1 - Connect to a GPU server using SSH.

2 - Move to the project folder:
```bash
cd flair-2
```

3 - Activate the conda environment:
```bash
conda activate flair-2-env
```

4 - Launch the model training in background:
```bash
nohup python src/models/train_model.py </dev/null &>/dev/null &
```

(Bonus) - Kill the background process:
```bash
pkill -f 'python src/models/train_model.py'
```

## Create a submission

1 - Edit `run_name` in [predict_model.py](src%2Fmodels%2Fpredict_model.py).

2 - Launch the model inference:
```bash
python src/models/predict_model.py
```


