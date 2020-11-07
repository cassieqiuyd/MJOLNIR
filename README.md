# Learning hierarchical relationships for object-goal navigation

Yiding Qiu*, Anwesan Pal* and Henrik I. Christensen


PyTorch implementation of our paper on target driven visual navigation in Ai2Thor environment. This implementation is modified based on [Learning to Learn how to Learn: Self-Adaptive Visual Navigation using Meta-Learning](https://github.com/allenai/savn)


## Prerequisite

1. The code has been implemented and tested on Ubuntu 16.04, python 3.5, PyTorch 0.4 and CUDA 10.1
2. (Recommended) Create a virtual environment using virtualenv or conda:
```
virtualenv mjolnir_env --python=python3.5
source mjolnir_env/bin/activate
``` 

```
conda create -n mjolnir_env python=3.5
conda activate mjolnir_env
```

3. For the rest of dependencies, please run `pip install -r requirements.txt --ignore-installed`
4. Clone the repository as:
```
    git clone https://github.com/cassieqiuyd/MJOLNIR.git
```

Note: Upon running any code the first time, the AI2THOR 3D scenes will be downloaded (~500MB) to your home directory. 

## Data

The offline data can be found [here](https://drive.google.com/drive/folders/1i6V_t6TqaTpUdUFpOJT3y3KraJjak-sa?usp=sharing).

"data.zip" (~5 GB) contains everything needed for evalution. Please unzip it and put it into the MJOLNIR folder.

For training, please also download "train.zip" (~9 GB), and put all "Floorplan" folders into `MJOLNIR/data/thor_v1_offline_data`




## Evaluation

Note: if you are not using gpu, you can remove the argument "--gpu-ids 0"

#### Evaluate Pretrained model
```bash
python main.py --eval \
    --test_or_val test \
    --episode_type TestValEpisode \
    --load_model pretrained_models/mjolnir_o_pretrain.dat \
    --model MJOLNIR_O \
    --results_json mjolnir_o.json \
    --gpu-ids 0
```
This should also generate a log of the actions taken by the agents (required for visualization)

To print the evaluation results,

```   
cat mjolnir_o.json 
```

## Visualization

If you did evaluation, the action log should be generated. 

```bash
cd visualization
python visualization.py --actionList ../saved_action_mjolnir_o_test.log
```


## Train

```bash
python main.py \
    --title mjolnir_train \
    --model MJOLNIR_O \
    --gpu-ids 0\
    --workers 8
    --vis False
    --save-model-dir trained_models
```

#### Full evaluation on training
```bash
python full_eval.py \
    --title mjolnir \
    --model MJOLNIR_O \
    --results_json mjolnir_o.json \
    --gpu-ids 0
    --save-model-dir trained_models
    
cat mjolnir_o.json
```