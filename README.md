# Target Driven Visual Navigation

PyTorch implementation of our paper on target driven visual navigation in Ai2Thor environment. This implementation is modified based on [Learning to Learn how to Learn: Self-Adaptive Visual Navigation using Meta-Learning](https://github.com/allenai/savn)


## Prerequisite

1. The code has been implemented and tested on Ubuntu 16.04, python 3.5, PyTorch 0.4 and CUDA 10.1
2. For the rest of dependencies, please run `pip install -r requirements.txt`
3. Clone the repository as:
```
    git clone https://github.com/cassieqiuyd/MJOLNIR.git
```

## Data

The data and pretrained model can be found [here](https://drive.google.com/drive/folders/1i6V_t6TqaTpUdUFpOJT3y3KraJjak-sa?usp=sharing).(~ 15 GB)

Please unzip them (~30GB) and put both folders into the MJOLNIR folder.

## Evaluation

#### Pretrained model
```bash
python main.py --eval \
    --test_or_val test \
    --episode_type TestValEpisode \
    --load_model pretrained_models/mjolnir_o_pretrain.dat \
    --model mjolnir_o \
    --results_json mjolnir_o.json

cat mjolnir_o.json 
```

####  Random Agent
```bash
python main.py \
    --eval \
    --test_or_val test \
    --episode_type TestValEpisode \
    --title random_test \
    --agent_type RandomNavigationAgent \
    --results_json random_results.json
    
cat random_results.json
```

#### full evaluation
```bash
python full_eval.py \
    --title mjolnir \
    --model mjolnir_o \
    --results_json mjolnir_o.json \
    --gpu-ids 0
    
cat mjolnir_o.json
```

## Train

```bash
python main.py \
    --title mjolnir_train \
    --model mjolnir_o \
    --gpu-ids 0\
    --workers 8
```

## Visualization

go to visualization folder. If you did evaluation, the action log should be generated. 

```bash
python visualization.py --actionList ../saved_action_mjolnir_o_test.log
```


## Cite

```
@misc{qiu2020target,
    title={Target driven visual navigation exploiting object relationships},
    author={Yiding Qiu and Anwesan Pal and Henrik I. Christensen},
    year={2020},
    eprint={2003.06749},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}
```
