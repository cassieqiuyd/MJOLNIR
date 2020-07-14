# Target Driven Visual Navigation

### [Project](https://sites.google.com/eng.ucsd.edu/mjolnir) | [Paper](https://arxiv.org/pdf/2003.06749.pdf) | [Video](https://www.youtube.com/watch?v=aEbXjGZDZSQ)  <br>
PyTorch implementation of our paper on target driven visual navigation in Ai2Thor environment. This implementation is modified based on [Learning to Learn how to Learn: Self-Adaptive Visual Navigation using Meta-Learning](https://github.com/allenai/savn)


## Prerequisite

1. The code has been implemented and tested on Ubuntu 16.04, python 3.5, PyTorch 0.4 and CUDA 10.1
2. For the rest of dependencies, please run `pip install -r requirements.txt`
3. Clone the repository as:
```
    git clone https://github.com/cassieqiuyd/MJOLNIR.git
```

## Data

1. Pretrained models can be found [here]https://drive.google.com/drive/folders/1dHLbmKgVuDLoIPMb5V0lNFc17GWMvyFq?usp=sharing).

2. data: The data can be found [here](https://drive.google.com/drive/folders/1TNkjWVDbagTgFalvHrwR_TbH3YG9GMhM?usp=sharing).

Please update the locations accordingly in the config file.

## Evaluation

#### Pretrained model
```bash
python main.py --eval \
    --test_or_val test \
    --episode_type TestValEpisode \
    --load_model final_models/mjolnir_o_pre.dat \
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

#### Trained model
```bash
python full_eval.py \
    --title mjolnir \
    --model mjolnir_o \
    --results_json mjolnir_o.json \
    --gpu-ids 0 1
    
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
