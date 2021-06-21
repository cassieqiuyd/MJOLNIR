# Learning hierarchical relationships for object-goal navigation

Yiding Qiu*, Anwesan Pal* and Henrik I. Christensen

[Paper](https://arxiv.org/abs/2003.06749) | [Website](https://sites.google.com/eng.ucsd.edu/mjolnir) | [Video](https://www.youtube.com/watch?v=eCxWwohbOd8)

PyTorch implementation of our CoRL 2020 paper **Learning hierarchical relationships for object-goal navigation** in AI2-THOR environment. This implementation is modified based on [SAVN](https://github.com/allenai/savn).


## Prerequisite

1. The code has been implemented and tested on Ubuntu 18.04, python 3.6, PyTorch 0.6 and CUDA 10.1
2. (Recommended) Create a virtual environment using virtualenv or conda:
```
virtualenv mjolnir_env --python=python3.6
source mjolnir_env/bin/activate
``` 

```
conda create -n mjolnir_env python=3.6
conda activate mjolnir_env
```

3. For the rest of dependencies, please run `pip install -r requirements.txt --ignore-installed`
4. Clone the repository as:
```
    git clone https://github.com/cassieqiuyd/MJOLNIR.git
```

Note: Upon running any code the first time, the AI2-THOR 3D scenes will be downloaded (~500MB) to your home directory. 

## Data

The offline data can be found [here](https://drive.google.com/drive/folders/1i6V_t6TqaTpUdUFpOJT3y3KraJjak-sa?usp=sharing).

"data.zip" (~5 GB) contains everything needed for evalution. Please unzip it and put it into the MJOLNIR folder.

For training, please also download "train.zip" (~9 GB), and put all "Floorplan" folders into `MJOLNIR/data/thor_v1_offline_data`

## Knowledge Graph construction

1. Run the following script to generate the list of iThor v1.0.1 objects.
```
python kg_prep/object_list.py
```
This will generate a .txt file in `MJOLNIR/kg_prep/kg_data/`.

2. Download the [Visual Genome](https://visualgenome.org/) relationships file.
```
wget https://visualgenome.org/static/data/dataset/relationships.json.zip -P kg_prep/kg_data/
unzip kg_prep/kg_data/relationships.json.zip -d kg_prep/kg_data/ && rm -rf kg_prep/kg_data/relationships.json.zip
```

3. Download the relationship aliases.
```
wget https://visualgenome.org/static/data/dataset/relationship_alias.txt -P kg_prep/kg_data/
```

4. Run the relationship denoising step which will prune the object relationships of Visual-Genome dataset for the objects in iThor.
```
python kg_prep/kg_denoising.py
```

This will generate several files containing the statistics of the knowledge graph.

5. Generate adjacency matrix.
```
python kg_prep/make_adjacency.py
```

This will generate the adjacency matrix file `adjmat.dat` file inside `MJOLNIR/kg_prep/kg_data/`. This is the same file which can be found inside `MJOLNIR/data/gcn` folder downloaded in the Data section.

Note: If you want the adjacency matrix to be weighted, pass the `--weighted` argument to the above command line.

## Evaluation

Note: if you are not using gpu, you can remove the argument `--gpu-ids 0`

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
Other model options are "SAVN" and "GCN"

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

## Citations

Please cite our work if you found this research useful for your work:

```bash
@article{qiu2020learning,
  title={Learning hierarchical relationships for object-goal navigation},
  author={Qiu, Yiding and Pal, Anwesan and Christensen, Henrik I},
  journal={arXiv preprint arXiv:2003.06749},
  year={2020}
}
```
