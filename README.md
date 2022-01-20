# Code Implementation For Paper "GypSum: Learning Hybrid Representations for Code Summarization"

## Description

We propose GypSum, a new deep learning model that learns hybrid representations using graph neural networks and a pre-trained programming and natural language model. GypSum uses two encoders to learn from the AST-based graph and the token sequence of source code, respectively, and modifies the encoder-decoder sublayer in the Transformer's decoder to fuse the representations.

## Requirement

This repos is developed based on the environment of:

Python 3.7

PyTorch 1.7.0

## Usage
1. How to construct graph for python and java program?
   > you can use the function `get_graph_from_source` from `proprocess/java(python)_graph_construction.py`
   After processing the code, save the data as `pkl` into data folder. 
2. How to run?
   > ./run $GPU_ID$ $DATASET$ $TASK$ 
   > For example, if you want to train java model from scrach with `cuda:0`, you can just run command `./run 0 java train`

## Structure

Gypsum-main
├─ README.md
├─ c2nl
│	   ├─ __init__.py
|	   ├─ encoders
|    |  └─ ...
|    ├─ decoders
|    |  └─ ...
|    ├─ eval
|    |  └─ ...
|    ├─ inputters
|    |  └─ ...
|    ├─ models
|    |  └─ ...
|    ├─ modules
|    |  └─ ...
|    ├─ objects
|    |  └─ ...
|    ├─ tokenizers
|    |  └─ ...
|    ├─ translator
|    |  └─ ...
|    ├─ utils
|      └─ ...
├─ code_clone_detection
├─ code_search
└─ model_implementation
       ├─ baselines
       ├─ TBCNN
       |	├─ ...
       ├─ TreeCaps
       |	├─ ...
       ├─ code2vec
       |    ├─ data_process.py
       |    └─ model.py
       ├─ code2seq
       |	├─ ...
       ├─ GGNN
       |  	├─ ...
       ├─ ASTNN
       |  	├─ ...
       └─ TPTrans
      		└─ ...



## How To Get Data 
We have uploaded our dataset on google drive: [Dataset For Experiment](https://drive.google.com/file/d/1hQWQE6qm-qNGYKEPMoVMepMEZ72nXJL3/view?usp=sharing)

### Data Directory Structure:
![image](https://user-images.githubusercontent.com/79627998/109446893-7f4a2b80-7a7d-11eb-8526-59b5ac275658.png)


## How to run: 
> python -u -u bert_nmt/train.py config/general_config.yml config/xxxx.yml
