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
```bash
Gypsum-main
├─- README.md
├── c2nl
│   ├── __init__.py
│   ├── __pycache__
│   ├── config.py
│   ├── decoders
│   ├── encoders
│   ├── eval
│   ├── inputters
│   ├── models
│   ├── modules
│   ├── objects
│   ├── tokenizers
│   ├── translator
│   └── utils
├── config
│   ├── general_config.yml
│   ├── java_xxx_xxx.yml
│   ├── ...
├── data
│   ├── java
│   └── python
├── evaluation
│   ├── bleu
│   ├── evaluate.py
│   ├── meteor
│   └── rouge
├── gypsum
│   ├── __pycache__
│   ├── data
│   ├── metor.ipynb
│   ├── model.py
│   ├── modules
│   ├── predict.py
│   ├── train.py
│   └── utils
├── modules
│   ├── __pycache__
│   └── attention_zoo.py
├── preprocess
│   ├── generate_java_graph.ipynb
│   ├── java_graph_construct.py
│   ├── python_ast.ipynb
│   └── python_graph.py
└── run
```


## Data Availability 
> We have uploaded our dataset on google drive: [Dataset For Experiment](https://drive.google.com/file/d/1hQWQE6qm-qNGYKEPMoVMepMEZ72nXJL3/view?usp=sharing)


## Acknowledgement
> We borrowed and modified code from [DrQA](https://github.com/facebookresearch/DrQA), [OpenNMT](https://github.com/OpenNMT/OpenNMT-py). We would like to expresse our gratitdue for the authors of these repositeries.
