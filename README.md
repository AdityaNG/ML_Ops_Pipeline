# Toy ML Ops Pipeline

A simple demonstration of an ML Ops pipeline involving three stages:
1. Data Ingestion
2. Model Training
3. Model Analysis

# Getting started

Download and unzip the <a href="https://www.kaggle.com/datasets/karthika95/pedestrian-detection">karthika95-pedestrian-detection kaggle dataset</a> to `~/Downloads/karthika95-pedestrian-detection/`.

Data ingestion can be run with the following. It will validate the dataset and store it to `data/`.
```bash
python data_ingestion.py --input_dir ~/Downloads/karthika95-pedestrian-detection/ --pipeline_name obj_det --interpreter_name karthika95-pedestrian-detection
```

To generate individual model outputs, run as follows.
```bash
python model_analysis.py
```

To generate ensembler outputs, run as follows.
```bash
python ensemble_analysis.py
```