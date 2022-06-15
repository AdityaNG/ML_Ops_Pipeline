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
python model_analysis.py --single
```

To generate ensembler outputs, run as follows.
```bash
python ensemble_analysis.py --single
```

To visualize model results
```bash
python model_visualizer.py --pipeline_name obj_det --interpreter_name karthika95-pedestrian-detection --dataset_name 2022-06-15_21:25:11.138663 --model_name obj_det_pipeline_model_yolov5s --visualizer_name obj_det_data_visualizer
```

To visualize ensemble model results
```bash
python ensemble_visualizer.py --pipeline_name obj_det --interpreter_name karthika95-pedestrian-detection --dataset_name 2022-06-15_21:25:11.138663 --ensemble_name obj_det_pipeline_ensembler_1 --visualizer_name obj_det_data_visualizer
```

# Depth Perception Demo

Download <a href="https://drive.google.com/file/d/1yMPo_ux8tYT-gtinamRU-8qLPhmFmmUw/view?usp=sharing">Airsim Dataset</a>

```bash
python data_ingestion.py --input_dir ~/Downloads/2022-05-22-11-10-49 --pipeline_name depth_det --interpreter_name depth_interp_airsim
```

# Code Quality
Static Code Analysis
```bash
python -m pylint *.py
python -m pycodestyle *.py
```

# Unit Testing

TODO

