# Toy ML Ops Pipeline

<img width="70%" src="imgs/demo.gif" />

A simple demonstration of an ML Ops pipeline involving three stages:
1. Data Ingestion
2. Model Training
3. Model Analysis

To add your own pipeline, model, datasets, etc., take a look at <a href="pipelines/README.md">pipelines/README.md</a>

# Getting started

Download and unzip the <a href="https://www.kaggle.com/datasets/karthika95/pedestrian-detection">karthika95-pedestrian-detection kaggle dataset</a> to `~/Downloads/karthika95-pedestrian-detection/`.

Data ingestion can be run with the following. It will validate the dataset and store it to `data/`.
```bash
python data_ingestion.py --input_dir ~/Downloads/karthika95-pedestrian-detection/ --pipeline_name obj_det --interpreter_name karthika95-pedestrian-detection

python3 data_ingestion.py --input_dir ~/datasets/klemenko-kitti-dataset/ --pipeline_name obj_det --interpreter_name KITTI_lemenko_interp
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
python model_visualizer.py \
	--pipeline_name obj_det \
	--interpreter_name karthika95-pedestrian-detection \
	--dataset_name 2022-06-15_21:25:11.138663 \
	--model_name obj_det_pipeline_model_yolov5s \
	--visualizer_name obj_det_data_visualizer
```

To visualize ensemble model results
```bash
python ensemble_visualizer.py \
	--pipeline_name obj_det \
	--interpreter_name karthika95-pedestrian-detection \
	--dataset_name 2022-06-15_21:25:11.138663 \
	--ensemble_name obj_det_pipeline_ensembler_1 \
	--visualizer_name obj_det_data_visualizer
```

# Running as a systemd service
The file <a href="mlops.service">mlops.service</a> is to be copied to `/etc/systemd/system/`. The service can then be started and status can be checked on using the following commands.
```bash
sudo cp mlops.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start mlops.service
sudo systemctl status mlops.service
```

To view the logs of a specific subprocess, use the tmux script
```bash
./logs.sh
# OR
tmux source-file mlops.tmux
```

# Depth Perception Demo

Download <a href="https://drive.google.com/file/d/1yMPo_ux8tYT-gtinamRU-8qLPhmFmmUw/view?usp=sharing">Airsim Dataset</a>

```bash
python data_ingestion.py \
	--input_dir ~/Downloads/2022-05-22-11-10-49 \
	--pipeline_name depth_det \
	--interpreter_name depth_interp_airsim
```

# Web UI

Start the REST API server
```bash
FLASK_APP=rest_server.py FLASK_ENV=development flask run
```

Start the web UI
```bash
cd mlops-react-dashboard
yarn install
yarn start
```

Install this specific version of `browsepy`
```bash
python -m pip install git+https://github.com/AdityaNG/browsepy.git@galary_support
```

Run streamlit web UI
```bash
streamlit run web_ui.py
```

# Code Quality
Static Code Analysis
```bash
python -m pylint *.py
python -m pycodestyle *.py
```

# Unit Testing

TODO

