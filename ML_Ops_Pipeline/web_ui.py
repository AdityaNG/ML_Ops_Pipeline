import os
import glob
import base64
import pickle
#from sklearn import pipeline
import streamlit as st
import pandas as pd

from ML_Ops_Pipeline.all_pipelines_git import get_all_inputs
from ML_Ops_Pipeline.constants import DATASET_DIR, MODEL_TRAINING, MODEL_TESTING, MODEL_VISUAL, folder_last_modified

all_inputs = get_all_inputs()
all_input_names = list(all_inputs.keys())

def show_image(img_path):
	file_ = open(img_path, "rb")
	contents = file_.read()
	data_url = base64.b64encode(contents).decode("utf-8")
	file_.close()

	st.markdown(
		f'<img width="80%" src="data:image/gif;base64,{data_url}" alt="cat gif">',
		unsafe_allow_html=True,
	)


def main():
	# Add a selectbox to the sidebar:
	pipeline_name = st.sidebar.selectbox(
		'Pipeline',
		all_input_names,
		key='pipeline',
	)
	print("pipeline_name",pipeline_name)
	# pipeline_name = st.session_state.pipeline
	if not pipeline_name:
		st.markdown("# Error")
		st.markdown("No pipelines available")
		return
	pipeline = all_inputs[pipeline_name]['pipeline']
	viz_class = pipeline.get_pipeline_streamlit_visualizer()
	viz = viz_class(pipeline, st)
	viz.visualize()
		

if __name__=="__main__":
	main()