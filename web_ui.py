import os
import glob
import base64
#from sklearn import pipeline
import streamlit as st

from all_pipelines import get_all_inputs
from constants import DATASET_DIR, MODEL_TESTING, MODEL_VISUAL, folder_last_modified

all_inputs = get_all_inputs()
all_input_names = list(all_inputs.keys())

global add_selectbox
add_selectbox = ""

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
	global add_selectbox
	pipeline_name = st.session_state.pipeline
	pipeline = all_inputs[pipeline_name]
	all_dataset_dir = DATASET_DIR.format(pipeline_name=pipeline_name)
	interpreters = all_inputs[pipeline_name].get_pipeline_dataset_interpreter()
	interpreter_list = pipeline.get_pipeline_dataset_interpreter()
	for interpreter_name in interpreter_list:
		interpreter_dataset_dir = os.path.join(all_dataset_dir, interpreter_name)
		interpreter_datasets = glob.glob(os.path.join(interpreter_dataset_dir,"*"))
		for dataset_dir in interpreter_datasets:
			
			# dat = interpreters[interpreter_name](dataset_dir).get_dataset()
			model_classes = all_inputs[pipeline_name].get_pipeline_model()
			for model_name in model_classes:
				print("-"*10)
				print("model_name:\t",model_name)
				print("interpreter_name:\t",interpreter_name)
				print("dataset_dir:\t",dataset_dir)
				visualizers = all_inputs[pipeline_name].get_pipeline_visualizer()
				for visualizer_name in visualizers:
					pass
	
	model_classes = all_inputs[pipeline_name].get_pipeline_model()
	model_name = st.sidebar.selectbox(
    	'Model',
		list(model_classes.keys()),
		key='model_name',
	)

	visualizers = all_inputs[pipeline_name].get_pipeline_visualizer()
	visualizer_name = st.sidebar.selectbox(
    	'Visualizers',
		list(visualizers.keys()),
		key='visualizer_name',
	)

	interpreter_list = pipeline.get_pipeline_dataset_interpreter()
	interpreter_name = st.sidebar.selectbox(
    	'Interpreter',
		list(interpreter_list.keys()),
		key='interpreter_name',
	)
	interpreter_dataset_dir = os.path.join(all_dataset_dir, interpreter_name)
	interpreter_datasets = glob.glob(os.path.join(interpreter_dataset_dir,"*"))
	
	dataset_dir = st.sidebar.selectbox(
    	'Dataset Directory',
		interpreter_datasets,
		key='dataset_dir',
	)
	

	# pipeline_name = st.session_state.pipeline
	# interpreter_name = st.session_state.interpreter_name
	# model_name = st.session_state.model_name
	# visualizer_name = st.session_state.visualizer_name
	visual_dir = MODEL_VISUAL.format(pipeline_name=pipeline_name, interpreter_name=interpreter_name, model_name=model_name, visualizer_name=visualizer_name)
	os.makedirs(visual_dir, exist_ok=True)

	for dirpath, dirnames, filenames in os.walk(visual_dir):
		for f in filenames:
			fp = os.path.join(dirpath, f)
			print(fp)
			if fp.endswith(".png"):
				show_image(fp)
			elif fp.endswith(".mp4"):
				st.video(fp)
			elif fp.endswith(".pkl"):
				pass
			else:
				pass



# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'Pipeline',
    all_input_names,
	key='pipeline',
	#on_change=main
)

if __name__=="__main__":
	main()