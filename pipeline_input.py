import numpy as np
import inspect
import hashlib

import os
import glob
import base64
import pickle
#from sklearn import pipeline
import streamlit as st
import pandas as pd

import tensorflow as tf
from constants import DATASET_DIR, MODEL_TRAINING, MODEL_TESTING, MODEL_VISUAL

def show_image(img_path):
	file_ = open(img_path, "rb")
	contents = file_.read()
	data_url = base64.b64encode(contents).decode("utf-8")
	file_.close()

	st.markdown(
		f'<img width="80%" src="data:image/gif;base64,{data_url}" alt="cat gif">',
		unsafe_allow_html=True,
	)

def source_hash(self) -> int:
	if self == object:
		return 0
	elif type(self) == type:
		type_class = self

		type_source = inspect.getsource(type_class)
		for super_class in type_class.__mro__:
			if super_class!=type_class:
				#print(type_class, "\t->\t", super_class)
				type_source += str(super_class) + ":" + str(source_hash(super_class)) + "\n"
		return int(hashlib.sha1(type_source.encode("utf-8")).hexdigest(), 16) 
	else:
		type_class = type(self)
		
		type_source = inspect.getsource(type_class)
		for super_class in type_class.__mro__:
			#print(type_class, "\t->\t", super_class)
			type_source += str(super_class) + ":" + str(source_hash(super_class)) + "\n"
		return int(hashlib.sha1(type_source.encode("utf-8")).hexdigest(), 16) 
		

class pipeline_classes:

	def __hash__(self) -> int:
		return source_hash(self)

class pipeline_dataset_interpreter(pipeline_classes):
	# TODO: Impractical to have entire dataset to be loaded into memory
	# look into alterative architectures which load things into memory chunk by chunk
	dataset = {
		'test': {
			'x':   None,
			'y': None
		},
		'train': {
			'x':   None,
			'y': None
		}
	}

	def __init__(self, input_dir, load=True) -> None:
		self.input_dir = input_dir
		if load:
			self.load()

	def get_dataset(self) -> dict:
		return self.dataset

	def load(self) -> None:
		# Load the dataset into self.dataset
		pass


class pipeline_model(pipeline_classes):
	model = None

	def __init__(self, training_dir, load=True) -> None:
		self.training_dir = training_dir
		if load:
			self.load()


	def get_model(self) -> tf.keras.Model:
		return self.model

	def load(self) -> None:
		# Load the model into self.model
		pass

	def predict(self, x: np.array) -> np.array:
		# Runs prediction on list of values x of length n
		# Returns a list of values of length n
		pass

	def evaluate(self, x, y) -> np.array:
		pass

	def train(self, x, y) -> np.array:
		pass


class pipeline_ensembler(pipeline_classes):

	def predict(self, x: np.array) -> np.array:
		# Given a list of lists of predictions from multiple learners
		# Say m learners and n predictions
		# x[m][n] -> the nth prediction for the mth learner
		# Returns a single result
		# y[n] -> merged nth prediction
		pass
		
	def evaluate(self, x, y) -> np.array:
		pass
	
	def train(self, x, y) -> np.array:
		pass

class pipeline_data_visualizer(pipeline_classes):

	def visualize(self, dataset_x, dataset_y, results, predictions, directory) -> None:
		# Visualize the data
		# TODO Save data to the given directory
		pass

class pipeline_streamlit_visualizer(pipeline_classes):
	def __init__(self, pipeline) -> None:
		super().__init__()
		self.pipeline = pipeline
		
	def load_data(self):
		self.pipeline_name = st.session_state.pipeline
		#self.pipeline_name = self.pipeline.get_pipeline_name()

		self.all_dataset_dir = DATASET_DIR.format(pipeline_name=self.pipeline_name)
		self.interpreters = self.pipeline.get_pipeline_dataset_interpreter()
		self.interpreter_list = self.pipeline.get_pipeline_dataset_interpreter()
		
		self.model_classes = self.pipeline.get_pipeline_model()
		
		if 'model_name' not in st.session_state:
			self.model_name = st.sidebar.selectbox(
				'Model',
				list(self.model_classes.keys()),
				key='model_name',
			)
		else:
			self.model_name = st.session_state['model_name']

		self.visualizers = self.pipeline.get_pipeline_visualizer()		

		self.interpreter_list = self.pipeline.get_pipeline_dataset_interpreter()
		if 'interpreter_name' not in st.session_state:
			self.interpreter_name = st.sidebar.selectbox(
				'Interpreter',
				list(self.interpreter_list.keys()),
				key='interpreter_name',
			)
		else:
			self.interpreter_name = st.session_state['interpreter_name']

		self.interpreter_dataset_dir = os.path.join(self.all_dataset_dir, self.interpreter_name)
		self.interpreter_datasets = glob.glob(os.path.join(self.interpreter_dataset_dir,"*"))

		if 'dataset_dir' not in st.session_state:
			self.dataset_dir = st.sidebar.selectbox(
				'Dataset Directory',
				self.interpreter_datasets,
				key='dataset_dir',
			)
		else:
			self.dataset_dir = st.session_state['dataset_dir']



		training_dir = MODEL_TRAINING.format(
			pipeline_name=self.pipeline_name,
			interpreter_name=self.interpreter_name,
			model_name=self.model_name
		)

		testing_dir = MODEL_TESTING.format(
			pipeline_name=self.pipeline_name,
			interpreter_name=self.interpreter_name,
			model_name=self.model_name
		)

		self.testing_results_pkl = os.path.join(testing_dir, "results.pkl")
		self.testing_predictions_pkl = os.path.join(testing_dir, "predictions.pkl")
		self.testing_results_handle = open(self.testing_results_pkl, 'rb')
		self.testing_results = pickle.load(self.testing_results_handle)
		self.testing_results_handle.close()

		self.testing_predictions_handle = open(self.testing_predictions_pkl, 'rb')
		self.testing_predictions = pickle.load(self.testing_predictions_handle)
		self.testing_predictions_handle.close()

		self.training_results_pkl = os.path.join(training_dir, "results.pkl")
		self.training_predictions_pkl = os.path.join(training_dir, "predictions.pkl")
		self.training_results_handle = open(self.training_results_pkl, 'rb')
		self.training_results = pickle.load(self.training_results_handle)
		self.training_results_handle.close()

		self.training_predictions_handle = open(self.training_predictions_pkl, 'rb')
		self.training_predictions = pickle.load(self.training_predictions_handle)
		self.training_predictions_handle.close()

		self.dat = self.interpreters[self.interpreter_name](self.dataset_dir).get_dataset()

		st.markdown("# Testing Results")
		st.write(self.testing_results)
		#st.write(self.testing_predictions)

		st.markdown("# Training Results")
		st.write(self.training_results)
		#st.write(self.training_predictions)

		try:
			st.markdown("# Dataset")
			dat_test = self.dat['test']['x'].join(self.dat['test']['y'])
			st.write(dat_test)
		except:
			pass
		finally:
			st.markdown("# Dataset X")
			st.write(self.dat['test']['x'])
			st.markdown("# Dataset Y")
			st.write(self.dat['test']['y'])

	def visualize(self):
		self.load_data()

		if 'visualizer_name' not in st.session_state:
			self.visualizer_name = st.sidebar.selectbox(
				'Visualizers',
				list(self.visualizers.keys()),
				key='visualizer_name',
			)
		else:
			self.visualizer_name = st.session_state['visualizer_name']
		
		self.visual_dir = MODEL_VISUAL.format(pipeline_name=self.pipeline_name, interpreter_name=self.interpreter_name, model_name=self.model_name, visualizer_name=self.visualizer_name)
		st.markdown("# Visuals")
		MAX_FRAMES = 5
		frame_count = 0
		for dirpath, dirnames, filenames in os.walk(self.visual_dir):
			for f in filenames:
				fp = os.path.join(dirpath, f)
				st.markdown(fp)
				if fp.endswith(".png"):
					show_image(fp)
				elif fp.endswith(".mp4"):
					st.video(fp)
				elif fp.endswith(".pkl"):
					pass
				else:
					pass

				if frame_count>MAX_FRAMES:
					return
				frame_count+=1
		

class pipeline_input(pipeline_classes):
	__pipeline_name = {}
	__pipeline_dataset_interpreter = {}
	__pipeline_model = {}
	__pipeline_ensembler = {}
	__pipeline_data_visualizer = {}
	__pipeline_streamlit_visualizer = None

	def __init__(self, p_name: str, p_dataset_interpreter: dict, p_model: dict, p_ensembler: dict, p_vizualizer: dict, p_pipeline_streamlit_visualizer=pipeline_streamlit_visualizer) -> None:
		assert isinstance(p_name, str)
		assert isinstance(p_dataset_interpreter,dict)
		for p in p_dataset_interpreter:
			assert issubclass(p_dataset_interpreter[p],pipeline_dataset_interpreter)
		assert isinstance(p_model,dict)
		for p in p_model:
			assert issubclass(p_model[p],pipeline_model)
		assert isinstance(p_ensembler,dict)
		for p in p_ensembler:
			assert issubclass(p_ensembler[p],pipeline_ensembler)
		assert isinstance(p_vizualizer,dict)
		for p in p_vizualizer:
			assert issubclass(p_vizualizer[p],pipeline_data_visualizer)
		self.__pipeline_name = p_name
		self.__pipeline_dataset_interpreter = p_dataset_interpreter
		self.__pipeline_model = p_model
		self.__pipeline_ensembler = p_ensembler
		self.__pipeline_data_visualizer = p_vizualizer
		self.__pipeline_streamlit_visualizer = p_pipeline_streamlit_visualizer

	def get_pipeline_name(self) -> str:
		return self.__pipeline_name

	def get_pipeline_streamlit_visualizer(self) -> str:
		return self.__pipeline_streamlit_visualizer

	def get_pipeline_dataset_interpreter_by_name(self, name: str) -> type:
		return self.__pipeline_dataset_interpreter[name]

	def get_pipeline_model_by_name(self, name: str) -> type:
		return self.__pipeline_model[name]

	def get_pipeline_ensembler_by_name(self, name: str) -> type:
		return self.__pipeline_ensembler[name]
	
	def get_pipeline_visualizer_by_name(self, name: str) -> type:
		return self.__pipeline_data_visualizer[name]

	def get_pipeline_dataset_interpreter(self) -> dict:
		return self.__pipeline_dataset_interpreter

	def get_pipeline_model(self) -> dict:
		return self.__pipeline_model

	def get_pipeline_ensembler(self) -> dict:
		return self.__pipeline_ensembler

	def get_pipeline_visualizer(self) -> dict:
		return self.__pipeline_data_visualizer
