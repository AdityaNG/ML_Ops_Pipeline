from distutils.dir_util import copy_tree

from sklearn import datasets
from constants import *
from tqdm import tqdm
import cv2
import datetime
import pickle
import numpy as np
from keras.utils import np_utils
import glob

from pipeline_input import *
from cifar10_demo import all_inputs


def ingest_data(p_input: pipeline_input, input_dir: str):
    assert isinstance(p_input, pipeline_input)
    dataset_interp_class = p_input.get_pipeline_dataset_interpreter()
    pipeline_name = p_input.get_pipeline_name()
    PKL_FOLDER = os.path.join(DATASET_DIR, pipeline_name)
    os.makedirs(PKL_FOLDER, exist_ok=True)
    PKL_PATH = os.path.join(PKL_FOLDER, str(datetime.datetime.now()) + '.pkl')
    print("Saving dataset to: {}".format(PKL_PATH))
    with open(PKL_PATH, 'wb') as handle:
        pickle.dump(dataset_interp_class(input_dir).get_dataset(), handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--pipeline_name', type=str, required=True)
    args = parser.parse_args()
    ingest_data(all_inputs[args.pipeline_name], args.input_dir)