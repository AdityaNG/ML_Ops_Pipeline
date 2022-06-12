from xml.parsers.expat import model
from helper import pure_cnn_model, cnn_model
from constants import *
import glob
import pickle

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.constraints import maxnorm
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import time
import datetime

from pipeline_input import *
from cifar10_demo import all_inputs

def main():
    all_pipelines = all_inputs.keys()
    for p in all_pipelines:
        training_settings = {
            "datasets": set()
        }
        MODEL_TRAINING_SETTINGS
        if os.path.exists(MODEL_TRAINING_SETTINGS):
            with open(MODEL_TRAINING_SETTINGS, 'rb') as handle:
                training_settings = pickle.load(handle)


        all_datasets = set(glob.glob(os.path.join(DATASET_DIR, p, "*.pkl")))
        new_datasets = all_datasets.difference(training_settings['datasets'])

        #new_datasets = {"data/dataset/2022-06-08 19:23:13.707565.pkl"}

        if len(new_datasets)>0:
            print("Training on new datasets: {}".format(new_datasets))
            for dataset_path in new_datasets:
                with open(dataset_path, 'rb') as handle:
                    dataset = pickle.load(handle)
                    trained_model_save_path = os.path.join(MODEL_CHECKPOINTS, os.path.basename(dataset_path))
                    log_dir = os.path.join(trained_model_save_path.replace(".pkl",""), "logs")
                    trained_model_and_model_details = train_on_dataset(dataset, log_dir)
                    
                    print("Saving trained_model to: {}".format(trained_model_save_path))
                    with open(trained_model_save_path, 'wb') as handle:
                        pickle.dump(trained_model_and_model_details, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            print("No new datasets found")
            print("Sleeping for",SLEEP_TIME,"sec")
            time.sleep(SLEEP_TIME)

        training_settings["datasets"] = all_datasets

        print("Saving training_settings to: {}".format(MODEL_TRAINING_SETTINGS))
        with open(MODEL_TRAINING_SETTINGS, 'wb') as handle:
            pickle.dump(training_settings, handle, protocol=pickle.HIGHEST_PROTOCOL)
            

def main_old():
    training_settings = {
        "datasets": set()
    }

    if os.path.exists(MODEL_TRAINING_SETTINGS):
        with open(MODEL_TRAINING_SETTINGS, 'rb') as handle:
            training_settings = pickle.load(handle)


    all_datasets = set(glob.glob(os.path.join(DATASET_DIR, "*.pkl")))
    new_datasets = all_datasets.difference(training_settings['datasets'])

    #new_datasets = {"data/dataset/2022-06-08 19:23:13.707565.pkl"}

    if len(new_datasets)>0:
        print("Training on new datasets: {}".format(new_datasets))
        for dataset_path in new_datasets:
            with open(dataset_path, 'rb') as handle:
                dataset = pickle.load(handle)
                trained_model_save_path = os.path.join(MODEL_CHECKPOINTS, os.path.basename(dataset_path))
                log_dir = os.path.join(trained_model_save_path.replace(".pkl",""), "logs")
                trained_model_and_model_details = train_on_dataset(dataset, log_dir)
                
                print("Saving trained_model to: {}".format(trained_model_save_path))
                with open(trained_model_save_path, 'wb') as handle:
                    pickle.dump(trained_model_and_model_details, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        print("No new datasets found")
        print("Sleeping for",SLEEP_TIME,"sec")
        time.sleep(SLEEP_TIME)

    training_settings["datasets"] = all_datasets

    print("Saving training_settings to: {}".format(MODEL_TRAINING_SETTINGS))
    with open(MODEL_TRAINING_SETTINGS, 'wb') as handle:
        pickle.dump(training_settings, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception as e:
            print("Exception: {}".format(e))