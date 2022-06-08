from xml.parsers.expat import model
from helper import pure_cnn_model
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
from keras.layers import GlobalAveragePooling2D, Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation
from keras.preprocessing.image import ImageDataGenerator
import time

from helper import predict_classes, visualize_errors, plot_model

def main():
    validation_settings = {
        "models": set()
    }

    if os.path.exists(MODEL_VALIDATION_SETTINGS):
        with open(MODEL_VALIDATION_SETTINGS, 'rb') as handle:
            validation_settings = pickle.load(handle)


    all_models = set(glob.glob(os.path.join(MODEL_CHECKPOINTS, "*.pkl")))
    new_models = all_models.difference(validation_settings['models'])

    #new_models = {"data/model_checkpoints/2022-06-08 19:23:13.707565.pkl"}

    if len(new_models)>0:
        print("Validating new models: {}".format(new_models))
        for model_path in new_models:
            with open(model_path, 'rb') as handle:
                trained_model, model_details = pickle.load(handle)
            
            dataset_path = os.path.join(DATASET_DIR, os.path.basename(model_path))
            with open(dataset_path, 'rb') as handle2:
                dataset = pickle.load(handle2)

            images_train = dataset['train']['img']
            class_train = dataset['train']['class']
            images_test = dataset['test']['img']
            labels_test = dataset['test']['label']
            class_test = dataset['test']['class']

            correct, labels_pred = predict_classes(trained_model, images_test, labels_test)
            
            num_images = len(correct)
            print("Accuracy: %.2f%%" % ((sum(correct)*100)/num_images))

            print(model_details.history)
            # Plot Model
            plot_model(model_details)
            plot_file_save_path = os.path.join(ANALYSIS_DIR, "plot_model_" + os.path.basename(model_path) + ".png")
            print("Saving plot:",plot_file_save_path)
            plt.savefig(plot_file_save_path)

            class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

            # Visualize errors
            #visualize_errors(dataset, labels_test, labels_pred, correct, num_images)
            visualize_errors(images_test, labels_test, class_names, labels_pred, correct)
            plot_file_save_path = os.path.join(ANALYSIS_DIR, "visualize_errors_" + os.path.basename(model_path) + ".png")
            print("Saving plot:",plot_file_save_path)
            plt.savefig(plot_file_save_path)
    else:
        print("No new models found")
        print("Sleeping for",SLEEP_TIME,"sec")
        time.sleep(SLEEP_TIME)

    validation_settings["models"] = all_models

    print("Saving validation_settings to: {}".format(MODEL_VALIDATION_SETTINGS))
    with open(MODEL_VALIDATION_SETTINGS, 'wb') as handle:
        pickle.dump(validation_settings, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception as e:
            print("Exception: {}".format(e))