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
from keras.layers import GlobalAveragePooling2D, Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import time
import datetime

def train_on_dataset(dataset, log_dir):
    #model = pure_cnn_model()
    model = cnn_model()
    checkpoint = ModelCheckpoint('best_model_improved.h5',  # model filename
                             monitor='val_loss', # quantity to monitor
                             verbose=0, # verbosity - 0 or 1
                             save_best_only= True, # The latest best model will not be overwritten
                             mode='auto') # The decision to overwrite model is made 
                                          # automatically depending on the quantity to monitor 

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.compile(loss='categorical_crossentropy', # Better loss function for neural networks
              optimizer=Adam(lr=LEARN_RATE), # Adam optimizer with 1.0e-4 learning rate
              metrics = ['accuracy']) # Metrics to be evaluated by the model
    
    images_train = dataset['train']['img']
    class_train = dataset['train']['class']
    images_test = dataset['test']['img']
    class_test = dataset['test']['class']
    
    model_details = model.fit(images_train, class_train,
                    batch_size = 128,
                    epochs = NUM_EPOCH, # number of iterations
                    validation_data= (images_test, class_test),
                    callbacks=[checkpoint, tensorboard_callback],
                    verbose=1)
    return model, model_details

def main():
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