from fishfinder.utils.world import Map
from fishfinder.utils.generate_data import DataGenerator
from fishfinder.utils import neural_network_keras
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import sys
import numpy as np
import json
import datetime


class FishFinder(Map):

    def __init__(self):

        # read json file
        current_directory = os.path.dirname(os.path.abspath(__file__))
        path_to_json = os.path.join(current_directory, '../conf.json')

        with open(path_to_json) as f:
            self.conf = json.load(f)

        Map.__init__(self, self.conf, "results")

    def generate_datasets(self, save_to_csv, test_size=0.20):

        print("Generating new data!")
        dg = DataGenerator(self.conf, save_to_csv)
        self.df = dg.generate_data()

        features = self.df.iloc[:, :10]
        labels = self.df.iloc[:, 10:]

        # Sets training and test data sets
        X_train, X_test, Y_train, Y_test = train_test_split(
            features, labels, test_size=test_size)  # returns numpy.ndarray

        # Save variables for saving time next time they're needed
        np.savez('data_arrays', X_train, Y_train, X_test, Y_test)

    def import_datasets(self, path_to_datasets, label_name):
        # Given any datasets and label name, it returns X_train, X_test, Y_train and Y_test
        pass

    def train(self):

        if os.path.isfile('data_arrays.npz'):
            print("Loading datasets from files")
            npzfile = np.load('data_arrays.npz')

            self.X_train, self.Y_train, self.X_test, self.Y_test = \
                npzfile['arr_0'], npzfile['arr_1'], npzfile['arr_2'], npzfile['arr_3']

            model = neural_network_keras.create_model_api()
            neural_network_keras.train_model(
                self.X_train, self.Y_train, self.X_test, self.Y_test, model)

        else:
            print(
                "No data found for training the DNN, run again the program with the argument --new")

    def get_dataframe(self):
        return self.df
