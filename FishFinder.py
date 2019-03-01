from utils.generate_data import generate_data
from utils import world
from utils import neural_network_keras
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import sys
import numpy as np
import json
import datetime

def main():

    # read json file
    with open('conf.json') as f:
        conf = json.load(f)

    if "--new" in sys.argv:
        # Generates new data and saves it into csv file
        print ("Generating new data!")
        generate_data(conf["map"]["number_of_buoys"], conf["map"]["data_per_buoy"],
        conf["map"]["world_width"], conf["map"]["world_height"])

        # Convert data in csv file into numpy arrays
        filename = os.path.dirname(os.path.realpath(__file__)) + "/pandas_data.csv"

        # Use pandas to handle csv file created
        df = pd.read_csv(filename)
        features = df.iloc[:, :10]
        labels = df.iloc[:, 10:]


        # Sets training and test data sets
        X_train, X_test, Y_train, Y_test = train_test_split(
            features, labels, test_size=0.20) # returns numpy.ndarray

        # Save variables for saving time next time they're needed
        np.savez('data_arrays', X_train, Y_train, X_test, Y_test)

        print("Size of %s is %s" % ("features", features.shape))
        print("Size of %s is %s" % ("labels", labels.shape))
        print("Size of %s is %s" % ("X_train", X_train.shape))
        print("Size of %s is %s" % ("Y_train", Y_train.shape))
        print("Size of %s is %s" % ("X_test", X_test.shape))
        print("Size of %s is %s" % ("Y_test", Y_test.shape))

    if "--train" in sys.argv or "--new" in sys.argv:



        if os.path.isfile('data_arrays.npz'):
            print ("Loading arrays from file")
            npzfile = np.load('data_arrays.npz')

            X_train, Y_train, X_test, Y_test = npzfile['arr_0'], npzfile['arr_1'], npzfile['arr_2'],npzfile['arr_3']

            model = neural_network_keras.create_model_api()
            neural_network_keras.train_model(X_train, Y_train, X_test, Y_test, model)

        else:
            print("No data found for training the DNN, run again the program with the argument --new")

    if "--showpath" in sys.argv:
        grid_parameters = world.create_grid(conf["map"]["start_x"], \
            conf["map"]["start_y"], conf["map"]["end_x"], conf["map"]["end_y"], \
            conf["map"]["displacement"])
        paths, fish_list = world.create_path(conf["weights"], grid_parameters[0])

        # create directory and save results

        # graph
        if not os.path.exists('Results'):
            os.mkdir('Results')
        date = datetime.datetime.now()
        dirname = str(date.year) + str(date.month) + str(date.day) + '_' + \
         str(date.hour) + str(date.minute) + str(date.second)
        path_results = os.path.join(os.getcwd(), 'Results', dirname)


        world.show_grid(grid_parameters[1], conf["map"]["show_points"], paths, path_results)

        # computations
        boat_features = [conf["boat_features"]["avg_speed"], conf["boat_features"]["fuel_consumption"], \
        conf["boat_features"]["fuel_consumption_turning"], conf["boat_features"]["time_consumption_turning"]]
        world.compute_feedback_data(paths, fish_list, path_results, boat_features)

        print('Results ready in ' + path_results)


if __name__ == '__main__':
    main()

