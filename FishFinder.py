from functions.generate_data import generate_data
from functions import read_csv
from functions import world
from functions import neural_network_keras
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
        filename = os.path.dirname(os.path.realpath(__file__)) + "/data_buoys.csv"
        csv_list = read_csv.convert_csv(filename)

        # Sets training and test data sets
        # 80% - 20% distribution
        test_length = int(0.2*len(csv_list))
        train_length =len(csv_list) - test_length

        print ("Converting data from file to list")
        # actually takes a lot of time
        # 3 minutes aprox for 100000 data entries
        X_train = csv_list[:(train_length),:(csv_list.shape[1] - 1)]
        Y_train = csv_list[:(train_length),(csv_list.shape[1] - 1)]
        Y_train = Y_train.reshape((train_length, 1))
        X_test = csv_list[train_length:(csv_list.shape[0]),:(csv_list.shape[1] - 1)]
        Y_test = csv_list[train_length:(csv_list.shape[0]),(csv_list.shape[1] - 1)]
        Y_test = Y_test.reshape((test_length, 1))

        # save variables for saving time next time they're needed
        np.savez('data_arrays', X_train, Y_train, X_test, Y_test)

    if "--train" in sys.argv or "--new" in sys.argv:



        if os.path.isfile('data_arrays.npz'):
            print ("Loading arrays from file")
            npzfile = np.load('data_arrays.npz')

            X_train, Y_train, X_test, Y_test = npzfile['arr_0'], npzfile['arr_1'], npzfile['arr_2'],npzfile['arr_3']

            # print('X_train.shape = ' + str(X_train.shape))
            # print('Y_train.shape = ' + str(Y_train.shape))
            # print('X_test.shape = ' + str(X_test.shape))
            # print('Y_test.shape = ' + str(Y_test.shape))
            #print(type(Y_test[0][0]))
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
