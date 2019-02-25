import numpy as np

def convert_csv(filename):
    """This function returns a csv file given by filename and it
    returns the file as a np.array, with the input features on one side
    and the output features of the neural network on the other side"""

    n_features_in, n_features_out = get_number_features(filename)
    n_features = n_features_in + n_features_out

    csv_list = np.zeros(n_features)
    csv_list = np.reshape(csv_list,[1,n_features])

    with open(filename) as inf:
        # Skip header
        next(inf)
        for line in inf:
            # Read data, using python, into our features
            features_list = line.strip().split(",")


            features_list = np.array(list(map(float, features_list)))

            features_list = np.reshape(features_list,[1,n_features])
            csv_list = np.concatenate((csv_list, features_list), axis = 0)

    # avoid returning first row (zeros)
    csv_list = csv_list[1:]

    return csv_list

def sort_by_year(arr_in):
    # converts list to type int and then into np.array
    dtype = [('ID',float),('minute',float),('hour',float),('day',float),('month',float),
    ('year',float),('temperature',float),('lat',float),('lon',float),
    ('velocity',float),('fish',float)]

    arr_in = np.array(arr_in, dtype=dtype)
    sorted_array = np.sort(arr_in, order='year')

    return sorted_array

def get_number_features(filename):
    """Reads the header of the csv file and gets the number of features for our model"""
    with open(filename, 'r') as inf:
        first_line = inf.readline().strip().split(",")
        n_outputs = 1
        n_inputs = len(first_line) - n_outputs

    return n_inputs, n_outputs

if __name__ == '__main__':
    pass
