from matplotlib import pyplot as plt
import numpy as np
import math
import random
import os

def plot(csv_list, n_buoys):
    """Plots current position and signal of buoys"""
    #TODO: plot for different years
    # PLOT CURRENT POSITION
    csv_list = csv_list[-(int(n_buoys)):,:]
    buoy_fish_lat = np.array([0])
    buoy_fish_lon = np.array([0])
    buoy_no_fish_lat = np.array([0])
    buoy_no_fish_lon = np.array([0])

    for i in range(len(csv_list)):
        if csv_list[i,10] == 1:
            buoy_fish_lat = np.insert(buoy_fish_lat, len(buoy_fish_lat), csv_list[i,7])
            buoy_fish_lon = np.insert(buoy_fish_lon, len(buoy_fish_lon), csv_list[i,8])
        else:
            buoy_no_fish_lat = np.insert(buoy_no_fish_lat, len(buoy_no_fish_lat), csv_list[i,7])
            buoy_no_fish_lon = np.insert(buoy_no_fish_lon, len(buoy_no_fish_lon), csv_list[i,8])

    buoy_fish_lat = buoy_fish_lat[1:]
    buoy_fish_lon = buoy_fish_lon[1:]
    buoy_no_fish_lat = buoy_no_fish_lat[1:]
    buoy_no_fish_lon = buoy_no_fish_lon[1:]


    plt.scatter(buoy_fish_lon,buoy_fish_lat, c='blue', label='Fish')
    plt.scatter(buoy_no_fish_lon,buoy_no_fish_lat, c='red', label='No fish')
    plt.title('Buoys')
    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    plt.legend()
    plt.grid(True,color='k')
    plt.show()

def create_grid(start_x, start_y, end_x, end_y, displacement):
    """ It takes two points as an input and returns a list of points and its properties,
    which make up a grid of a close area between those two points """

    p0 = np.array([start_x, start_y]) # initial point of the path
    pf = np.array([end_x, end_y]) # final point of the path

    l1 = np.linalg.norm(np.subtract(pf,p0))
    l2 = 0.2 * l1

    # we define the 2 unit vectors d1 and d2 in which directions we are moving along
    # then we apply to them the size of the desired displacement
    d1 = np.subtract(pf,p0)/(np.linalg.norm(np.subtract(pf,p0)))

    d2 = np.array([-d1[1], d1[0]])
    d2 = d2/(np.linalg.norm(d2))


    # we create our grid moving with d1 and d2
    # each point of the grid has the following properties:
    # position, distance to D1, distance to pf, velocity
    n_displacements_2 = 0
    lines_list = []

    # variables for making easier the plotting afterwards
    x_list = []
    y_list = []
    time_n = 1

    while True:
        current_point = np.subtract(p0,d2*(n_displacements_2*displacement))
        current_length_2 = np.linalg.norm(np.subtract(current_point, p0))
        if current_length_2 > l2/2:
            if time_n == 2:
                break
            else:
                time_n += 1
                lines_list = list(reversed(lines_list))
                d2 = -d2
                n_displacements_2 = 1
                current_point = np.subtract(p0,d2*(n_displacements_2*displacement))


        line_points = []
        x_list.append(current_point[0])
        y_list.append(current_point[1])

        #TODO: add fish real prediction
        line_points.append({
        "position": current_point.tolist(),
        "distance_to_l1": n_displacements_2*displacement,
        "distance_to_pf": np.linalg.norm(np.subtract(pf,current_point)),
        "fish": random.random()
        })


        initial_point_1 = current_point
        while True:
            current_point = np.sum([current_point, d1 * displacement], axis=0)
            current_length_1 = np.linalg.norm(np.subtract(current_point, initial_point_1))
            if current_length_1 >= l1:
                current_point = pf - d2*n_displacements_2*displacement
                x_list.append(current_point[0])
                y_list.append(current_point[1])
                #TODO: add fish real prediction
                line_points.append({
                "position": current_point.tolist(),
                "distance_to_l1": n_displacements_2*displacement,
                "distance_to_pf": np.linalg.norm(np.subtract(pf,current_point)),
                "fish": random.random()
                })
                break




            x_list.append(current_point[0])
            y_list.append(current_point[1])

            #TODO: add fish real prediction
            line_points.append({
            "position": current_point.tolist(),
            "distance_to_l1": n_displacements_2*displacement,
            "distance_to_pf": np.linalg.norm(np.subtract(pf,current_point)),
            "fish": random.random()
            })

        lines_list.append(line_points)
        n_displacements_2 += 1


    return [[p0, pf, d1, d2, lines_list, [l1, l2]], [x_list, y_list]]

def fuel_prediction(initial_pos, d1, d2, avg_speed):
    """ Makes a prediction of the fuel spent given currents velocity, initial and
    final position and average speed. The idea is to decompose the velocity in vectors
    parallel to d1 and d2 and see how much they help guiding the boat into the final point, giving
    us an idea of the fuel spent """



    return 0


def show_grid(grid_points, show_points, path_points, path_results):
    """ It shows each point in the map with a different color in function of
    its probability """

    if not os.path.exists(path_results):
        os.mkdir(path_results)

    fig = plt.figure()

    grid_points_x = grid_points[0]
    grid_points_y = grid_points[1]

    if show_points == 1:
        plt.scatter(grid_points_x, grid_points_y, c='blue')


    for plots in path_points:
        plt.plot(plots[0], plots[1], c=plots[2])
    plt.title('Paths')
    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    #plt.legend()

    fig.savefig(os.path.join(path_results, 'Paths.png') )

def create_path(weights, path_data, show_path=True):
    """ Given the first parameter returned from create_grid() and
    a weight list, it creates the path the boat should follow from
    its initial to its final position in order to maximize its
    fish capture along the way """

    # unroll path_data
    p0 = path_data[0]
    pf = path_data[1]
    d1 = path_data[2]
    d2 = path_data[3]
    lines_list = path_data[4]
    [l1, l2] = path_data[5]

    # control variables
    max_distance_to_pf = np.linalg.norm(np.subtract(pf, np.sum([p0, d2*l2/2], axis = 0)))
    current_position_index = [0, int(len(lines_list)/2)] # X, Y
    final_position_index = [len(lines_list[0]) - 1, int(len(lines_list)/2)] # X, Y
    points_displaced = 0

    x_list = []
    y_list = []
    x_list_filtered = []
    y_list_filtered = []
    x_list_main = []
    y_list_main = []

    fish_prob_list = []
    fish_prob_list_filtered = []
    fish_prob_list_main = []

    # main path, which goes directly from initial to final points
    for i in range(len(lines_list[0])):
        x_list_main.append(lines_list[int(len(lines_list)/2)][i]["position"][0])
        y_list_main.append(lines_list[int(len(lines_list)/2)][i]["position"][1])
        fish_prob_list_main .append(lines_list[int(len(lines_list)/2)][i]["fish"])

    path_index = [[current_position_index[0], current_position_index[1]]]

    fish_prob_list.append(lines_list[current_position_index[1]][current_position_index[0]]["fish"])

    while current_position_index != final_position_index:
        score = -math.inf
        max_score = -math.inf
        a_max = 0
        b_max = 0
        for a in range(-1,2,1): # d2 values
            for b in range(0,2,1): # d1 values
                if a == 0 and b == 0:
                    continue # we should always move
                elif [current_position_index[0] + b ,current_position_index[1] + a] not in path_index:
                    # we get the next point data from the lines_list variables
                    try:
                        values = lines_list[current_position_index[1] + a][current_position_index[0] + b]
                        score = weights["fish"] * values["fish"] \
                        - weights["straight_line_distance"] * values["distance_to_l1"] / (l2/2) \
                        - weights["final_point_distance"] * values["distance_to_pf"] / max_distance_to_pf
                        #+ weights["fuel"] * values["fuel"] + weights["area"] * values["area"]

                    except IndexError:
                        # Position not reachable
                        continue

                    if score > max_score:
                        max_score = score
                        max_fishing = values["fish"]
                        a_max = a
                        b_max = b


        current_position_index[0] += b_max # X
        current_position_index[1] += a_max # Y
        path_index.append([current_position_index[0], current_position_index[1]]) # X Y

        fish_prob_list.append(max_fishing)

        points_displaced += 1

    step_filter = 3
    paths_coincide = True
    path_index_filtered = []
    index_to_insert = []
    counter = 0

    # filter points
    while True:

        if counter % step_filter == 0 or path_index[counter] == path_index[-1]:
            if len(path_index) - counter < step_filter and not path_index[counter] == path_index[-1]:
                step_filter = len(path_index) - counter

            try:
                index_to_insert = []
                paths_coincide = True
                for i in range(counter,counter+step_filter,1):
                    paths_coincide = (path_index[i][1] == path_index[i+1][1]) and paths_coincide
                    index_to_insert.append(path_index[i])

            except Exception:
                paths_coincide = False

            if paths_coincide:
                for index in index_to_insert:
                    path_index_filtered.append(index)

                for j in range(counter,counter+step_filter,1):
                    fish_prob_list_filtered.append(fish_prob_list[j])

                counter += step_filter

            else:
                path_index_filtered.append(path_index[counter])
                fish_prob_list_filtered.append(fish_prob_list[counter])
                counter += 1

        else:
            counter +=1

        if counter >= len(path_index):
            break

    # we pass from index to real positions
    for index in path_index:
        x_list.append(lines_list[index[1]][index[0]]["position"][0])
        y_list.append(lines_list[index[1]][index[0]]["position"][1])

    for index in path_index_filtered:
        x_list_filtered.append(lines_list[index[1]][index[0]]["position"][0])
        y_list_filtered.append(lines_list[index[1]][index[0]]["position"][1])


    return [[x_list_main, y_list_main, 'green'],[x_list, y_list, 'red'] , \
    [x_list_filtered, y_list_filtered, 'blue']], \
    [fish_prob_list, fish_prob_list_filtered, fish_prob_list_main]


def compute_feedback_data(path_points_data, fish_list, path_results, boat_features):
    """ Computes values for different paths, so we can see if will worth it to change
    the direct path to another one.
    Data is average probability of finding fish, fuel and time spent """

    # supposed constants for computing this values
    avg_speed = boat_features[0] # distance per time
    fuel_consumption = boat_features[1] # volume per speed per distance
    fuel_consumption_turning = boat_features[2] # volume per angle
    time_consumption_turning = boat_features[3] # time

    # returned data
    time_spent = []
    fuel_spent = []
    fishing_avg_prob = []
    color = []

    # time and fuel spent
    for path in path_points_data:
        fuel = 0
        time = 0
        # we append color for making later file writing easier
        color.append(path[2])

        for i in range(len(path[0]) - 1): # last point won't have distance nor angle
            # time and fuel consumption
            current_point = np.array([path[0][i], path[1][i]])
            if i == 0:
                previous_point = current_point - np.array([1, 0]) # won't work if previous_point = np.array([0,0])
            else:
                previous_point = np.array([path[0][i-1], path[1][i-1]])

            final_point = np.array([path[0][i+1], path[1][i+1]])

            distance = np.linalg.norm(final_point - current_point)
            angle = compute_angle(previous_point, current_point, final_point)

            fuel = fuel_consumption * distance * avg_speed + \
            fuel_consumption_turning * angle / 360

            time = distance / avg_speed + time_consumption_turning * angle / 360


        time_spent.append(time)
        fuel_spent.append(fuel)

    # fishing average probabilities
    for fish_probs in fish_list:
        fishing_prob =  sum(fish_probs) / float(len(fish_probs))
        fishing_avg_prob.append(fishing_prob)



    # generates file with info
    open(os.path.join(path_results, 'report.txt'), 'w').close()
    with open(os.path.join(path_results, 'report.txt'), 'w') as file:
        for i in range(len(color)):
            file.write('{} path spends {:.4f} units of time, {:.4f} of fuel and has an average fishing probability of {:.4f} \n'.format(
            color[i], time_spent[i], fuel_spent[i], fishing_avg_prob[i]))

        file.write('\n')
        file.write('{} path spends the least time \n'.format(color[time_spent.index(min(time_spent))]))
        file.write('{} path spends the least fuel \n'.format(color[fuel_spent.index(min(fuel_spent))]))
        file.write('{} path has the greatest fishing average probability'.format( \
        color[fishing_avg_prob.index(max(fishing_avg_prob))]))


    return time_spent, fuel_spent, fishing_avg_prob, color

def compute_angle(a, b, c):
    """ Computes angle between three points """

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    # because of precision issues, sometimes cosine_angle is something linke -1.000000001
    # we make sure we only pass the correct arguments to np.arccos()
    if cosine_angle > 1:
        cosine_angle = 1
    elif cosine_angle < -1:
        cosine_angle = -1

    angle = np.arccos(cosine_angle)

    return np.degrees(angle)
