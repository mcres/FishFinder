from matplotlib import pyplot as plt
import numpy as np
import math
import random
import os

class Map():

    def __init__(self, conf, path_to_results):

        self.start_x = conf["map"]["start_x"]  
        self.start_y = conf["map"]["start_y"]  
        self.end_x = conf["map"]["end_x"] 
        self.end_y = conf["map"]["end_y"]  
        self.displacement = conf["map"]["displacement"]  
        self.weights = conf["weights"] 
        self.show_points = conf["map"]["show_points"] 

        self.avg_speed = conf["boat_features"]["avg_speed"] # distance per time
        self.fuel_consumption = conf["boat_features"]["fuel_consumption"] # volume per speed per distance
        self.fuel_consumption_turning = conf["boat_features"]["fuel_consumption_turning"] # volume per angle
        self.time_consumption_turning = conf["boat_features"]["time_consumption_turning"] # time
        
        self.path_to_results = path_to_results

        self.x_list = []
        self.y_list = []
        self.x_list_filtered = []
        self.y_list_filtered = []
        self.x_list_main = []
        self.y_list_main = []

        self.fish_prob_list = []
        self.fish_prob_list_filtered = []
        self.fish_prob_list_main = []


    def generate_map(self):
        self.create_grid()
        self.create_path()
        self.show_grid()
        self.compute_feedback_data()

    def plot(self, csv_list, n_buoys):
        """Plots current position and signal of buoys"""
        #TODO: implement with pandas and matplotlib
        pass

    def create_grid(self): #(self.start_x, self.start_y, self.end_x, self.end_y, self.displacement):
        """ It takes two points as an input and returns a list of points and its properties,
        which make up a grid of a close area between those two points """

        self.p0 = np.array([self.start_x, self.start_y]) # initial point of the path
        self.pf = np.array([self.end_x, self.end_y]) # final point of the path

        self.l1 = np.linalg.norm(np.subtract(self.pf,self.p0))
        self.l2 = 0.2 * self.l1

        # we define the 2 unit vectors self.d1 and self.d2 in which directions we are moving along
        # then we apply to them the size of the desired displacement
        self.d1 = np.subtract(self.pf,self.p0)/(np.linalg.norm(np.subtract(self.pf,self.p0)))

        self.d2 = np.array([-self.d1[1], self.d1[0]])
        self.d2 = self.d2/(np.linalg.norm(self.d2))


        # we create our grid moving with self.d1 and self.d2
        # each point of the grid has the following properties:
        # position, distance to self.d1, distance to self.pf, velocity
        n_displacements_2 = 0
        self.lines_list = []

        # variables for making easier the plotting afterwards
        self.x_list_grid = []
        self.y_list_grid = []
        time_n = 1

        while True:
            current_point = np.subtract(self.p0,self.d2*(n_displacements_2*self.displacement))
            current_length_2 = np.linalg.norm(np.subtract(current_point, self.p0))
            if current_length_2 > self.l2/2:
                if time_n == 2:
                    break
                else:
                    time_n += 1
                    self.lines_list = list(reversed(self.lines_list))
                    self.d2 = -1 * self.d2
                    n_displacements_2 = 1
                    current_point = np.subtract(self.p0,self.d2*(n_displacements_2*self.displacement))


            line_points = []
            self.x_list_grid.append(current_point[0])
            self.y_list_grid.append(current_point[1])

            #TODO: add fish real prediction
            line_points.append({
            "position": current_point.tolist(),
            "distance_to_l1": n_displacements_2*self.displacement,
            "distance_to_pf": np.linalg.norm(np.subtract(self.pf,current_point)),
            "fish": random.random()
            })


            initial_point_1 = current_point
            while True:
                current_point = np.sum([current_point, self.d1 * self.displacement], axis=0)
                current_length_1 = np.linalg.norm(np.subtract(current_point, initial_point_1))
                if current_length_1 >= self.l1:
                    current_point = self.pf - self.d2*n_displacements_2*self.displacement
                    self.x_list_grid.append(current_point[0])
                    self.y_list_grid.append(current_point[1])
                    #TODO: add fish real prediction
                    line_points.append({
                    "position": current_point.tolist(),
                    "distance_to_l1": n_displacements_2*self.displacement,
                    "distance_to_pf": np.linalg.norm(np.subtract(self.pf,current_point)),
                    "fish": random.random()
                    })
                    break




                self.x_list_grid.append(current_point[0])
                self.y_list_grid.append(current_point[1])

                #TODO: add fish real prediction
                line_points.append({
                "position": current_point.tolist(),
                "distance_to_l1": n_displacements_2*self.displacement,
                "distance_to_pf": np.linalg.norm(np.subtract(self.pf,current_point)),
                "fish": random.random()
                })

            self.lines_list.append(line_points)
            n_displacements_2 += 1


    def fuel_prediction(self): #(initial_pos, self.d1, self.d2, self.avg_speed):
        """ Makes a prediction of the fuel spent given currents velocity, initial and
        final position and average speed. The idea is to decompose the velocity in vectors
        parallel to self.d1 and self.d2 and see how much they help guiding the boat into the final point, giving
        us an idea of the fuel spent """



        return 0


    def show_grid(self): #(grid_points, show_points, path_points, self.path_to_results):
        """ It shows each point in the map with a different color in function of
        its probability """

        if not os.path.exists(self.path_to_results):
            os.mkdir(self.path_to_results)

        fig = plt.figure()



        if self.show_points == 1:
            plt.scatter(self.x_list_grid, self.y_list_grid, c='blue')


        plt.plot(self.x_list_main, self.y_list_main, 'green', label='straight path')
        plt.plot(self.x_list, self.y_list, 'red', label='first path')
        plt.plot(self.x_list_filtered, self.y_list_filtered, 'blue', label='filtered path')
        plt.title('Paths')
        plt.ylabel('Latitude')
        plt.xlabel('Longitude')
        #plt.legend()

        fig.savefig(os.path.join(self.path_to_results, 'Paths.png') )

    def create_path(self): #weights, path_data, show_path=True):
        """ Given the first parameter returned from create_grid() and
        a weight list, it creates the path the boat should follow from
        its initial to its final position in order to maximize its
        fish capture along the way """


        # control variables
        max_distance_to_pf = np.linalg.norm(np.subtract(self.pf, np.sum([self.p0, self.d2*self.l2/2], axis = 0)))
        current_position_index = [0, int(len(self.lines_list)/2)] # X, Y
        final_position_index = [len(self.lines_list[0]) - 1, int(len(self.lines_list)/2)] # X, Y
        points_displaced = 0
        max_fishing = 0

        # main path, which goes directly from initial to final points
        for i in range(len(self.lines_list[0])):
            self.x_list_main.append(self.lines_list[int(len(self.lines_list)/2)][i]["position"][0])
            self.y_list_main.append(self.lines_list[int(len(self.lines_list)/2)][i]["position"][1])
            self.fish_prob_list_main .append(self.lines_list[int(len(self.lines_list)/2)][i]["fish"])

        path_index = [[current_position_index[0], current_position_index[1]]]

        self.fish_prob_list.append(self.lines_list[current_position_index[1]][current_position_index[0]]["fish"])

        while current_position_index != final_position_index:
            score = -float('inf')
            max_score = -float('inf')
            a_max = 0
            b_max = 0
            for a in range(-1,2,1): # self.d2 values
                for b in range(0,2,1): # self.d1 values
                    if a == 0 and b == 0:
                        continue # we should always move
                    elif [current_position_index[0] + b ,current_position_index[1] + a] not in path_index:
                        # we get the next point data from the self.lines_list variables
                        try:
                            values = self.lines_list[current_position_index[1] + a][current_position_index[0] + b]
                            score = self.weights["fish"] * values["fish"] \
                            - self.weights["straight_line_distance"] * values["distance_to_l1"] / (self.l2/2) \
                            - self.weights["final_point_distance"] * values["distance_to_pf"] / max_distance_to_pf
                            #+ self.weights["fuel"] * values["fuel"] + self.weights["area"] * values["area"]

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

            self.fish_prob_list.append(max_fishing)

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
                        self.fish_prob_list_filtered.append(self.fish_prob_list[j])

                    counter += step_filter

                else:
                    path_index_filtered.append(path_index[counter])
                    self.fish_prob_list_filtered.append(self.fish_prob_list[counter])
                    counter += 1

            else:
                counter +=1

            if counter >= len(path_index):
                break

        # we pass from index to real positions
        for index in path_index:
            self.x_list.append(self.lines_list[index[1]][index[0]]["position"][0])
            self.y_list.append(self.lines_list[index[1]][index[0]]["position"][1])

        for index in path_index_filtered:
            self.x_list_filtered.append(self.lines_list[index[1]][index[0]]["position"][0])
            self.y_list_filtered.append(self.lines_list[index[1]][index[0]]["position"][1])


        return [[self.x_list_main, self.y_list_main, 'green'],[self.x_list, self.y_list, 'red'] , \
        [self.x_list_filtered, self.y_list_filtered, 'blue']], \
        [self.fish_prob_list, self.fish_prob_list_filtered, self.fish_prob_list_main]


    def compute_feedback_data(self): #(path_points_data, fish_list, self.path_to_results, boat_features):
        """ Computes values for different paths, so we can see if will worth it to change
        the direct path to another one.
        Data is average probability of finding fish, fuel and time spent """

        # supposed constants for computing this values
        

        # returned data
        time_spent = []
        fuel_spent = []
        fishing_avg_prob = []
        color = []

        path_points_data = [[self.x_list_main, self.y_list_main, 'green'],
        [self.x_list, self.y_list, 'red'],
        [self.x_list_filtered, self.y_list_filtered, 'blue']]

        fish_list = [self.fish_prob_list, 
            self.fish_prob_list_filtered, 
            self.fish_prob_list_main]

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
                angle = self.compute_angle(previous_point, current_point, final_point)

                fuel = self.fuel_consumption * distance * self.avg_speed + \
                self.fuel_consumption_turning * angle / 360

                time = distance / self.avg_speed + self.time_consumption_turning * angle / 360


            time_spent.append(time)
            fuel_spent.append(fuel)

        # fishing average probabilities
        for fish_probs in fish_list:
            fishing_prob =  sum(fish_probs) / float(len(fish_probs))
            fishing_avg_prob.append(fishing_prob)



        # generates file with info
        open(os.path.join(self.path_to_results, 'report.txt'), 'w').close()
        with open(os.path.join(self.path_to_results, 'report.txt'), 'w') as file:
            for i in range(len(color)):
                file.write('{} path spends {:.4f} units of time, {:.4f} of fuel and has an average fishing probability of {:.4f} \n'.format(
                color[i], time_spent[i], fuel_spent[i], fishing_avg_prob[i]))

            file.write('\n')
            file.write('{} path spends the least time \n'.format(color[time_spent.index(min(time_spent))]))
            file.write('{} path spends the least fuel \n'.format(color[fuel_spent.index(min(fuel_spent))]))
            file.write('{} path has the greatest fishing average probability'.format( \
            color[fishing_avg_prob.index(max(fishing_avg_prob))]))


    def compute_angle(self, a, b, c):
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

    def get_straight_path(self):
        return self.x_list_main, self.y_list_main

    def get_first_path(self):
        return self.x_list, self.y_list

    def get_filtered_path(self):
        return self.x_list_filtered, self.y_list_filtered

    def get_grid_points(self):
        return self.x_list_grid, self.y_list_grid