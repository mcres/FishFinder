## Description

This project uses Machine Learning in order to compute the probabilities of a ship to find
a shoal of fish given a specific trajectory. It uses artificially created data of buoys which are
supposed to give the following data: ID of the own buoy, time (i.e. minutes, hours, day, month and year), temperature, position, velocity and detection or absence of a shoal of fish.

The goal is to predict a ship's best path given a starting and ending point, so the model can tell the areas between these two points where there is the best probability of fishing as well as giving information about the fuel and time spent for each trajectory.

When the computations for this are finished, you can check your results in 'Results/date_and_time_when_you_run_the_script'. There will be a .png file showing three paths and a .txt file showing and comparing some of its characteristics.


## How it works:

In order to train a DNN, we need data. In this case, the approach is to generate some artificial data for this purpose. We save this data into '/data_buoys.csv' so the user can check it any moment, and then feed it to our DNN so we can train it using Keras.
Once trained, we need to define our algorithm to find the best path. That is the following:

- We suppose that our world (sizes previously given by the user) is made up only by water (no islands or whatever). Inside this world we define a line going from the initial point to the final point of the boat. That's the main line. Then we define a perpendicular line which is only 20% large. We can then form a rectangle that we divide into a lot of points. It helps to check the .png file I mentioned before to understand this.
-To each of this points, we assign the following values: coordinates, distance to the main line, distance to the final point and the probability of fish given by the DNN.
- Starting from the initial point, we give the boat the option to move to 5 different positions (forward, up, down, forward-up and forward-down) until it gets to the final point. Each of these points have a score given by the sum of their properties, each one multiplied by a specific weight.
- For each path, we compute how much time and how many fuel it uses, as well as the average fishing probability predicted by the DNN.

## How to use it:

It is quite simple.
Go to the root directory of the repository with the command line and execute

`$pip3 install -r requirements.txt` 

Jupyter Notebook and demo.py should give an idea of how to use this package.

As there's no GUI at the moment, we can configure some parameters in the '/conf.json' file.

"map"
  - "number_of_buoys": we'll feed our DNN with the data collected by 'number_of_buoys' buoys
  - "data_per_buoy": how many time, temperature, position, velocity and fish detection series of data will be provided by each buoy. The program makes sure that one of this series of data is updated (that is, 'fresh' data).
  - Our world's size is defined by "world_width" and "world_height".
  - Likewise, initial and final points will be defined by "start_x", "start_y", "end_x" and "end_y", respectively.
  - "displacement": distance between consecutive points in the main line.
  - "show_points": '/Results/Paths.png' will show the points which make up the grid if this is 1. Otherwise, it won't.

"boat_features"
  - "avg_speed": the average speed of the boat.
  - "fuel_consumption": volume per speed per distance.
  - "fuel_consumption_turning": increase of fuel consumption at a turning point.
  - "time_consumption_turning": increase of time spent at a turning point.

"weights"
  - "fish": the higher, the more probable it will pass by points with high fishing probabilities.
  - "straight_line_distance": the higher, the more probable it won't go very far from the main line (that is, the line connecting the starting and final points).
  - "fuel": the higher, the more probable the path will spend less fuel.
  - "area": the higher, the more probable it will pass by areas with high fishing probabilities.
  - "final_point_distance": this assures us that the path will converge into the final point that we want to go to.
