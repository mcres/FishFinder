# -*- coding: utf-8 -*-
import csv
import random
import sys
import os
import datetime

#TODO make sure each year recieves an equal amount of data

def generate_data(num_buoys, data_buoys, width_world, height_world):
    """Generates data for our model, where
    num_buoys are the number of buoys on the system,
    data_buoys are the number of entries per buoy,
    and width_world and height_world are the dimensions of the surface
    where both the buoys and the ships are supposed to be"""

    # control variables for the while loop
    data = 0
    buoy = 1
    now = datetime.datetime.now()
    data_buoys = data_buoys - 1
    year_start = 2010
    year_range = 7

    open('data_buoys.csv','w').close()    # creates file in root directory, not in functions
    with open('data_buoys.csv', 'w') as csvfile:
        fieldnames = ['ID', 'minute', 'hour', 'day', 'month', 'year', 'temperature',
        'lat', 'lon', 'velocity', 'fish']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        while buoy <= num_buoys:

            minute = round(random.random()*60)
            hour = round(random.random()*24)
            day = round(random.random()*30)
            month = round(random.random()*12)
            year = year_start + round(random.random()*year_range)
            temp = round(random.random()*40)         #operating temperature between 0ºC and 40ºC
            lat = round(random.random()*width_world)
            lon = round(random.random()*height_world)
            vel = round(random.random()*10)
            fish = round(random.random())

            writer.writerow({'ID': buoy, 'minute': int(minute), 'hour': int(hour),
            'day': int(day), 'month': int(month), 'year': int(year), 'temperature': int(temp),
            'lat': int(lat), 'lon': int(lon), 'velocity': int(vel), 'fish': int(fish)})

            data += 1
            if data == data_buoys:
                buoy+=1
                data = 0

        # write current data from buoys
        for i in range(num_buoys):
            buoy = i + 1
            minute = now.minute
            hour = now.hour
            day = now.day
            month = now.month
            year = now.year
            temp = round(random.random()*40)         #operating temperature between 0ºC and 40ºC
            lat = round(random.random()*width_world)
            lon = round(random.random()*height_world)
            vel = round(random.random()*10)
            fish = round(random.random())

            writer.writerow({'ID': buoy, 'minute': int(minute), 'hour': int(hour),
            'day': int(day), 'month': int(month), 'year': int(year), 'temperature': int(temp),
            'lat': int(lat), 'lon': int(lon), 'velocity': int(vel), 'fish': int(fish)})




def return_parameters():
    return len(fieldnames) - 1

if __name__ == '__main__':
    generate_data(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
