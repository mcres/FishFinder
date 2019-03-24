# -*- coding: utf-8 -*-
import csv
import random
import sys
import os
import datetime
import pandas as pd

#TODO make sure each year recieves an equal amount of data
#TODO add timestamp to each csv file, and save them in /data

class DataGenerator():

    def __init__(self, conf, save_to_csv=True, year=2010, year_range=7):

        self.data = 0
        self.buoy = 1
        self.now = datetime.datetime.now()
        self.data_buoys = conf["map"]["data_per_buoy"] - 1
        self.year_start = year
        self.year_range = year_range

        self.cols_order = ['ID','minute','hour','day','month','year','temperature','lat','lon','velocity','fish']
        self.row_dict = {}
        self.save_to_csv = save_to_csv
        self.df = pd.DataFrame()

        self.num_buoys = conf["map"]["number_of_buoys"]
        self.data_per_buoy = conf["map"]["data_per_buoy"]
        self.width_world = conf["map"]["world_width"]
        self.height_world = conf["map"]["world_height"]


    def generate_data(self):
        """Generates data for our model, where
        num_buoys are the number of buoys on the system,
        data_buoys are the number of entries per buoy,
        and width_world and height_world are the dimensions of the surface
        where both the buoys and the ships are supposed to be"""        

        while self.buoy <= self.num_buoys:

            minute = round(random.random()*60)
            hour = round(random.random()*24)
            day = round(random.random()*30)
            month = round(random.random()*12)
            year = self.year_start + round(random.random()*self.year_range)
            #operating temperature between 0ºC and 40ºC
            temp = round(random.random()*40)         
            lat = round(random.random()*self.width_world)
            lon = round(random.random()*self.height_world)
            vel = round(random.random()*10)
            fish = round(random.random())

            self.row_dict = {'ID': self.buoy, 'minute': int(minute), 'hour': int(hour),
            'day': int(day), 'month': int(month), 'year': int(year), 'temperature': int(temp),
            'lat': int(lat), 'lon': int(lon), 'velocity': int(vel), 'fish': int(fish)}

            self.df = self.df.append(self.row_dict, ignore_index=True)

            self.data += 1
            if self.data == self.data_buoys:
                self.buoy+=1
                self.data = 0

        # write current data from buoys
        for i in range(self.num_buoys):
            buoy = i + 1
            minute = self.now.minute
            hour = self.now.hour
            day = self.now.day
            month = self.now.month
            year = self.now.year
            temp = round(random.random()*40)         
            lat = round(random.random()*self.width_world)
            lon = round(random.random()*self.height_world)
            vel = round(random.random()*10)
            fish = round(random.random())

            self.row_dict = {'ID': buoy, 'minute': int(minute), 'hour': int(hour),
            'day': int(day), 'month': int(month), 'year': int(year), 'temperature': int(temp),
            'lat': int(lat), 'lon': int(lon), 'velocity': int(vel), 'fish': int(fish)}

            self.df = self.df.append(self.row_dict, ignore_index=True)

        self.df = self.df[self.cols_order]

        if self.save_to_csv:
            self.df.to_csv('generated_data.csv')

        return self.df
