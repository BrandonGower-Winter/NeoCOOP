# Note! This script will only work if the agents have the same attributes
import math

import Animate
import json
import numpy as np
import os
import pandas
import statistics
import matplotlib.pyplot as plt

from ECAgent.Environments import discreteGridPosToID


def get_csv_iteration(filename: str) -> int:
    return int(filename[filename.index('_')+1:-4])

def load_csv(filename: str) -> pandas.DataFrame:
    return pandas.read_csv(filename)


def load_csvs(folder_path: str, sort: bool = True, key=get_csv_iteration) -> [pandas.DataFrame]:

    pandas_snapshots = []

    for root, _, files in os.walk(folder_path, topdown=True):

        csv_files = [f for f in files if f[-3:] == 'csv']

        if sort:
            csv_files.sort(key=key)

        for file in csv_files:
            pandas_snapshots.append(load_csv(os.path.join(root, file)))

    return pandas_snapshots


def get_all_keys(item: dict) -> [str]:
    return [key for key in item]


def create_composite_property_as_panda(snapshots: [pandas.DataFrame], func, kwargs_to_pass: dict = {}):

    for snapshot in snapshots:
        func(snapshot, **kwargs_to_pass)


def pandas_to_animat(title: str, width: int, height: int, data: [pandas.DataFrame], property: str, filename: str,
                     fps: int = 1, x_label: str = 'X', y_label: str = 'Y', vmin: int = 0, vmax: int = 100000) -> None:

    records = [df[property].to_numpy().reshape(height, width) for df in data]
    Animate.generateAnimat(title, records, fps, vmin, vmax, filename, x_label, y_label)


def log_file_to_list(file_path: str) -> [dict]:

    log_list = []

    with open(file_path, 'r') as log_file:
        iter_dict = {}

        for line in log_file.readlines():
            keyword = line[:line.find(':')]

            if keyword == 'ITERATION':
                log_list.append(iter_dict)
                iter_dict = {}
            elif keyword == 'GES':
                vals = str.split(line[line.find(':')+1:])
                iter_dict['temp'] = float(vals[0])
                iter_dict['rainfall'] = float(vals[1])
                iter_dict['flood'] = float(vals[2])
            elif keyword == 'HOUSEHOLD.FARM':
                if 'HOUSEHOLD.FARM' not in iter_dict:
                    iter_dict['HOUSEHOLD.FARM'] = 0
                    iter_dict['FARM_LOC'] = []

                iter_dict['HOUSEHOLD.FARM'] += 1

                vals = str.split(line[line.find(':')+1:])
                iter_dict['FARM_LOC'].append(int(vals[1]))

            elif keyword in iter_dict:
                iter_dict[keyword] += 1
            else:
                iter_dict[keyword] = 1

    return log_list


def dynamic_farm_animat(parser, pixels):

    environment_snapshots = load_csvs(parser.path + '/environment')
    log_list = log_file_to_list(parser.path + '/events.log')

    count = 0

    def dynamic_farm(df, **kwargs):
        land = []
        nonlocal count

        for i in range(len(df)):
            if df['isSettlement'][i] != -1:
                land.append(300)
            elif 'FARM_LOC' in log_list[count] and i in log_list[count]['FARM_LOC']:
                land.append(300)
            else:
                land.append((1.0 - kwargs['pixels'][i]) * 250)

        df['dynamic_ownership'] = land
        count += 1

    create_composite_property_as_panda(environment_snapshots, dynamic_farm, {'pixels': pixels})

    pandas_to_animat('Settlement and Farm Locations on Map', parser.width, parser.height, environment_snapshots,
                     'dynamic_ownership',
                     parser.path + '/environment_plots/dynamic_farm_animat', 10, vmin=0, vmax=300)



if __name__ == '__main__':
    import argparse
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='The path to the folder containing all of the generated data', type=str)
    parser.add_argument('width', help='The width of the map.', type=int)
    parser.add_argument('height', help='The height of the map.', type=int)
    parser.add_argument('map')

    parser = parser.parse_args()

    im_path = parser.map
    im = Image.open(im_path).convert('L')

    pixels = []

    for y in range(parser.height):
        for x in range(parser.width):
            pixels.append(im.getpixel((x, y)) / 255.0)


    dynamic_farm_animat(parser, pixels)
