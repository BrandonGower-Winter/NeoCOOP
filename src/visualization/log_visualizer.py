import argparse
import json
import matplotlib
import matplotlib.pyplot as pyplot

from matplotlib.ticker import FormatStrFormatter
import numpy as np
import scikit_posthocs as sp

from scipy import stats

import math
import os

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

array_size = 2500
iteration_multiplier = 1

def get_log_data(data : {}):
    perc = np.zeros(array_size)
    moves = np.zeros(array_size // 5)
    for key in data:
        for i, item in enumerate(data[key]):
            FARM = item['HOUSEHOLD.FARM'] if 'HOUSEHOLD.FARM' in item else 0
            FORAGE = item['HOUSEHOLD.FORAGE'] if 'HOUSEHOLD.FORAGE' in item else 0
            TOTAL = FARM + FORAGE
            perc[i] += FARM / TOTAL * 100.0 if TOTAL > 0 else 0.0
            if i % 5 == 0:
                RAND_MOVE = item['HOUSEHOLD.MOVE.RANDOM'] if 'HOUSEHOLD.MOVE.RANDOM' in item else 0
                SETTLE_MOVE = item['HOUSEHOLD.MOVE.SETTLEMENT'] if 'HOUSEHOLD.MOVE.SETTLEMENT' in item else 0
                moves[i // 5] += RAND_MOVE + SETTLE_MOVE
    return [perc, moves]

def main():
    # Process the params
    print("Parsing arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='The path to the folder containing all of the processed data',
                        required=True)
    parser.add_argument('-o', '--output', help='The path to write all of the processed data to.',
                        required=True)
    parser.add_argument('-s', '--scenario', help='The scenario name',
                        required=True)
    parser.add_argument('-v', '--verbose', help='Will print out informative information to the terminal.',
                        action='store_true')

    parser = parser.parse_args()

    data = {}
    for root, _, files in os.walk(parser.path):
        for agent_type in files:
            with open(os.path.join(root, agent_type)) as json_file:
                key = agent_type[0:-5]
                json_file = json.load(json_file)
                data[key] = get_log_data(json_file)
                for item in data[key]:
                    item /= len(json_file)

    iterations = np.arange(array_size) * iteration_multiplier


    fig, ax = pyplot.subplots(dpi=200)
    ax.set_title('')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fraction of FARM Actions (%)')

    for agent_type in data:
        ax.plot(iterations, data[agent_type][0], label=agent_type)


    ax.legend(loc='lower right')
    ax.set_aspect('auto')

    fig.savefig(parser.output + '/logs/farm_percentage')
    pyplot.close(fig)

    iterations = np.arange(array_size // 5) * iteration_multiplier * 5
    fig, ax = pyplot.subplots(dpi=200)
    ax.set_title('')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Number of MOVE Actions')

    for agent_type in data:
        ax.plot(iterations, data[agent_type][1], label=agent_type)


    ax.legend(loc='upper right')
    ax.set_aspect('auto')

    fig.savefig(parser.output + '/logs/move_actions')
    pyplot.close(fig)

if __name__ == '__main__':
    main()