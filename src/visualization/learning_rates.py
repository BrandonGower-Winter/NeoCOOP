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

size = 11

def get_learning_rates(data : [], nparray):

    for line in data:
        lsplit = line.split(':')
        ids = lsplit[0].split(',')
        ldata = lsplit[1].split(',')

        x = int(float(ids[0]) / 0.01)
        y = int(float(ids[1]) / 0.01)

        pop = float(ldata[0])
        res = float(ldata[1])
        gini = float(ldata[2])
        nparray[0][x][y] += pop
        nparray[1][x][y] += res
        nparray[2][x][y] += gini

    return nparray


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

    data = np.zeros((3, 11, 11))
    count = 0
    for root, _, files in os.walk(parser.path):
        for run in files:
            with open(os.path.join(root, run)) as file:
                data = get_learning_rates(file.readlines(), data)
                count += 1

    data /= count

    x_labels = [0.0, None, 0.02, None, 0.04, None, 0.06, None, 0.08, None, 0.1]
    y_labels = x_labels
    tick_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    max_labels = np.round(np.linspace(0.0, 0.75, 16), 2)

    fig, ax = pyplot.subplots(dpi=200)
    ax.set_title('')
    ax.set_xlabel('Stubbornness')
    ax.set_ylabel('Conformity')

    # Generate the pix map
    plot = ax.imshow(np.transpose(data[0]), origin='lower', interpolation='none')
    clb = fig.colorbar(plot)
    clb.ax.set_title('Households')
    ax.set_aspect('auto')

    ax.set_xticks(tick_index)
    ax.set_xticklabels(x_labels)

    ax.set_yticks(tick_index)
    ax.set_yticklabels(y_labels)

    fig.savefig('%s/learning_rates_population' % parser.output)
    pyplot.close(fig)

    fig, ax = pyplot.subplots(dpi=200)
    ax.set_title('')
    ax.set_xlabel('Stubbornness')
    ax.set_ylabel('Conformity')

    # Generate the pix map
    plot = ax.imshow(np.transpose(data[1]), origin='lower', interpolation='none')
    clb = fig.colorbar(plot)
    clb.ax.set_title('Resources')
    ax.set_aspect('auto')

    ax.set_xticks(tick_index)
    ax.set_xticklabels(x_labels)

    ax.set_yticks(tick_index)
    ax.set_yticklabels(y_labels)

    fig.savefig('%s/learning_rates_resources' % parser.output)
    pyplot.close(fig)

    fig, ax = pyplot.subplots(dpi=200)
    ax.set_title('')
    ax.set_xlabel('Stubbornness')
    ax.set_ylabel('Conformity')

    # Generate the pix map
    plot = ax.imshow(np.transpose(data[2]), origin='lower', interpolation='none')
    clb = fig.colorbar(plot)
    clb.ax.set_title('Gini Index')
    ax.set_aspect('auto')

    ax.set_xticks(tick_index)
    ax.set_xticklabels(x_labels)

    ax.set_yticks(tick_index)
    ax.set_yticklabels(y_labels)

    fig.savefig('%s/learning_rates_gini' % parser.output)
    pyplot.close(fig)

if __name__ == '__main__':
    main()