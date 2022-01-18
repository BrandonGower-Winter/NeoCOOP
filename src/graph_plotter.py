import argparse
import json
import matplotlib.pyplot as pyplot
import numpy as np

import os


def get_scenario_data(runs: {}):

    placeholder = np.zeros((2000, len(runs)), dtype=float)
    peer_placeholder = np.zeros((2000, len(runs)), dtype=float)
    sub_placeholder = np.zeros((2000, len(runs)), dtype=float)

    index = 0
    for seed in runs:
        for i in range(len(runs[seed]['population']['total'])):
            placeholder[i][index] = runs[seed]['population']['number'][i]
            peer_placeholder[i][index] = runs[seed]['peer_transfer']['mean'][i]
            sub_placeholder[i][index] = runs[seed]['sub_transfer']['mean'][i]
        index += 1

    data = np.zeros((6, 2000), dtype=float)

    for i in range(2000):
        data[0][i] = np.mean(placeholder[i])
        data[1][i] = np.std(placeholder[i])

        data[2][i] = np.mean(peer_placeholder[i])
        data[3][i] = np.std(peer_placeholder[i])

        data[4][i] = np.mean(sub_placeholder[i])
        data[5][i] = np.std(sub_placeholder[i])

    return data


def write_plot(agent_types: [], filename, data, title: str, index: [int], x_axis: str, y_axis: str,
               legend: str = 'lower right', show_std: int = -1, data_types={}):

    fig, ax = pyplot.subplots()
    ax.set_title(title)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    iterations = np.arange(2000) * 5

    for agent_type in agent_types:
        if len(index) > 1:
            for i in range(len(index)):
                ax.plot(iterations, data[agent_type][index[i]], label=data_types[agent_type][i])
        else:
            ax.plot(iterations, data[agent_type][index[0]], label=agent_type)

    ax.legend(loc=legend)
    ax.set_aspect('auto')

    fig.savefig(filename)
    pyplot.close(fig)


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
                data[agent_type[0:-5]] = get_scenario_data(json.load(json_file))

    data_types = {}

    for a in data:
        data_types[a] = ['Peer', 'Sub']

    write_plot(['R-HIGH', 'F-HIGH', 'A-HIGH', 'S-HIGH'], '%s/HIGH_population' % parser.output, data,
               'Total Population of Initial Peer / Sub Distributions for \nHIGH Frequency Environmental Stress Scenarios.',
               [0], 'Iteration', 'Population')
    write_plot(['R-LOW', 'F-LOW', 'A-LOW', 'S-LOW'], '%s/LOW_population' % parser.output, data,
               'Total Population of Initial Peer / Sub Distributions for \nLOW Frequency Environmental Stress Scenarios.',
               [0], 'Iteration', 'Population')
    write_plot(['R-UNIFORM', 'F-UNIFORM', 'A-UNIFORM', 'S-UNIFORM'], '%s/UNIFORM_population' % parser.output, data,
               'Total Population of Initial Peer / Sub Distributions for \nUNIFORM Environmental Stress Scenarios.',
               [0], 'Iteration', 'Population')

    for a in data:
        write_plot([a], '%s/%s_transfer_chance' % (parser.output, a), data,
                                                                'Peer and Sub Transfer Properties for %s Scenario' % a,
                                                          [2, 4], 'Iteration', 'Probability (%)', data_types=data_types)


if __name__ == '__main__':
    main()
