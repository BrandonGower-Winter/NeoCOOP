import argparse
import json
import matplotlib.pyplot as pyplot
import numpy as np

import os

from CythonFunctions import CGlobalEnvironmentCurves


def scenario_to_curve(scenario, i):

    if i > 1000:
        return 1.0

    if scenario == 'scenario1':
        return 1.0
    elif scenario == 'scenario2':
        return CGlobalEnvironmentCurves.linear_lerp(0., 1., i)
    elif scenario == 'scenario3':
        return CGlobalEnvironmentCurves.cosine_lerp(0., 1., i, 16)
    else:
        return CGlobalEnvironmentCurves.linear_modified_dsinusoidal_lerp(0., 1., i, 16, 1, 1)


def get_scenario_data(runs: {}):

    placeholder = np.zeros((2000, len(runs)), dtype=float)
    population_in_delta_placeholder = np.zeros((2000, len(runs)), dtype=float)

    index = 0
    for seed in runs:
        for i in range(len(runs[seed]['population']['total'])):
            placeholder[i][index] = runs[seed]['population']['total'][i]
            population_in_delta_placeholder[i][index] = runs[seed]['pop_dist']['delta'][i]
        index += 1

    data = np.zeros((8, 2000), dtype=float)

    for i in range(2000):
        data[0][i] = np.mean(placeholder[i])
        data[1][i] = np.std(placeholder[i])
        data[2][i] = (data[1][i] / data[0][i]) * 100.0
        lower, upper = data[0][i] - data[1][i],  data[0][i] + data[1][i]
        data[3][i] = len([s for s in placeholder[i] if lower < s < upper]) / 50.0
        data[4][i] = len([s for s in placeholder[i] if lower - data[1][i] < s < upper + data[1][i]]) / 50.0

        # Resistance Metric
        if i != 0:
            data[5][i] = (data[0][i] - data[0][0]) / data[0][i] * 100.0

        data[6][i] = np.mean(population_in_delta_placeholder[i])
        data[7][i] = data[0][i] - data[6][i]

    return data


def write_plot(agent_types: [], scenario, filename, data, title: str, index: int, x_axis: str, y_axis: str,
               legend: str = 'lower right'):

    fig, ax = pyplot.subplots()
    ax.set_title(title)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    iterations = np.arange(2000)

    for agent_type in agent_types:
        ax.plot(iterations, data[agent_type][index], label=agent_type)

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

    write_plot([a for a in data], parser.scenario, '%s/population' % parser.output, data,
               'Total Population of Agent Types for %s averaged over 50 simulation runs.' % parser.scenario,
               0, 'iterations', 'Population')
    write_plot([a for a in data], parser.scenario, '%s/SD' % parser.output, data,
               'SD of Population of Agent Types for %s averaged over 50 simulation runs.' % parser.scenario,
               1, 'iterations', 'SD')
    write_plot([a for a in data], parser.scenario, '%s/RSD' % parser.output, data,
               'RSD of Population of Agent Types for %s averaged over 50 simulation runs.' % parser.scenario,
               2, 'iterations', 'RSD(%)')
    write_plot([a for a in data], parser.scenario, '%s/onestd' % parser.output, data,
               '%s of simulation runs within 1 STD of the mean\nfor %s averaged over 50 simulation runs.' % ('%', parser.scenario),
               3, 'iterations', '%')
    write_plot([a for a in data], parser.scenario, '%s/twostd' % parser.output, data,
               '%s of simulation runs within 2 STD of the mean\nfor %s averaged over 50 simulation runs.' % ('%', parser.scenario),
               4, 'iterations', '%')
    write_plot([a for a in data], parser.scenario, '%s/resistance' % parser.output, data,
               'Relative Resistance of population \nfor %s averaged over 50 simulation runs.' % parser.scenario,
               5, 'iterations', '%')

    write_plot([a for a in data], parser.scenario, '%s/delta_pop' % parser.output, data,
               'Total Population of Agent Types within the Delta \nfor %s averaged over 50 simulation runs.' % parser.scenario,
               6, 'iterations', 'Population')
    write_plot([a for a in data], parser.scenario, '%s/out_delta_pop' % parser.output, data,
               'Total Population of Agent Types outside the Delta \nfor %s averaged over 50 simulation runs.' % parser.scenario,
               7, 'iterations', 'Population')


if __name__ == '__main__':
    main()
