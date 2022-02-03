import argparse
import json
import matplotlib
import matplotlib.pyplot as pyplot
import numpy as np
import scikit_posthocs as sp

from scipy import stats

import math
import os

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def get_scenario_data(runs: {}):

    placeholder = np.zeros((2000, len(runs)), dtype=float)
    placeholder_pop = np.zeros((2000, len(runs)), dtype=float)
    peer_placeholder = np.zeros((2000, len(runs)), dtype=float)
    sub_placeholder = np.zeros((2000, len(runs)), dtype=float)
    gini_pop = np.zeros((2000, len(runs)), dtype=float)
    settlements_pop = np.zeros((2000, len(runs)), dtype=float)

    sub_requests = np.zeros((2000, len(runs)), dtype=float)
    auth_requests = np.zeros((2000, len(runs)), dtype=float)
    peer_requests = np.zeros((2000, len(runs)), dtype=float)

    peer_dst_placeholder = np.zeros((2000, len(runs), 10), dtype=float)
    sub_dst_placeholder = np.zeros((2000, len(runs), 10), dtype=float)

    index = 0
    for seed in runs:
        for i in range(len(runs[seed]['population']['total'])):
            placeholder[i][index] = runs[seed]['population']['number'][i]
            placeholder_pop[i][index] = runs[seed]['population']['total'][i]
            peer_placeholder[i][index] = runs[seed]['peer_transfer']['mean'][i]
            sub_placeholder[i][index] = runs[seed]['sub_transfer']['mean'][i]
            gini_pop[i][index] = runs[seed]['gini']['gini'][i]
            settlements_pop[i][index] = runs[seed]['settlements']['count'][i]
            auth_requests[i][index] = runs[seed]['logs']['AUTH'][i]
            sub_requests[i][index] = runs[seed]['logs']['SUB'][i]
            peer_requests[i][index] = runs[seed]['logs']['PEER'][i]

            peer_dst_placeholder[i][index] = np.array(runs[seed]['peer_transfer']['dist'][i])
            sub_dst_placeholder[i][index] = np.array(runs[seed]['sub_transfer']['dist'][i])

        index += 1

    line_data = np.zeros((27, 2000), dtype=float)
    bar_data = np.zeros((2, 3, 10), dtype=float)
    stats_data = np.zeros((3,50), dtype=float)

    mask = peer_placeholder[0] < sub_placeholder[0]

    for i in range(2000):
        line_data[0][i] = np.mean(placeholder[i])
        line_data[1][i] = np.std(placeholder[i])

        line_data[2][i] = np.mean(peer_placeholder[i])
        line_data[3][i] = np.std(peer_placeholder[i])

        line_data[4][i] = np.mean(sub_placeholder[i])
        line_data[5][i] = np.std(sub_placeholder[i])

        temp = (peer_placeholder[i] - peer_placeholder[0])/(peer_placeholder[0]) * 100.0
        line_data[6][i] = np.mean(temp)
        line_data[7][i] = np.std(temp)

        temp = (sub_placeholder[i] - sub_placeholder[0])/(sub_placeholder[0]) * 100.0
        line_data[8][i] = np.mean(temp)
        line_data[9][i] = np.std(temp)

        temp = (placeholder_pop[i] - placeholder_pop[0]) / placeholder_pop[0]
        line_data[10][i] = np.mean(temp)
        line_data[11][i] = np.std(temp)

        temp = (peer_placeholder[i][mask] - sub_placeholder[i][mask])
        line_data[12][i] = np.mean(temp)
        line_data[13][i] = np.std(temp)

        temp = (peer_placeholder[i][~mask] - sub_placeholder[i][~mask])
        line_data[14][i] = np.mean(temp)
        line_data[15][i] = np.std(temp)

        line_data[16][i] = np.mean(gini_pop[i])
        line_data[17][i] = np.std(gini_pop[i])

        line_data[18][i] = stats.shapiro(placeholder[i]).pvalue

        temp = settlements_pop[i]
        line_data[19][i] = np.mean(temp)
        line_data[20][i] = np.std(temp)

        line_data[21][i] = np.mean(auth_requests[i])
        line_data[22][i] = np.std(auth_requests[i])

        line_data[23][i] = np.mean(sub_requests[i])
        line_data[24][i] = np.std(sub_requests[i])

        line_data[25][i] = np.mean(peer_requests[i])
        line_data[26][i] = np.std(peer_requests[i])

        if i == 0:
            bar_data[0][0] = np.mean(peer_dst_placeholder[i], axis=0)
            bar_data[1][0] = np.mean(sub_dst_placeholder[i], axis=0)
        elif i == 999:
            bar_data[0][1] = np.mean(peer_dst_placeholder[i], axis=0)
            bar_data[1][1] = np.mean(sub_dst_placeholder[i], axis=0)
        elif i == 1999:
            bar_data[0][2] = np.mean(peer_dst_placeholder[i], axis=0)
            bar_data[1][2] = np.mean(sub_dst_placeholder[i], axis=0)

        if i == 1999:
            stats_data[0] = placeholder[i]
            stats_data[1] = peer_placeholder[i]
            stats_data[2] = sub_placeholder[i]

    return line_data, bar_data, stats_data


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
                p = ax.plot(iterations, data[agent_type][index[i]], label=data_types[agent_type][i])
                if show_std > -1:
                    ax.fill_between(iterations, data[agent_type][index[i]] - data[agent_type][index[i] + show_std],
                                    data[agent_type][index[i]] + data[agent_type][index[i] + show_std],
                                    color=p[0].get_color(), alpha=.1)
        else:
            p = ax.plot(iterations, data[agent_type][index[0]], label=agent_type)
            if show_std > -1:
                ax.fill_between(iterations, data[agent_type][index[0]] - data[agent_type][index[0] + show_std],
                                data[agent_type][index[0]] + data[agent_type][index[0] + show_std],
                                color=p[0].get_color(), alpha=.1)

    ax.legend(loc=legend)
    ax.set_aspect('auto')

    fig.savefig(filename)
    pyplot.close(fig)


def write_bar_plot(key, filename, data, title: str, x_axis: str, y_axis: str,
               legend: str = 'lower right'):

    divide_arr = ['beginning', 'middle', 'end']

    fig, ax = pyplot.subplots(1, 3)
    bins = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
    x_bins = np.arange(len(bins))
    for index in range(3):
        ax[index].set_title(title % (divide_arr[index], key))
        ax[index].set_xlabel(x_axis)
        ax[index].set_ylabel(y_axis)

        ax[index].bar(x_bins - 0.2, data[0][index], 0.4, label='Peer')
        ax[index].bar(x_bins + 0.2, data[1][index], 0.4, label="Sub")

        ax[index].set_xticks(x_bins, bins)
        ax[index].set_yticks(np.arange(0.0, 0.7, 0.1)) # setting the ticks

        ax[index].legend(loc=legend)
        ax[index].set_aspect('auto')

    fig.set_size_inches(18.5, 10.5)
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
    bar_data = {}
    stats_data = {}
    for root, _, files in os.walk(parser.path):
        for agent_type in files:
            with open(os.path.join(root, agent_type)) as json_file:
                key = agent_type[0:-5]
                data[key], bar_data[key], stats_data[key] = get_scenario_data(json.load(json_file))

    data_types = {}

    for a in data:
        data_types[a] = ['Peer', 'Sub']

    write_plot(['R-HIGH', 'F-HIGH', 'A-HIGH', 'S-HIGH'], '%s/population/HIGH_households' % parser.output, data,
               'Total Population of Initial Peer / Sub Distributions for \nHIGH Frequency Environmental Stress Scenarios.',
               [0], 'Iteration', 'Population')
    write_plot(['R-LOW', 'F-LOW', 'A-LOW', 'S-LOW'], '%s/population/LOW_households' % parser.output, data,
               'Total Population of Initial Peer / Sub Distributions for \nLOW Frequency Environmental Stress Scenarios.',
               [0], 'Iteration', 'Population')
    write_plot(['R-MED', 'F-MED', 'A-MED', 'S-MED'], '%s/population/MED_households' % parser.output, data,
               'Total Population of Initial Peer / Sub Distributions for \nMED Environmental Stress Scenarios.',
               [0], 'Iteration', 'Population')

    write_plot(['R-HIGH', 'F-HIGH', 'A-HIGH', 'S-HIGH'], '%s/population/HIGH_population' % parser.output, data,
               'Total Population of Initial Peer / Sub Distributions for \nHIGH Frequency Environmental Stress Scenarios.',
               [10], 'Iteration', 'Population')
    write_plot(['R-LOW', 'F-LOW', 'A-LOW', 'S-LOW'], '%s/population/LOW_population' % parser.output, data,
               'Total Population of Initial Peer / Sub Distributions for \nLOW Frequency Environmental Stress Scenarios.',
               [10], 'Iteration', 'Population')
    write_plot(['R-MED', 'F-MED', 'A-MED', 'S-MED'], '%s/population/MED_population' % parser.output, data,
               'Total Population of Initial Peer / Sub Distributions for \nMED Environmental Stress Scenarios.',
               [10], 'Iteration', 'Population')

    write_plot(['R-HIGH', 'F-HIGH', 'A-HIGH', 'S-HIGH'], '%s/gini/HIGH_gini' % parser.output, data,
               'Social Status Gini-Index for HIGH Frequency Environmental Stress Scenarios.',
               [16], 'Iteration', 'Gini')
    write_plot(['R-LOW', 'F-LOW', 'A-LOW', 'S-LOW'], '%s/gini/LOW_gini' % parser.output, data,
               'Social Status Gini-Index for LOW Frequency Environmental Stress Scenarios.',
               [16], 'Iteration', 'Gini')
    write_plot(['R-MED', 'F-MED', 'A-MED', 'S-MED'], '%s/gini/MED_gini' % parser.output, data,
               'Social Status Gini-Index for MED Frequency Environmental Stress Scenarios.',
               [16], 'Iteration', 'Gini')

    write_plot(['F-HIGH', 'F-MED', 'F-LOW'], '%s/settlements/%s_count' % (parser.output, 'F'), data,
               'Number of Settlements for %s Agent-Type' % 'F',
               [19], 'Iteration', 'Settlements', legend='center right')

    write_plot(['A-HIGH', 'A-MED', 'A-LOW'], '%s/settlements/%s_count' % (parser.output, 'A'), data,
               'Number of Settlements for %s Agent-Type' % 'A',
               [19], 'Iteration', 'Settlements', legend='center right')

    write_plot(['R-HIGH', 'R-MED', 'R-LOW'], '%s/settlements/%s_count' % (parser.output, 'R'), data,
               'Number of Settlements for %s Agent-Type' % 'R',
               [19], 'Iteration', 'Settlements', legend='center right')

    write_plot(['S-HIGH', 'S-MED', 'S-LOW'], '%s/settlements/%s_count' % (parser.output, 'S'), data,
               'Number of Settlements for %s Agent-Type' % 'S',
               [19], 'Iteration', 'Settlements', legend='center right')


    relative_scenario_keys =[
        ['A-HIGH', 'F-HIGH', 'R-HIGH', 'S-HIGH'],
        ['A-MED', 'F-MED', 'R-MED', 'S-MED'],
        ['A-LOW', 'F-LOW', 'R-LOW', 'S-LOW']
    ]

    for key in relative_scenario_keys:

        temp_data = np.zeros((4, 2000))

        for a_type in key:
            temp_data[0] += data[a_type][6]
            temp_data[1] += data[a_type][7]
            temp_data[2] += data[a_type][8]
            temp_data[3] += data[a_type][9]

        temp_data /= len(key)

        name = key[0][2:]

        temp_data = {name: temp_data}

        write_plot([name], '%s/transfer_difference/%s_transfer_difference' % (parser.output, name),
               temp_data,
               'Relative Change in Peer and Sub Transfer properties\nfor %s Frequency Scenarios' % name,
               [0, 2], 'Iteration', '%', data_types={name: ['Peer', 'Sub']},
               show_std=1, legend='lower left')


    as_combo = np.zeros((4, 2000))
    relative_scenario_keys =[
        ['A-LOW', 'A-MED'],
        ['S-LOW', 'S-MED']
    ]

    for a_type in relative_scenario_keys[0]:
        as_combo[0] += data[a_type][6]
        as_combo[1] += data[a_type][8]

    for a_type in relative_scenario_keys[1]:
        as_combo[2] += data[a_type][6]
        as_combo[3] += data[a_type][8]

    as_combo /= 2

    write_plot(['data'], '%s/%s_transfer_difference' % (parser.output, 'AS_EVO'),
               {'data': as_combo},
               'Relative Change in Peer and Sub Transfer Properties\n Averaged Across LOW and MED Scenarios',
               [0, 1, 2, 3], 'Iteration', '%', data_types={'data': ['A-Peer', 'A-Sub', 'S-Peer', 'S-Sub']},
               legend='lower left')


    relative_scenario_keys = [['A-HIGH', 'F-HIGH', 'R-HIGH', 'S-HIGH', 'A-MED', 'F-MED', 'R-MED', 'S-MED', 'A-LOW', 'F-LOW', 'R-LOW', 'S-LOW']]

    for key in relative_scenario_keys:

        temp_data = np.zeros((2, 2000))

        for a_type in key:
            temp_data[0] += data[a_type][23]
            temp_data[1] += data[a_type][25]

        name = key[0][2:]

        temp_data = {name: temp_data}

        write_plot([name], '%s/actions/avg_actions' % (parser.output), temp_data,
                   'Number of Resource Transfer Requests\nAcross All Scenarios',
                   [1,0], 'Iteration', 'Count', data_types={name: ['Peer', 'Sub']}, legend='upper right')

    for a in data:
        write_plot([a], '%s/transfer_chance/%s_transfer_chance' % (parser.output, a), data,
                   'Peer and Sub Transfer Properties for Scenario %s' % a,
                   [2, 4], 'Iteration', 'Probability (%)', data_types=data_types,
                   show_std=1)
        write_plot([a], '%s/transfer_difference/%s_transfer_difference' % (parser.output, a), data,
                   'Relative Change in Peer and Sub Transfer properties for Scenario %s' % a,
                   [6, 8], 'Iteration', '%', data_types=data_types,
                   show_std=1)

        write_plot([a], '%s/relative_prop_diff/%s_prop_difference' % (parser.output, a), data,
                   'Relative Difference in Peer and Sub Transfer properties for Scenario %s' % a,
                   [12, 14], 'Iteration', '%', data_types={a: ['Peer<Sub', 'Peer>Sub']},
                   show_std=1)

        write_plot([a], '%s/actions/%s_actions' % (parser.output, a), data,
                   'Number of Resource Transfer Actions\n averaged across all scenarios' % a,
                   [23,25], 'Iteration', 'Count', data_types={a: ['Sub', 'Peer']},
                   show_std=1)

        write_bar_plot(a, '%s/dist/%s_distribution' % (parser.output, a), bar_data[a],
                   'Distribution of Peer and Sub Transfer properties\n at the %s of Scenario %s', 'Bins', 'Distribution (%)')

        print('%s:\n0: %s +/- %s p:%s\t 5000: %s +/- %s p:%s\t10000: %s +/- %s p:%s' % (a, data[a][0][0], data[a][1][0], data[a][18][0],
                                                                           data[a][0][999], data[a][1][999], data[a][18][999],
                                                                           data[a][0][1999], data[a][1][1999], data[a][18][1999]))
        print('Ranksums:' + str(stats.ranksums(stats_data[a][1], stats_data[a][2], alternative='greater').pvalue))

    print("Kruskal Tests:")

    kruskal_keys = [
        ['R-HIGH', 'F-HIGH', 'A-HIGH', 'S-HIGH'],
        ['R-MED', 'F-MED', 'A-MED', 'S-MED'],
        ['R-LOW', 'F-LOW', 'A-LOW', 'S-LOW']
    ]

    for kkey in kruskal_keys:
        toTest = np.zeros((4, 50), dtype=float)

        toTest[0] = stats_data[kkey[0]][0]
        toTest[1] = stats_data[kkey[1]][0]
        toTest[2] = stats_data[kkey[2]][0]
        toTest[3] = stats_data[kkey[3]][0]

        print(kkey)
        print('Kruskal: ' + str(stats.kruskal(toTest[0], toTest[1], toTest[2], toTest[3]).pvalue))
        print(sp.posthoc_dunn([toTest[0], toTest[1], toTest[2], toTest[3]], p_adjust = 'bonferroni'))


if __name__ == '__main__':
    main()