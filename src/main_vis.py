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
from CythonFunctions import CAgentUtilityFunctions


def get_json_iteration(filename: str) -> int:
    return int(filename[filename.index('_')+1:-5])


def get_csv_iteration(filename: str) -> int:
    return int(filename[filename.index('_')+1:-4])


def load_json_file(filename: str):
    with open(filename) as json_file:
        return json.load(json_file)


def load_json_files(folder_path: str, sort: bool = True, key=get_json_iteration) -> []:

    json_snapshots = []

    for root, _, files in os.walk(folder_path, topdown=True):

        json_files = [f for f in files if f[-4:] == 'json']
        if sort:
            json_files.sort(key=key)

        for file in json_files:
            with open(os.path.join(root, file)) as json_file:
                json_snapshots.append(json.load(json_file))

    return json_snapshots


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


def reformat_snapshots_to_per_entity_dicts(snapshots: [[dict]], id_str: str = 'id', filter: [str] = None):
    agent_data = {}

    if filter is None:
        filter = get_all_keys(snapshots[0][0])
        filter.remove(id_str)

    for i in range(len(snapshots)):

        for agent in snapshots[i]:
            a_id = agent[id_str]
            if a_id in agent_data:
                for prop in filter:
                    agent_data[a_id][prop].append(agent[prop])
                agent_data[a_id]['iterations'].append(i)
            else:
                agent_data[a_id] = {}
                for prop in filter:
                    agent_data[a_id][prop] = [agent[prop]]
                agent_data[a_id]['iterations'] = [i]

    return agent_data


def generate_composite_val(props: [str], snapshot: dict, comp_func, sort: bool = False):

    if len(props) > 1:
        ls = [[agent[prop] for prop in props] for agent in snapshot]
    else:
        ls = [agent[props[0]] for agent in snapshot]

    if len(ls) == 0:
        return 0

    if sort:
        ls.sort()

    return comp_func(ls)


def get_composite_property_as_dict(snapshots: [[dict]], props: [str], comp_funcs: [(str, any)],
                                   over_range: (int, int) = (0, -1), sort: bool = False) -> dict:

    prop_dict = {'iterations': []}

    over_range = over_range if over_range[1] != -1 else (over_range[0], len(snapshots))

    for i in range(over_range[0], over_range[1]):
        for func in comp_funcs:

            val = generate_composite_val(props, snapshots[i], func[1], sort)

            if func[0] in prop_dict:
                prop_dict[func[0]].append(val)
            else:
                prop_dict[func[0]] = [val]

        prop_dict['iterations'].append(i)

    return prop_dict


def create_composite_property_as_panda(snapshots: [pandas.DataFrame], func, kwargs_to_pass: dict = {}):

    for snapshot in snapshots:
        func(snapshot, **kwargs_to_pass)


def generate_plot_from_dict(title: str, data: dict, filename: str, filter: [str] = None, y_label: str = '',
                            x_label: str = 'Iterations', legend: str = None) -> None:

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if filter is None:
        filter = get_all_keys(data)
        filter.remove('iterations')

    for prop in filter:
        ax.plot(data['iterations'], data[prop], label=prop)

    if legend is not None:
        ax.legend(loc=legend)

    ax.set_aspect('auto')

    fig.savefig(filename)


def generate_plot_from_entity_dicts(title: str, data: dict, property: str, filename: str,  y_label: str = '',
                            x_label: str = 'Iterations', legend: str = None):

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    for e_id in data:
        ax.plot(data[e_id]['iterations'], data[e_id][property], label=e_id)

    if legend is not None:
        ax.legend(loc=legend)

    ax.set_aspect('auto')

    fig.savefig(filename)


def pandas_to_plot(title: str, width: int, height: int, data: [pandas.DataFrame], property:str, filename:str,
                   slices: [int] = None, x_label: str = 'X', y_label: str = 'Y', vmin: int = 0, vmax: int = 100000) -> None:

    # Generate the pix map
    for i in slices:
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plot = ax.imshow(data[i][property].to_numpy().reshape(height, width), interpolation='none', cmap='jet', vmin=vmin, vmax=vmax)
        fig.colorbar(plot)
        fig.savefig(filename + '_{}.png'.format(i))


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


def generate_plot_from_log_list(title: str, data: [dict], filename: str, filter: [str], y_label: str = '', default_val = 0,
                            x_label: str = 'Iterations', legend: str = None) -> None:

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plot_content = {}

    for property in filter:
        plot_content[property] = []

    for i in range(len(data)):
        for property in filter:
            if property in data[i]:
                plot_content[property].append(data[i][property])
            else:
                plot_content[property].append(default_val)

    for property in filter:
        ax.plot(np.arange(len(data)), plot_content[property], label=property)

    if legend is not None:
        ax.legend(loc=legend)

    ax.set_aspect('auto')

    fig.savefig(filename)


def gini(x):

    # Mean Absolute Difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative Mean Absolute difference
    rmad = mad / np.mean(x)

    return 0.5 * rmad


def land_possesion(df : pandas.DataFrame, **kwargs):

    land = []

    for i in range(len(df)):
        if df['isSettlement'][i] != -1:
            land.append(2)
        elif df['isOwned'][i] != -1:
            land.append(1)
        else:
            land.append(0)

    df['land_ownership'] = land


def xtent_map(settlement_data : [], pixels):

    ret_data = []
    pos_data = []
    # Generating positions:
    for y in range(parser.height):
        for x in range(parser.width):
            pos_data.append((x, y))

    count = 1
    for it_set in settlement_data:
        print('Iteration: {}'.format(count))
        it_data = []
        for y in range(parser.height):
            it_row = []
            for x in range(parser.width):

                ws = np.zeros((len(it_set)))
                ds = np.zeros((len(it_set)))

                for i in range(len(it_set)):

                    ws[i] = it_set[i]['wealth'] + it_set[i]['load']
                    n_x = pos_data[
                        it_set[i]['pos'][0]
                    ][0]
                    n_y = pos_data[it_set[i]['pos'][0]][1]
                    ds[i] = math.sqrt(((x - n_x) ** 2) + ((y - n_y) ** 2))

                ds = ds * 2000  # Cell Size
                dst = CAgentUtilityFunctions.xtent_distribution(ws, ds, 0.75, 1.5)

                iSettlement = np.argmax(dst)

                it_row.append(it_set[iSettlement]['id']+1 if dst[iSettlement] > 0.0 else 0)

            it_data.append(it_row)

        ret_data.append(it_data)
        count += 1

    return ret_data


def get_dict_by_id(l : [], id: int):
    for item in l:
        if item['id'] == id:
            return item
    return None


def xtent_to_property(xtent_arr: [], settlement_data: [], property: str, modifier = 0.0):
    to_ret = []

    for z in range(len(xtent_arr)):
        map = []
        for y in range(len(xtent_arr[z])):
            row = []
            for x in range(len(xtent_arr[z][y])):
                if xtent_arr[z][y][x] != 0:
                    settlement = get_dict_by_id(settlement_data[z], int(xtent_arr[z][y][x] - 1))
                    if settlement is not None and 'belief_space' in settlement:
                        row.append(settlement['belief_space'][property] + modifier)
                    else:
                        row.append(0.0)
                else:
                    row.append(0.0)
            map.append(row)
        to_ret.append(map)

    return to_ret


def household_social_status_weighted_mean(data: []):

    total_social_status = sum([point[1] + point[2] for point in data])
    return sum([point[0] * (point[1] + point[2]) / total_social_status for point in data])


def bin01(data: []):

    counts = [0 for _ in range(10)]

    for val in data:
        index = int(max(min(math.ceil(val * 10) - 1, 9), 0))
        counts[index] += 1

    return [p / float(len(data)) for p in counts]


def generate_household_plots(parser):
    agent_snapshots = load_json_files(parser.path + '/agents')

    if not os.path.isdir(parser.path + '/agent_plots'):
        os.mkdir(parser.path + '/agent_plots')

    population_dict = get_composite_property_as_dict(agent_snapshots, ['resources'],
                                                     [('mean', statistics.mean),
                                                      ('median', statistics.median),
                                                      ('min', min),
                                                      ('max', max),
                                                      ('total', sum),
                                                      ('gini', gini)], sort=True)

    household = get_composite_property_as_dict(agent_snapshots, ['occupants'],
                                                     [('mean', statistics.mean),
                                                      ('median', statistics.median),
                                                      ('min', min),
                                                      ('max', max),
                                                      ('total', sum)], sort=True)

    generate_plot_from_dict('Summary of Household Resources over 1000 years', population_dict,
                            parser.path + '/agent_plots/resource_summary.png',
                            filter=['mean', 'median', 'min', 'max'],
                            y_label='Resources (KG)', legend='center left')

    generate_plot_from_dict('Total Household Resources over 1000 years', population_dict,
                            parser.path + '/agent_plots/resource_total.png',
                            filter=['total'],
                            y_label='Resources (KG)', legend='center right')

    generate_plot_from_dict('Summary of Household Population over 1000 years', household,
                            parser.path + '/agent_plots/population_summary.png',
                            filter=['mean', 'median', 'min', 'max'],
                            y_label='Population', legend='center left')

    generate_plot_from_dict('Total Household Population', household,
                            parser.path + '/agent_plots/occupants_total.png',
                            filter=['total'],
                            y_label='Population', legend='center right')

    generate_plot_from_dict('Gini Coeffecient for Households over 1000 years', population_dict,
                            parser.path + '/agent_plots/resources_gini.png',
                            filter=['gini'], legend='center right')

    transfer_dict = {}
    transfer_dict['Peer Transfer'] = get_composite_property_as_dict(agent_snapshots, ['peer_chance'],
                                     [('mean', statistics.mean)], sort=True)['mean']
    transfer_dict['Subordinate Transfer'] = get_composite_property_as_dict(agent_snapshots, ['sub_chance'],
                                     [('mean', statistics.mean)], sort=True)['mean']
    transfer_dict['iterations'] = np.arange(len(transfer_dict['Peer Transfer']))

    generate_plot_from_dict('Average Transfer Percentage of Agents over 2000 iterations', transfer_dict,
                            parser.path + '/agent_plots/transfer_chance.png',
                            y_label='Probability', legend='center left')

    bins = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']

    precords = get_composite_property_as_dict(agent_snapshots, ['peer_chance'],
                                             [('bin', bin01)])['bin']

    srecords = get_composite_property_as_dict(agent_snapshots, ['sub_chance'],
                                             [('bin', bin01)])['bin']

    records = [[precords[i], srecords[i]] for i in range(len(precords))]

    Animate.generateBarAnimat("Animation of Distribution of the Agent Resource Transfer Probabilities", records, bins,
                              50, parser.path + "/agent_plots/peer_transfer", 'Transfer Chance', '%',
                              colors=['r', 'b'], labels=['Peer Chance', 'Sub Chance'])

    percentage_to_farm = get_composite_property_as_dict(agent_snapshots, ['percentage_to_farm'],
                                                        [('mean', statistics.mean)], sort=True)

    generate_plot_from_dict('Average Percentage to Farm over 2000 Iterations', percentage_to_farm,
                            parser.path + '/agent_plots/percentage_to_farm.png',
                            filter=['mean'],
                            y_label='%', legend='center right')


def generate_settlement_plots(parser, pixels):
    # Settlement Plots
    if not os.path.isdir(parser.path + '/settlement_plots'):
        os.mkdir(parser.path + '/settlement_plots')

    settlement_snapshots = load_json_files(parser.path + '/settlements')

    settlement_dict = get_composite_property_as_dict(settlement_snapshots, ['wealth'],
                                                     [('mean', statistics.mean),
                                                      ('median', statistics.median),
                                                      ('min', min),
                                                      ('max', max),
                                                      ('total', sum),
                                                      ('gini', gini)], sort=True)

    generate_plot_from_dict('Summary of Settlement Resources over 1000 years', settlement_dict,
                            parser.path + '/settlement_plots/resource_summary.png',
                            filter=['mean', 'median', 'min', 'max'],
                            y_label='Resources (KG)', legend='center left')

    generate_plot_from_dict('Total Settlement Resources over 1000 years', settlement_dict,
                            parser.path + '/settlement_plots/resource_total.png',
                            filter=['total'],
                            y_label='Resources (KG)', legend='center right')

    generate_plot_from_dict('Gini Coeffecient for Settlements over 1000 years', settlement_dict,
                            parser.path + '/settlement_plots/resources_gini.png',
                            filter=['gini'], legend='center right')

    xtent_arr = xtent_map(settlement_snapshots, pixels)

    farm_utility_arr = xtent_to_property(xtent_arr, settlement_snapshots, 'farm_utility')
    forage_utility_arr = xtent_to_property(xtent_arr, settlement_snapshots, 'forage_utility')

    for z in range(len(farm_utility_arr)):
        for y in range(len(farm_utility_arr[z])):
            for x in range(len(farm_utility_arr[z][y])):
                if farm_utility_arr[z][y][x] == 0.0 and forage_utility_arr[z][y][x] == 0.0:
                    farm_utility_arr[z][y][x] = 0.0
                else:
                    farm_utility_arr[z][y][x] = 2.0 if farm_utility_arr[z][y][x] > forage_utility_arr[z][y][x] else 1.0

    learning_rate_arr = xtent_to_property(xtent_arr, settlement_snapshots, 'learning_rate', 1)
    conformity_arr = xtent_to_property(xtent_arr, settlement_snapshots, 'conformity', 1)
    peer_arr = xtent_to_property(xtent_arr, settlement_snapshots, 'peer_transfer', 1)
    sub_arr = xtent_to_property(xtent_arr, settlement_snapshots, 'sub_transfer', 1)

    Animate.generateAnimat('Xtent model showing Settlement Territory', xtent_arr, fps=100, vmin=0, vmax=300,
                           filename=parser.path + '/settlement_plots/xtent_animat')

    Animate.generateAnimat('Influence of Settlement Farm/Forage Preference', farm_utility_arr, fps=100, vmin=0, vmax=2,
                           filename=parser.path + '/settlement_plots/farm_utility_influence_animat')

    Animate.generateAnimat('Influence of Settlement Learning Rate', learning_rate_arr, fps=100, vmin=0, vmax=1.2,
                           filename=parser.path + '/settlement_plots/learning_rate_influence_animat')

    Animate.generateAnimat('Influence of Settlement Conformity', conformity_arr, fps=100, vmin=0, vmax=1.2,
                           filename=parser.path + '/settlement_plots/conformity_influence_animat')

    Animate.generateAnimat('Influence of Settlement Peer Exchange', peer_arr, fps=100, vmin=0, vmax=2,
                           filename=parser.path + '/settlement_plots/peer_influence_animat')

    Animate.generateAnimat('Influence of Settlement Sub Exchange', sub_arr, fps=100, vmin=0, vmax=2,
                           filename=parser.path + '/settlement_plots/sub_influence_animat')


def generate_environment_plots(parser, pixels):

    if not os.path.isdir(parser.path + '/environment_plots'):
        os.mkdir(parser.path + '/environment_plots')

    environment_snapshots = load_csvs(parser.path + '/environment')

    create_composite_property_as_panda(environment_snapshots, land_possesion, {'pixels': pixels})
    pandas_to_animat('NeoCOOP Visual Representation', parser.width, parser.height, environment_snapshots, 'land_ownership',
                     parser.path + '/environment_plots/land_ownership_animat', 100, vmin=0, vmax=2)
    pandas_to_animat('Animation of `Vegetation` over 1000 years', parser.width, parser.height, environment_snapshots,
                     'vegetation',
                     parser.path + '/environment_plots/vegetation_animat', 10, vmin=0, vmax=10500)
    pandas_to_animat('Animation of `Soil Moisture` over 1000 years', parser.width, parser.height, environment_snapshots,
                     'moisture',
                     parser.path + '/environment_plots/moisture_animat', 10, vmin=0, vmax=700)


def generate_log_plots(parser):

    if not os.path.isdir(parser.path + '/log_plots'):
        os.mkdir(parser.path + '/log_plots')

    log_list = log_file_to_list(parser.path + '/events.log')

    generate_plot_from_log_list('Household Farm and Forage actions over 2000 iterations', log_list,
                                parser.path + '/log_plots/FarmForage.png', ['HOUSEHOLD.FARM', 'HOUSEHOLD.FORAGE'],
                                y_label='Number of actions', default_val=0, legend='center left')
    generate_plot_from_log_list("House Resource Transfer Actions over 2000 iterations", log_list,
                                parser.path + '/log_plots/ResourceTransfer.png',
                                ['HOUSEHOLD.RESOURCES.TRANSFER.SUCCESS.PEER',
                                 'HOUSEHOLD.RESOURCES.TRANSFER.SUCCESS.AUTH',
                                 'HOUSEHOLD.RESOURCES.TRANSFER.SUCCESS.SUB',
                                 'HOUSEHOLD.RESOURCES.TRANSFER.FAIL.PEER',
                                 'HOUSEHOLD.RESOURCES.TRANSFER.FAIL.AUTH',
                                 'HOUSEHOLD.RESOURCES.TRANSFER.FAIL.SUB',
                                 'HOUSEHOLD.RESOURCES.TRANSFER.REJECT.PEER',
                                 'HOUSEHOLD.RESOURCES.TRANSFER.REJECT.AUTH'],
                                y_label='Number of actions', default_val=0, legend='center left')


def dynamic_farm_animat(parser, pixels):

    environment_snapshots = load_csvs(parser.path + '/environment')
    log_list = log_file_to_list(parser.path + '/events.log')

    count = 0

    def dynamic_farm(df, **kwargs):
        land = []
        nonlocal count

        for i in range(len(df)):
            if df['isSettlement'][i] != -1:
                land.append(2)
            elif 'FARM_LOC' in log_list[count] and i in log_list[count]['FARM_LOC']:
                land.append(3)
            else:
                land.append(kwargs['pixels'][i])

        df['dynamic_ownership'] = land
        count += 1

    create_composite_property_as_panda(environment_snapshots, dynamic_farm, {'pixels': pixels})

    pandas_to_animat('Settlement and Farm Locations on Map', parser.width, parser.height, environment_snapshots,
                     'dynamic_ownership',
                     parser.path + '/environment_plots/dynamic_farm_animat', 10, vmin=0, vmax=3)


def other_stuff():
    tradagent_snapshots = load_json_files('trad_sc2/agents')
    utilagent_snapshots = load_json_files('utility_sc2/agents')
    adaptive_snapshots = load_json_files('adaptive_sc2/agents')

    plot_dict = {}
    plot_dict['Traditional'] = get_composite_property_as_dict(tradagent_snapshots, 'occupants',
                                               [('total', sum)], sort=True)['total']

    plot_dict['iterations'] = np.arange(len(plot_dict['Traditional']))

    plot_dict['Utility'] = get_composite_property_as_dict(utilagent_snapshots, 'occupants',
                                               [('total', sum)], sort=True)['total']

    plot_dict['Adaptive'] = get_composite_property_as_dict(adaptive_snapshots, 'occupants',
                                               [('total', sum)], sort=True)['total']

    generate_plot_from_dict('Total Household Population', plot_dict,
                            'pop_comparison_all.png',
                            filter=['Traditional', 'Utility', 'Adaptive'],
                            y_label='Population', legend='center right')


if __name__ == '__main__':
    import argparse
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='The path to the folder containing all of the generated data', type=str)
    parser.add_argument('width', help='The width of the map.', type=int)
    parser.add_argument('height', help='The height of the map.', type=int)
    parser.add_argument('-v', '--verbose', help='Will print out informative information to the terminal.',
                        action='store_true')

    parser = parser.parse_args()

    pixels = []

    for y in range(parser.height):
        for x in range(parser.width):
            pixels.append(0)

    generate_settlement_plots(parser, pixels)
    generate_log_plots(parser)
    generate_environment_plots(parser, pixels)
    generate_household_plots(parser)

    #other_stuff()
    #dynamic_farm_animat(parser, pixels)
