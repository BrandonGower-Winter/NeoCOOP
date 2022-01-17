import argparse
import gc
import json
import os
import statistics

import numpy as np

from math import ceil
from Progress import progress


def gini(x):

    # Mean Absolute Difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative Mean Absolute difference
    rmad = mad / np.mean(x)

    return 0.5 * rmad


def get_json_iteration(filename: str) -> int:
    return int(filename[filename.index('_')+1:-5])


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

    prop_dict = {}

    over_range = over_range if over_range[1] != -1 else (over_range[0], len(snapshots))

    for i in range(over_range[0], over_range[1]):
        for func in comp_funcs:

            val = generate_composite_val(props, snapshots[i], func[1], sort)

            if func[0] in prop_dict:
                prop_dict[func[0]].append(val)
            else:
                prop_dict[func[0]] = [val]

    return prop_dict

def bin01(data: []):

    counts = [0 for _ in range(10)]

    for val in data:
        index = int(max(min(ceil(val * 10) - 1, 9), 0))
        counts[index] += 1

    return [p / float(len(data)) for p in counts]



def main():

    # Process the params
    print("Parsing arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--paths', help='The path(s) to the folder(s) containing all of the generated data',
                        nargs='+', required=True)
    parser.add_argument('-o', '--output', help='The path to write all of the processed data to.',
                        required=True)
    parser.add_argument('-v', '--verbose', help='Will print out informative information to the terminal.',
                    action='store_true')

    parser = parser.parse_args()

    for path in parser.paths:
        if not os.path.isdir(path):
            print('Please make sure the path(s) specified are directories...')
            return

    print("Loading all json files...")

    # For all folders in path

    # File Format: agent_type/scenario/seed/resources+population
    for path in parser.paths:
        scenarios = [s for s in os.listdir(path) if os.path.isdir(os.path.join(path, s))]
        print('\t- Found %s scenarios...' % len(scenarios))
        for scenario in scenarios:
            to_write = {}
            print('\t- Scenario: %s' % scenario)
            scenario_path = os.path.join(path, scenario)

            runs = [r for r in os.listdir(scenario_path) if os.path.isdir(os.path.join(scenario_path, r))]
            run_len = len(runs)
            print('\t\t- Found %s Simulation Runs...' % run_len)
            for i in range(run_len):
                progress(i, run_len)
                to_write[runs[i]] = {}
                # Get all agent json files in this simulation run
                agent_snapshots = load_json_files(str(scenario_path) + '/' + runs[i] + '/agents')

                to_write[runs[i]]['resources'] = get_composite_property_as_dict(agent_snapshots, ['wealth'],
                                                                 [('mean', statistics.mean),
                                                                  ('total', sum),
                                                                  ('gini', gini)])

                to_write[runs[i]]['population'] = get_composite_property_as_dict(agent_snapshots, ['population'],
                                               [('number', len),
                                                ('total', sum)])

                to_write[runs[i]]['peer_transfer'] = get_composite_property_as_dict(agent_snapshots, ['peer_transfer'],
                                                                                 [('mean', statistics.mean),
                                                                                  ('dist', bin01)])

                to_write[runs[i]]['sub_transfer'] = get_composite_property_as_dict(agent_snapshots, ['sub_transfer'],
                                                                                    [('mean', statistics.mean),
                                                                                     ('dist', bin01)])
                gc.collect()
            print()
            print('Writing data to output file:' + parser.output + '/processed_agents_' + scenario + '.json:')
            with open(parser.output + '/processed_agents_' + scenario + '.json', 'w') as outfile:
                json.dump(to_write, outfile, indent=4)


if __name__ == '__main__':
    main()
