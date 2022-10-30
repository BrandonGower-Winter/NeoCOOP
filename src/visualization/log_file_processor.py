import argparse
import gc
import json
import os
import statistics

import numpy as np

from math import ceil, isinf, isnan

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
            elif keyword in iter_dict:
                iter_dict[keyword] += 1
            else:
                iter_dict[keyword] = 1

    return log_list

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

    # File Format: agent_type/scenario/seed/events.log
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
                gc.collect()
                print(i)
                to_write[runs[i]] = log_file_to_list(str(scenario_path) + '/' + runs[i] + '/events.log')

            print()

            print('Writing data to output file:' + parser.output + '/processed_logs_' + scenario + '.json:')
            with open(parser.output + '/processed_logs_' + scenario + '.json', 'w') as outfile:
                json.dump(to_write, outfile, indent=4)


if __name__ == '__main__':
    main()