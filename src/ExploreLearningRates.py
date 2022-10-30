import argparse
import random
import statistics
import NeoCOOP
import logging

import numpy as np

from Agents import AgentIEAdaptationSystem, HouseholdPreferenceComponent, IEComponent, ResourceComponent
from ECAgent.Decode import JsonDecoder
from multiprocessing import Process

REALIZATIONS = 1
ITERATIONS = 2500
SEED = 1361
LRANGE = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
CRANGE = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

def gini(x):

    # Mean Absolute Difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative Mean Absolute difference
    rmad = mad / np.mean(x)

    return 0.5 * rmad

def run_models(learning_rate, conformity, seed):

    NeoCOOP.NeoCOOP.seed = seed

    models = [JsonDecoder().decode(NeoCOOP.NeoCOOP.parser.file + '/decoder.json') for _ in range(REALIZATIONS)]

    # Reassign agent properties to fit within explorable bounds
    for model in models:
        for agent in model.environment.getAgents():
            agent[HouseholdPreferenceComponent].learning_rate = learning_rate
            agent[IEComponent].conformity = conformity

    # Run Simulations
    for _ in range(ITERATIONS):
        for model in models:
            model.systemManager.executeSystems()

    pop = []
    res = []
    inequality = []

    # Get final population and resource levels
    for model in models:
        pop.append(sum([len(agent[ResourceComponent].occupants) for agent in model.environment.getAgents()]))
        resource_arr = [agent[ResourceComponent].resources for agent in model.environment.getAgents()]
        res.append(statistics.mean(resource_arr))
        inequality.append(gini(np.array(resource_arr)))

    print('%s, %s: %s, %s, %s' % (learning_rate, conformity, statistics.mean(pop), statistics.mean(res),
                                  statistics.mean(inequality)))

def main():
    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='Path to decoder json file.', default=None)

    parser = parser.parse_args()
    NeoCOOP.NeoCOOP.parser = parser

    fh_logger = logging.getLogger('model')
    fh_logger.propagate = False  # This is needed to prevent the terminal from being spammed with NeoCOOP logged events.
    fh_logger.setLevel(logging.WARNING)

    seed = random.randint(0, 10000000)

    # Loop Over Conformity
    for conformity in CRANGE:
        processes = []
        # Loop Over Learning Rate
        for learning_rate in LRANGE:
            p = Process(target=run_models, args=(learning_rate, conformity, seed,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


if __name__ == '__main__':
    main()