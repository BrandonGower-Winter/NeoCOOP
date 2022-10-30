import argparse
import logging
import NeoCOOP
import optuna
import optuna.logging as olog
import random
import statistics

from Agents import AgentIEAdaptationSystem, HouseholdPreferenceComponent, IEComponent, ResourceComponent
from ECAgent.Decode import JsonDecoder
from VegetationModel import GlobalEnvironmentSystem


study_name = 'ABMHUB2022'
mysql_database_name = 'mysql://root:root@localhost/' + study_name

# This is just an estimate, does not need to be accurate but it makes understanding how much better the found solutions
# are compared to your best guesses
carry_cap = 8000
num_iterations = 10000
# Put the scenario frequencies you want to investigate here.
environment_frequencies = [10000, 5000, 2500, 1250, 625, 313]

def objective(trial):

    # Create Model
    model = JsonDecoder().decode(NeoCOOP.NeoCOOP.parser.file + '/decoder.json')

    # Randomize Frequency
    model.systemManager.systems['GES'].rainfall_dict['frequency'] = random.choice(environment_frequencies)

    # Parameters to optimize
    # Influence Frequency
    AgentIEAdaptationSystem.frequency = trial.suggest_int('influence_frequency', 5, 50)
    # Influence Rate
    AgentIEAdaptationSystem.influence_rate = trial.suggest_uniform('influence_rate', 0.01, 1.0)
    # Mutation Rate
    IEComponent.mutation_rate = trial.suggest_uniform('mutation_rate', 0.01, 0.25)
    # Learning and Conformity rate range
    lower_bound = trial.suggest_uniform('learning_rate_lower', 0.0, 0.99)
    upper_bound = trial.suggest_uniform('learning_rate_upper', lower_bound, 1.0)

    HouseholdPreferenceComponent.learning_rate_range = [lower_bound, upper_bound]
    IEComponent.conformity_range = [lower_bound, upper_bound]

    # Reassign agent properties to fit within explorable bounds
    for agent in model.environment.getAgents():
        agent[HouseholdPreferenceComponent].learning_rate = random.uniform(lower_bound, upper_bound)
        agent[IEComponent].conformity = random.uniform(lower_bound, upper_bound)

    # Run Simulation
    for _ in range(model.iterations):
        model.systemManager.executeSystems()
        if len(model.environment.agents) == 0:
            return 0.0 , 0.0

    # Return Final Score
    pop = sum([len(agent[ResourceComponent].occupants) for agent in model.environment.getAgents()]) / carry_cap
    res = statistics.mean([agent[ResourceComponent].resources for agent in model.environment.getAgents()])
    return pop, res

def main():

    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='Path to decoder json file.', default=None)
    parser.add_argument('-n', '--ntrials', help='Number of trials to run.', default=20, type=int)
    parser.add_argument('-t', '--timeout', help='Timeout of trials.', default=None, type=int)
    parser.add_argument('-v', '--visualize', help='Will visualize the current best found hyper-parameters.',
                        action='store_true')
    parser.add_argument('-p', '--present',
                        help='Will print out a report of the best model hyper-parameters found thus far.',
                        action='store_true')

    parser = parser.parse_args()

    study = optuna.create_study(study_name=study_name, storage=mysql_database_name, directions=['maximize', 'maximize'],
                                load_if_exists = True)

    fh_logger = logging.getLogger('model')
    fh_logger.propagate = False  # This is needed to prevent the terminal from being spammed with NeoCOOP logged events.
    fh_logger.setLevel(logging.WARNING)

    if parser.file is not None:
        # Transform environment frequencies into necessary format
        for i in range(len(environment_frequencies)):
            environment_frequencies[i] = GlobalEnvironmentSystem.convert_to_freq(environment_frequencies[i], num_iterations)
        NeoCOOP.NeoCOOP.parser = parser
        olog.set_verbosity(olog.WARNING)
        study.optimize(objective, n_trials = parser.ntrials, timeout = parser.timeout)

    if parser.present:
        params = {}
        str_trials = []

        for t in study.best_trials:
            str_trials.append(str(t.params))
            for p in t.params:
                if p not in params:
                    params[p] = t.params[p]
                else:
                    params[p] += t.params[p]

        for p in params:
            params[p] = params[p]/len(study.best_trials)

        bst_trials = "\n".join(str_trials)
        print("Best Model Parameters found thus far:\n{}".format(bst_trials))
        print("Average Results:\n{}".format(str(params)))

    return

if __name__ == '__main__':
    main()