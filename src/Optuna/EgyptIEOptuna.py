import NeoCOOP
import logging
import optuna
import optuna.logging as olog
import random
import statistics

from Agents import AgentIEAdaptationSystem, HouseholdPreferenceComponent, IEComponent, ResourceComponent
from ECAgent.Decode import JsonDecoder

def objective(trial):

    # Create Model
    model = JsonDecoder().decode(NeoCOOP.NeoCOOP.parser.file + '/decoder.json')

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
    pop = sum([len(agent[ResourceComponent].occupants) for agent in model.environment.getAgents()])
    res = statistics.mean([agent[ResourceComponent].resources for agent in model.environment.getAgents()])
    return pop, res

def run_trials(study_name : str, mysql_database_name : str, parser):

    study = optuna.create_study(study_name=study_name, storage=mysql_database_name, directions=['maximize', 'maximize'],
                                load_if_exists = True)

    fh_logger = logging.getLogger('model')
    fh_logger.propagate = False  # This is needed to prevent the terminal from being spammed with NeoCOOP logged events.
    fh_logger.setLevel(logging.WARNING)

    if parser.file is not None:
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