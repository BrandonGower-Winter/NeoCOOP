import NeoCOOP
import logging
import optuna
import optuna.logging as olog
import random
import statistics

from Agents import AgentResourceAcquisitionSystem, ResourceComponent
from ECAgent.Decode import JsonDecoder

def objective(trial):

    # Create Model
    model = JsonDecoder().decode(NeoCOOP.NeoCOOP.parser.file + '/decoder.json')

    # Learning and Conformity rate range
    AgentResourceAcquisitionSystem.forage_grad = trial.suggest_uniform('forage_grad', 0.0, 1.0)
    AgentResourceAcquisitionSystem.forage_margin = trial.suggest_uniform('forage_offset', 0.0, 0.5)

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