import argparse

import Optuna.EgyptIEOptuna as IEOptuna
import Optuna.EgyptUtilityOptuna as UtilityOptuna
import Optuna.EgyptTraditional as TraditionalOptuna

mysql_database_name = 'mysql://root:root@localhost/'

def main():
    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--study', help='Name of the study.', default='EgyptTrad')
    parser.add_argument('-f', '--file', help='Path to decoder json file.', default=None)
    parser.add_argument('-n', '--ntrials', help='Number of trials to run.', default=20, type=int)
    parser.add_argument('-t', '--timeout', help='Timeout of trials.', default=None, type=int)
    parser.add_argument('-v', '--visualize', help='Will visualize the current best found hyper-parameters.',
                        action='store_true')
    parser.add_argument('-p', '--present',
                        help='Will print out a report of the best model hyper-parameters found thus far.',
                        action='store_true')

    parser = parser.parse_args()

    if parser.study == 'EgyptIE':
        IEOptuna.run_trials('EgyptIE', mysql_database_name + 'EgyptIE', parser)
    elif parser.study == 'EgyptUtility':
        UtilityOptuna.run_trials('EgyptUtility', mysql_database_name + 'EgyptUtility', parser)
    elif parser.study == 'EgyptTrad':
        TraditionalOptuna.run_trials('EgyptTrad', mysql_database_name + 'EgyptTrad', parser)

if __name__ == '__main__':
    main()