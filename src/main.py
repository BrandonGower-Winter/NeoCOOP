import argparse
import logging
import multiprocessing
import NeoCOOP
import numpy as np
import os
import SimpleVegetationModel
import time

from Agents import *
from datetime import datetime
from ECAgent.Decode import JsonDecoder
import ECAgent.Environments as env
from VegetationModel import *

from Progress import progress

# Default Decoder file path
default_path = './resources/traditional/scenario3/'
store_path = 'base'


def parseArgs():
    """Create the EgyptModel Parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='Path to decoder json file.', default=default_path)
    parser.add_argument('-v', '--visualize', help='Will start a dash applet to display a summary of the simulation.',
                        action='store_true')
    parser.add_argument('-d', '--debug', help='Sets the model to debug mode. Output printed to terminal will be verbose',
                        action='store_true')
    parser.add_argument('-p', '--profile',
                        help='Sets the model to profile mode. Will write all data to file called profile.html',
                        action='store_true')
    parser.add_argument('-l', '--log', help='Tell the application to generate a log file for this run of the simulation.',
                        action='store_true')
    parser.add_argument('-r', '--record', help='Tell the application to record all of the model data to a vegetation and agent csv file',
                        action='store_true')
    parser.add_argument('-b', '--base', help='Base name of all the generated files.', default=store_path)
    parser.add_argument('--frequency', help='Frequency at which to capture snapshots.', default=1, type=int)
    parser.add_argument('--thread', help='Tell the application to multi-thread your application. Still WIP.',
                        action='store_true')
    parser.add_argument('--seed', help="Specify the seed for the Model's pseudorandom number generator", default=None,
                        type=int)

    return parser.parse_args()


def addDebugHandler(isDebug: bool):
    root = logging.getLogger('')
    root.setLevel(logging.DEBUG if isDebug else logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if isDebug else logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    root.addHandler(console)


def addFileHandler(file_path, log_data: bool):
    fh_logger = logging.getLogger('model')
    fh_logger.propagate = False
    fh_logger.setLevel(logging.INFO)

    if log_data:
        fh = logging.FileHandler(filename=file_path, mode='w')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)

        fh_logger.addHandler(fh)


def main():
    parser = parseArgs()
    start_time = time.time()

    addDebugHandler(parser.debug)

    logging.info('Simulation run invoked at: {}'.format(datetime.now()))

    if parser.record or parser.log:
        os.mkdir(parser.base)

    addFileHandler(parser.base + '/events.log', parser.log)

    logging.info("Creating Model...")
    NeoCOOP.NeoCOOP.seed = parser.seed
    NeoCOOP.NeoCOOP.parser = parser
    model = JsonDecoder().decode(parser.file + '/decoder.json')

    if parser.record:
        #logging.info('\t-Adding Environment Data Collector.')
        #os.mkdir(parser.base + '/environment')
        #model.systemManager.addSystem(
            #VegetationSnapshotCollector('VSC', model, parser.base + '/environment', parser.frequency))
        if len(model.environment.agents) > 0:
            logging.info('\t-Adding Agent Data Collector.')
            os.mkdir(parser.base + '/agents')
            model.systemManager.addSystem(AgentSnapshotCollector('ASC', model, parser.base + '/agents', parser.frequency))
            #logging.info('\t-Adding Settlement Data Collector.')
            #os.mkdir(parser.base + '/settlements')
            #model.systemManager.addSystem(
                #SettlementSnapshotCollector('SSC', model, parser.base + '/settlements', parser.frequency))

    logging.info('...Done!')

    model.debug = parser.debug
    NeoCOOP.pool_count = multiprocessing.cpu_count() if parser.thread else 0
    NeoCOOP.pool = multiprocessing.Pool(processes=NeoCOOP.pool_count) if NeoCOOP.pool_count != 0 else None

    if parser.thread:
        logging.info('Multiprocessing Enabled, {} processes created.'.format(model.pool_count))

    if parser.record:
        logging.info('All data snapshots will be written to: {}/environment/, {}/agents/ and {}/settlements/'.format(
            parser.base, parser.base, parser.base))

    if parser.log:
        logging.info('All logged events will be written to the file: {}/events.log'.format(parser.base))

    for i in range(model.iterations):
        model.logger.info('ITERATION: {}'.format(i))

        if not model.debug:
            progress(i, model.iterations)

        model.systemManager.executeSystems()
        if len(model.environment.agents) == 0:
            model.logger.info('POPULATION.ZERO: {}'.format(i))
            break

    logging.info('Simulation Completed:\nTime Elapsed ({}s)'.format(time.time() - start_time))


if __name__ == '__main__':
    main()