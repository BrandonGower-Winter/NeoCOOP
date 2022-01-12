import os

import argparse
import multiprocessing
import numpy as np
import time

import ECAgent.Environments as env

from Agents import *
from datetime import datetime
from VegetationModel import *
from Logging import ILoggable

from Progress import progress

# Default Decoder file path
default_path = './resources/traditional/scenario3/'
store_path = 'EgyptModel'


class EgyptModel(Model, IDecodable, ILoggable):

    pool_count = 0
    pool = None

    # Use the scope of the argparser to get the custom seed value. It is a bit hacky but, it is needed for
    # specification of the seed from the command line.
    seed = None

    def __init__(self, width: int, height: int, iterations: int, start_x: int, start_y: int, min_height: int,
                 max_height: int, cellSize: int, debug: bool = True):

        Model.__init__(self, seed=EgyptModel.seed, logger=None)
        IDecodable.__init__(self)
        ILoggable.__init__(self, logger_name='model', level=logging.INFO)

        logging.info('\t-Creating Gridworld')
        self.debug = debug
        self.environment = env.GridWorld(width, height, self)
        self.cellSize = cellSize
        self.iterations = iterations

        self.start_x = start_x
        self.start_y = start_y
        self.min_height = min_height
        self.max_height = max_height

        self.load_environment_layers(parser.file)  # Assumes a parser.file exists

    def load_environment_layers(self, path_to_decoder_file: str):

        logging.info('\t-Loading Heightmap')

        # Get heightmap
        from PIL import Image

        im = Image.open(path_to_decoder_file + 'heightmap.png').convert('L')
        water_im = Image.open(path_to_decoder_file + 'rivermask.png').convert('L')
        soil_mask = Image.open(path_to_decoder_file + 'soilmask.png').convert('L')
        slope_img = Image.open(path_to_decoder_file + 'slope_map.png').convert('L')

        height_diff = self.max_height - self.min_height

        heightmap = []
        water_map = []
        soil_map = []
        slope_map = []

        for x in range(self.start_x, self.start_x + self.environment.width):
            row = []
            water_row = []
            soil_row = []
            slope_row = []

            for y in range(self.start_y, self.start_y + self.environment.height):
                row.append(self.min_height + (im.getpixel((x, y)) / 255.0 * height_diff))
                water_row.append(water_im.getpixel((x, y)))
                soil_row.append((soil_mask.getpixel((x, y)) / 255.0 * 100))
                slope_row.append(slope_img.getpixel((x, y)) / 255.0)

            heightmap.append(row)
            water_map.append(water_row)
            soil_map.append(soil_row)
            slope_map.append(slope_row)

        def elevation_generator_functor(pos, cells):
            return heightmap[pos[0]][pos[1]]

        def is_water_generator(pos, cells):
            return water_map[pos[0]][pos[1]] != 0.0

        def soil_generator(pos, cells):
            return soil_map[pos[0]][pos[1]]

        def slope_generator(pos, cells):
            return 1.0 - slope_map[pos[0]][pos[1]]

        self.environment.addCellComponent('height', elevation_generator_functor)

        logging.info('\t-Generating Watermap')

        self.environment.addCellComponent('isWater', is_water_generator)

        # Generate slope data

        logging.info('\t-Generating Slopemap')

        self.environment.addCellComponent('slope', slope_generator)

        logging.info('\t-Generating Soil Data')

        self.environment.addCellComponent('sand_content', soil_generator)

    @staticmethod
    def decode(params: dict):

        width, height = params['img_width'], params['img_height']
        max_height, min_height = params['max_height'], params['min_height']

        start_x, start_y = 0, 0

        if 'start_x' in params:
            start_x = params['start_x']

        if 'start_y' in params:
            start_y = params['start_y']

        return EgyptModel(width, height, params['iterations'],
                          start_x, start_y, min_height, max_height, params['cell_dim'])


class CellComponent(Component):

    def __init__(self, agent, model):
        super().__init__(agent, model)

        self.height = 0
        self.slope = 0
        self.isWater = False
        self.waterAvailability = 0


def parseArgs():
    """Create the EgyptModel Parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='Path to decoder json file.', default=default_path)
    parser.add_argument('-v', '--visualize', help='Will start a dash applet to display a summary of the simulation.',
                        action='store_true')
    parser.add_argument('-d', '--debug', help='Sets the model to debug mode. Output printed to terminal will be verbose',
                        action='store_true')
    parser.add_argument('-l', '--log', help='Tell the application to generate a log file for this run of the simulation.',
                        action='store_true')
    parser.add_argument('-r', '--record', help='Tell the application to record all of the model data to a vegetation and agent csv file',
                        action='store_true')
    parser.add_argument('-b', '--base', help='Base name of all the generated files.', default=store_path)
    parser.add_argument('--frequency', help='Frequency at which to capture snapshots.', default=1, type=int)
    parser.add_argument('--thread', help='Tell the application to multi-thread your application.',
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


def init_settlements(params : dict):
    model = params['model']
    # Assign Settlement Positions based on selected strategy

    for sID in model.environment[SettlementRelationshipComponent].settlements:
        while True:
            if params['strategy'] == 'cluster':
                cluster = model.random.choice(params['clusters'])
                pos_x = model.random.randrange(max(cluster[0] - params['range'], 0),
                                           min(cluster[0] + params['range'], model.environment.width))
                pos_y = model.random.randrange(max(cluster[1] - params['range'], 0),
                                           min(cluster[1] + params['range'], model.environment.height))
            else:
                pos_x = model.random.randrange(0, model.environment.width)
                pos_y = model.random.randrange(0, model.environment.height)

            unq_id = env.discreteGridPosToID(pos_x, pos_y, model.environment.width)
            if model.environment.cells['isSettlement'][unq_id] == -1 and not model.environment.cells['isWater'][unq_id]:
                model.environment[SettlementRelationshipComponent].settlements[sID].pos.append(unq_id)
                model.environment.cells.at[unq_id, 'isSettlement'] = sID
                model.logger.info('CREATE.SETTLEMENT: {} {}'.format(sID, unq_id))
                break

    # Update num_households to APS
    model.systemManager.systems['APS'].num_households = len(model.environment.agents)

    # Assign Households to Settlements
    for agent in model.environment.getAgents():

        if params['strategy'] == 'grouped':
            s = model.environment[SettlementRelationshipComponent].settlements[agent.id % 4]
            agent[HouseholdRelationshipComponent].peer_resource_transfer_chance = model.random.uniform(
                params['adjust_ranges'][s.id % 4][0], params['adjust_ranges'][s.id % 4][1])
            agent[HouseholdRelationshipComponent].sub_resource_transfer_chance = model.random.uniform(
                params['adjust_ranges'][s.id % 4][2], params['adjust_ranges'][s.id % 4][3])

        else:
            s = model.random.choice(model.environment[SettlementRelationshipComponent].settlements)  # Get ID

        model.environment[SettlementRelationshipComponent].add_household_to_settlement(agent, s.id)  # Add household

        # Set household position
        h_pos = model.environment.cells['pos'][
            model.environment[SettlementRelationshipComponent].settlements[s.id].pos[-1]]

        agent[PositionComponent].x = h_pos[0]
        agent[PositionComponent].y = h_pos[1]


        model.logger.info('CREATE.HOUSEHOLD: {} {} {}'.format(agent.id, s.id, model.environment[
            SettlementRelationshipComponent].settlements[s.id].pos[-1]))

    # Clean up empty settlements
    for sID in [s for s in model.environment[SettlementRelationshipComponent].settlements]:

        if len(model.environment[SettlementRelationshipComponent].settlements[sID].occupants) == 0:
            model.environment[SettlementRelationshipComponent].remove_settlement(sID)
            model.logger.info('REMOVE.SETTLEMENT.DELETED: {}'.format(sID))


if __name__ == '__main__':

    parser = parseArgs()

    start_time = time.time()

    addDebugHandler(parser.debug)

    logging.info('Simulation run invoked at: {}'.format(datetime.now()))

    if parser.record or parser.log:
        os.mkdir(parser.base)

    addFileHandler(parser.base + '/events.log', parser.log)

    logging.info("Creating Model...")
    EgyptModel.seed = parser.seed
    model = JsonDecoder().decode(parser.file + '/decoder.json')

    if parser.record:
        logging.info('\t-Adding Environment Data Collector.')
        os.mkdir(parser.base + '/environment')
        model.systemManager.addSystem(VegetationSnapshotCollector('VSC', model, parser.base + '/environment', parser.frequency))
        logging.info('\t-Adding Agent Data Collector.')
        os.mkdir(parser.base + '/agents')
        model.systemManager.addSystem(AgentSnapshotCollector('ASC', model, parser.base + '/agents', parser.frequency))
        logging.info('\t-Adding Settlement Data Collector.')
        os.mkdir(parser.base + '/settlements')
        model.systemManager.addSystem(SettlementSnapshotCollector('SSC', model, parser.base + '/settlements', parser.frequency))

    logging.info('...Done!')

    model.debug = parser.debug
    EgyptModel.pool_count = multiprocessing.cpu_count() if parser.thread else 0
    EgyptModel.pool = multiprocessing.Pool(processes=EgyptModel.pool_count) if EgyptModel.pool_count != 0 else None

    if parser.thread:
        logging.info('Multiprocessing Enabled, {} processes created.'.format(model.pool_count))

    if parser.record:
        logging.info('All data snapshots will be written to: {}/environment/, {}/agents/ and {}/settlements/'.format(
            parser.base,parser.base, parser.base))

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