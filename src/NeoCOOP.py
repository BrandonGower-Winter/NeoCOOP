import ECAgent.Environments as env
import logging
import numpy

from Agents import HouseholdRelationshipComponent, SettlementRelationshipComponent
from ECAgent.Core import Model
from ECAgent.Decode import IDecodable
from ECAgent.Environments import PositionComponent
from Logging import ILoggable

class NeoCOOP(Model, IDecodable, ILoggable):

    # Environment Settings
    ENV_SIMPLE = 0
    ENV_VEGETATION_MODEL = 1

    # Env Layer Keys
    HEIGHT_KEY = 'height'
    SLOPE_KEY = 'slope'
    IS_WATER_KEY = 'iswater'
    FLOOD_KEY = 'flood'
    SAND_KEY = 'sand'
    CLAY_KEY = 'clay'

    # Img Aliases
    HEIGHTMAP_ALIAS = 'height.png'
    SLOPEMAP_ALIAS = 'slope.png'
    IS_WATER_ALIAS = 'iswater.png'
    FLOOD_ALIAS = 'flood.png'
    SAND_ALIAS = 'sand.png'
    CLAY_ALIAS = 'clay.png'

    parser = None
    pool_count = 0
    pool = None

    # Use the scope of the argparser to get the custom seed value. It is a bit hacky but, it is needed for
    # specification of the seed from the command line.
    seed = None

    def __init__(self, width: int, height: int, iterations: int, start_x: int, start_y: int, min_height: int,
                 max_height: int, cell_size: int, env_type:int = ENV_SIMPLE, env_flood = False, debug: bool = True):

        Model.__init__(self, seed=NeoCOOP.seed, logger=None)
        IDecodable.__init__(self)
        ILoggable.__init__(self, logger_name='model', level=logging.INFO)

        self.debug = debug
        self.environment = env.GridWorld(width, height, self)
        self.cell_size = cell_size
        self.iterations = iterations

        self.start_x = start_x
        self.start_y = start_y
        self.min_height = min_height
        self.max_height = max_height

        self.env_type = env_type
        self.enable_flood = env_flood

        self.load_environment_layers(NeoCOOP.parser.file)  # Assumes a parser.file exists

    def load_environment_layers(self, path_to_decoder_file: str):

        if self.env_type == NeoCOOP.ENV_SIMPLE:
            logging.info('\t-Creating Simple Gridworld of Size %s x %s' % (self.environment.width, self.environment.height))
        else:
            logging.info('\t-Creating Gridworld with Vegetation Model of Size %s x %s' % (self.environment.width, self.environment.height))
            # Get heightmap
            from PIL import Image

            logging.info('\t-Loading Heightmap')
            height_diff = self.max_height - self.min_height
            heightmap = self.min_height + (
                numpy.asarray(Image.open(path_to_decoder_file + NeoCOOP.HEIGHTMAP_ALIAS).convert('L')) / 255.0 * height_diff)

            def elevation_generator_functor(pos, cells):
                return heightmap[pos[1]][pos[0]]

            self.environment.addCellComponent(NeoCOOP.HEIGHT_KEY, elevation_generator_functor)

            # Load slope data
            logging.info('\t-Creating Slopemap')
            slope_map = numpy.asarray(Image.open(path_to_decoder_file + NeoCOOP.SLOPEMAP_ALIAS).convert('L')) / 255.0

            def slope_generator(pos, cells):
                return 1.0 if slope_map[pos[1]][pos[0]] < 1.0 else 0.0

            self.environment.addCellComponent(NeoCOOP.SLOPE_KEY, slope_generator)

            # Load Soil Data
            logging.info('\t-Loading Soil Data')

            sand_map = numpy.asarray(Image.open(path_to_decoder_file + NeoCOOP.SAND_ALIAS).convert('L')) / 255.0 * 100
            clay_map = numpy.asarray(Image.open(path_to_decoder_file + NeoCOOP.CLAY_ALIAS).convert('L')) / 255.0 * 100

            def sand_generator(pos, cells):
                return sand_map[pos[1]][pos[0]]

            def clay_generator(pos, cells):
                return clay_map[pos[1]][pos[0]]

            self.environment.addCellComponent(NeoCOOP.SAND_KEY, sand_generator)
            self.environment.addCellComponent(NeoCOOP.CLAY_KEY, clay_generator)

            # Load Soil Data
            logging.info('\t-Loading Water Cell Data')

            water_map = numpy.asarray(Image.open(path_to_decoder_file + NeoCOOP.IS_WATER_ALIAS).convert('L')) > 0.005

            def water_generator(pos, cells):
                return water_map[pos[1]][pos[0]]

            self.environment.addCellComponent(NeoCOOP.IS_WATER_KEY, water_generator)

            # Load Flood Map
            if self.enable_flood:
                logging.info('\t-Loading Flood Map Data')
                flood_map = self.min_height + (numpy.asarray(Image.open(path_to_decoder_file + NeoCOOP.FLOOD_ALIAS).convert('L')) / 255.0 * height_diff)

                def flood_generator(pos, cells):
                    return flood_map[pos[1]][pos[0]]

                self.environment.addCellComponent(NeoCOOP.FLOOD_KEY, flood_generator)

    @staticmethod
    def decode(params: dict):

        # Set environment size (defaults to 100x100)
        width = params['img_width'] if 'img_width' in params else 100
        height = params['img_height'] if 'img_height' in params else 100

        # Defaults to 1ha
        cell_dim = params['cell_dim'] if 'cell_dim' in params else 100

        # Set max and min heightmap size
        max_height = params['max_height'] if 'max_height' in params else 0.0
        min_height = params['min_height'] if 'min_height' in params else 0.0

        # Set map offsets
        start_x = params['start_x'] if 'start_x' in params else 0
        start_y = params['start_y'] if 'start_y' in params else 0

        # Set ENV Type (Defaults to SIMPLE)
        env_type = params['env_type'] if 'env_type' in params else NeoCOOP.ENV_SIMPLE

        # Grab desired simulation time (defaults to 1000)
        iterations =  params['iterations'] if 'iterations' in params else 1000

        # Get flood dynamics mode
        env_flood = 'env_flood' in params

        # Check if file aliases have been changed
        if 'heightmap_alias' in params:
            NeoCOOP.HEIGHTMAP_ALIAS = params['heightmap_alias']
        if 'slopemap_alias' in params:
            NeoCOOP.SLOPEMAP_ALIAS = params['slopemap_alias']
        if 'iswater_alias' in params:
            NeoCOOP.IS_WATER_ALIAS = params['iswater_alias']
        if 'flood_alias' in params:
            NeoCOOP.FLOOD_ALIAS = params['flood_alias']
        if 'sand_alias' in params:
            NeoCOOP.SAND_ALIAS = params['sand_alias']
        if 'clay_alias' in params:
            NeoCOOP.CLAY_ALIAS = params['clay_alias']

        return NeoCOOP(width, height, iterations, start_x, start_y, min_height,
                       max_height, cell_dim, env_type=env_type, env_flood=env_flood)

# Function for initializing agent positions
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
            if model.environment.cells['isSettlement'][unq_id] == -1:
                model.environment[SettlementRelationshipComponent].settlements[sID].pos.append(unq_id)
                model.environment.cells.at[unq_id, 'isSettlement'] = sID
                model.logger.info('CREATE.SETTLEMENT: {} {}'.format(sID, unq_id))
                break

    # Update num_households to APS
    model.systemManager.systems['APS'].num_households = len(model.environment.agents)

    div_length = len(params['adjust_ranges']) if 'adjust_ranges' in params else 0
    settlement_length = len(model.environment[SettlementRelationshipComponent].settlements)
    # Assign Households to Settlements
    for agent in model.environment.getAgents():

        if params['strategy'] == 'grouped':
            s = model.environment[SettlementRelationshipComponent].settlements[agent.id % settlement_length]
            agent[HouseholdRelationshipComponent].peer_resource_transfer_chance = model.random.uniform(
                params['adjust_ranges'][s.id % div_length][0], params['adjust_ranges'][s.id % div_length][1])
            agent[HouseholdRelationshipComponent].sub_resource_transfer_chance = model.random.uniform(
                params['adjust_ranges'][s.id % div_length][2], params['adjust_ranges'][s.id % div_length][3])

        elif params['strategy'] == 'grouped2':
            s = model.environment[SettlementRelationshipComponent].settlements[0]
            agent[HouseholdRelationshipComponent].peer_resource_transfer_chance = model.random.uniform(
                params['adjust_ranges'][agent.id % div_length][0], params['adjust_ranges'][agent.id % div_length][1])
            agent[HouseholdRelationshipComponent].sub_resource_transfer_chance = model.random.uniform(
                params['adjust_ranges'][agent.id % div_length][2], params['adjust_ranges'][agent.id % div_length][3])

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