import ECAgent.Environments as env
import logging

from Agents import HouseholdRelationshipComponent, SettlementRelationshipComponent
from ECAgent.Core import Model
from ECAgent.Decode import IDecodable
from ECAgent.Environments import PositionComponent
from Logging import ILoggable

class NeoCOOP(Model, IDecodable, ILoggable):

    parser = None
    pool_count = 0
    pool = None

    # Use the scope of the argparser to get the custom seed value. It is a bit hacky but, it is needed for
    # specification of the seed from the command line.
    seed = None

    def __init__(self, width: int, height: int, iterations: int, start_x: int, start_y: int, min_height: int,
                 max_height: int, cellSize: int, debug: bool = True):

        Model.__init__(self, seed=NeoCOOP.seed, logger=None)
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

        self.load_environment_layers(NeoCOOP.parser.file)  # Assumes a parser.file exists

    def load_environment_layers(self, path_to_decoder_file: str):

        logging.info('\t-Loading Heightmap')

        # Get heightmap
        from PIL import Image

        im = Image.open(path_to_decoder_file + 'heightmap.png').convert('L')
        soil_mask = Image.open(path_to_decoder_file + 'soilmask.png').convert('L')
        slope_img = Image.open(path_to_decoder_file + 'slope_map.png').convert('L')

        height_diff = self.max_height - self.min_height

        heightmap = []
        soil_map = []
        slope_map = []

        for x in range(self.start_x, self.start_x + self.environment.width):
            row = []
            soil_row = []
            slope_row = []

            for y in range(self.start_y, self.start_y + self.environment.height):
                row.append(self.min_height + (im.getpixel((x, y)) / 255.0 * height_diff))
                soil_row.append((soil_mask.getpixel((x, y)) / 255.0 * 100))
                slope_row.append(slope_img.getpixel((x, y)) / 255.0)

            heightmap.append(row)
            soil_map.append(soil_row)
            slope_map.append(slope_row)

        def elevation_generator_functor(pos, cells):
            return heightmap[pos[0]][pos[1]]

        def soil_generator(pos, cells):
            return soil_map[pos[0]][pos[1]]

        def slope_generator(pos, cells):
            return 1.0 - slope_map[pos[0]][pos[1]]

        self.environment.addCellComponent('height', elevation_generator_functor)

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

        return NeoCOOP(width, height, params['iterations'],
                          start_x, start_y, min_height, max_height, params['cell_dim'])

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

    # Assign Households to Settlements
    for agent in model.environment.getAgents():

        if params['strategy'] == 'grouped':
            s = model.environment[SettlementRelationshipComponent].settlements[agent.id % 20]
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