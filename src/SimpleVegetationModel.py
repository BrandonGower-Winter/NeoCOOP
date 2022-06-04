
import logging
import math
import numpy
import time

from Agents import Household, HouseholdRelationshipComponent, ResourceComponent, SettlementRelationshipComponent
from ECAgent.Core import Model, Component, System
from ECAgent.Environments import PositionComponent
from ECAgent.Decode import *

from Logging import ILoggable

# Cython Modules
from CythonFunctions import CSoilMoistureSystemFunctions, CVegetationGrowthSystemFunctions, CGlobalEnvironmentCurves

class SimpleGlobalEnvironmentComponent(Component):
    """ This Environment Layer Component is responsible for tracking environmental resources."""

    def __init__(self, agent, model: Model, start_stress_range: [float, float], end_stress_range: [float, float]):
        super().__init__(agent, model)
        self.start_stress_range = start_stress_range
        self.end_stress_range = end_stress_range

class SimpleGlobalEnvironmentSystem(System, IDecodable, ILoggable):
    """ This System calculates the global environmental stress values every iteration."""
    def __init__(self, id: str, model: Model, start_stress_range: [float, float], end_stress_range: [float, float],
                 stress_dict: dict, interpolater_range: int, priority=0, frequency=1, start=0):

        System.__init__(self,id, model, priority, frequency, start)
        IDecodable.__init__(self)
        ILoggable.__init__(self, 'model.GES')

        self.stress_dict = stress_dict
        self.interpolator_range = interpolater_range

        if 'frequency' in self.stress_dict:
            self.stress_dict['frequency'] = SimpleGlobalEnvironmentSystem.convert_to_freq(stress_dict['frequency'],
                                                                                          self.interpolator_range)

        model.environment.addComponent(SimpleGlobalEnvironmentComponent(model.environment, model, start_stress_range,
                                                                        end_stress_range))

        def resource_generator(pos, cells):
            return 0.0

        # Generate the initial resources (Set at 0)
        model.environment.addCellComponent('resources', resource_generator)

    @staticmethod
    def convert_to_freq(f, total_duration):
        return 2 * math.pi * (total_duration / f)

    @staticmethod
    def decode(params: dict):
        return SimpleGlobalEnvironmentSystem(params['id'], params['model'], params['start_stress'],
                                             params['end_stress'], params['stress_dict'],
                                             params['interpolator_range'], priority=params['priority'])

    @staticmethod
    def calcStressVal(startArr, endArr, percentage, func_dict: dict):

        if func_dict['id'] == 'cosine':
            min = CGlobalEnvironmentCurves.cosine_lerp(startArr[0], endArr[0], percentage, func_dict['frequency'])
            max = CGlobalEnvironmentCurves.cosine_lerp(startArr[1], endArr[1], percentage, func_dict['frequency'])
        elif func_dict['id'] == 'exponential':
            min = CGlobalEnvironmentCurves.exponential_lerp(startArr[0], endArr[0], percentage, func_dict['k'])
            max = CGlobalEnvironmentCurves.exponential_lerp(startArr[1], endArr[1], percentage, func_dict['k'])
        elif func_dict['id'] == 'dampened_sinusoidal':
            min = CGlobalEnvironmentCurves.dampening_sinusoidal_lerp(startArr[0], endArr[0], percentage,
                                                                     func_dict['frequency'], func_dict['k'])
            max = CGlobalEnvironmentCurves.dampening_sinusoidal_lerp(startArr[1], endArr[1], percentage,
                                                                     func_dict['frequency'], func_dict['k'])
        elif func_dict['id'] == 'linear_dampened_sinusoidal':
            min = CGlobalEnvironmentCurves.linear_modified_dsinusoidal_lerp(startArr[0], endArr[0], percentage,
                                                                            func_dict['frequency'], func_dict['k'], func_dict['m'])

            # This account for the fact that it can be challenging getting the function to lower bound at zero
            if min < 0.2:
                min = 0.0

            max = CGlobalEnvironmentCurves.linear_modified_dsinusoidal_lerp(startArr[1], endArr[1], percentage,
                                                                            func_dict['frequency'], func_dict['k'], func_dict['m'])

            if max < 0.2:
                max = 0.0

        else:
            min = CGlobalEnvironmentCurves.linear_lerp(startArr[0], endArr[0], percentage)
            max = CGlobalEnvironmentCurves.linear_lerp(startArr[1], endArr[1], percentage)

        return min, max

    def execute(self):

        logging.debug("Generating Global Data Variables...")

        env_comp = self.model.environment.getComponent(SimpleGlobalEnvironmentComponent)

        percentage = min(self.model.systemManager.timestep / self.interpolator_range, 1.0)

        vmin, vmax = SimpleGlobalEnvironmentSystem.calcStressVal(env_comp.start_stress_range, env_comp.end_stress_range,
                                                               percentage, self.stress_dict)

        # Update resource cells
        cells = numpy.ones(self.model.environment.width * self.model.environment.height)

        for i in range(len(cells)):
            cells[i] *= self.model.random.uniform(vmin, vmax)

        self.model.environment.cells.update({'resources': cells})

        # Set Stress
        self.logger.info('GES:  {}'.format((vmin, vmax)))

class SimpleAgentResourceAcquisitionSystem(System, IDecodable, ILoggable):
    """ This system is responsible for executing the Agent Resource Acquisition process. """
    farms_per_patch = 0
    max_acquisition_distance = 0

    @staticmethod
    def decode(params: dict):
        SimpleAgentResourceAcquisitionSystem.farms_per_patch = params['farms_per_patch']
        SimpleAgentResourceAcquisitionSystem.max_acquisition_distance = params['max_acquisition_distance']
        return SimpleAgentResourceAcquisitionSystem(params['id'], params['model'], params['priority'])

    def __init__(self, id: str, model: Model,priority):

        System.__init__(self, id, model, priority=priority)
        IDecodable.__init__(self)
        ILoggable.__init__(self, 'model.RAS')

        def owned_generator(pos, cells):
            return -1
        # And isOwned environment layer
        model.environment.addCellComponent('isOwned', owned_generator)

    def acquire_land(self, household: Household, target: int, last_neighbour_count):
        new_land = target - len(household[ResourceComponent].ownedLand)

        # Get a list of all the available patches of land

        available_land = []

        neighbour_count = last_neighbour_count
        while len(available_land) < new_land:
            temp_land, neighbour_count = [x for x in self.model.environment[SettlementRelationshipComponent
            ].getEmptySettlementNeighbours(household[HouseholdRelationshipComponent].settlementID, new_land - len(available_land),
                                           neighbour_count, available_land)]
            if len(temp_land) != 0:
                available_land += temp_land
            else:  # If there is no land available, break
                break

        toClaim = []
        # Remove least promising land patches
        while new_land > len(toClaim) and len(available_land) > 0:
            choice = self.model.random.choice(available_land)
            toClaim.append(choice)
            available_land.remove(choice)

        for land_id in toClaim:
            household[ResourceComponent].claim_land(land_id)
            self.model.environment.cells.at[land_id, 'isOwned'] = household.id

            self.logger.info('HOUSEHOLD.CLAIM: {} {}'.format(household.id, land_id))

        # Return the neighbour count property to allow for faster searching of available farmland.
        return neighbour_count

    def execute(self):

        start_time = time.time()

        # Instantiate numpy arrays of environment dataframe

        owned_cells = self.model.environment.cells['isOwned'].to_numpy()
        settlement_cells = self.model.environment.cells['isSettlement'].to_numpy()
        resource_cells = self.model.environment.cells['resources'].to_numpy()
        position_cells = self.model.environment.cells['pos'].to_numpy()

        # This is just for dynamic programming purposes
        settlement_neighbour_count = {}

        is_land_available = len([0 for x in range(len(owned_cells)) if owned_cells[x] == -1
                                     and settlement_cells[x] == -1]) > 0

        log_string = ''
        for household in self.model.environment.getAgents():
            # Get Settlement ID
            sID = household[HouseholdRelationshipComponent].settlementID

            # Determine how many patches a household can farm
            able_workers = household[ResourceComponent].able_workers()
            numToFarm = math.ceil(able_workers / SimpleAgentResourceAcquisitionSystem.farms_per_patch)

            # If ownedLand < patches to farm allocate more land to farm
            if len(household[ResourceComponent].ownedLand) < numToFarm and is_land_available:
                settlement_neighbour_count[sID] = self.acquire_land(household, numToFarm,
                                                                        settlement_neighbour_count[sID] if sID in settlement_neighbour_count else 1)

            # Farm numToFarm Cells
            if numToFarm > 0:
                # Select land patches
                farmableLand = [x for x in household[ResourceComponent].ownedLand]

                for i in range(numToFarm):

                    worker_diff = max(able_workers - SimpleAgentResourceAcquisitionSystem.farms_per_patch, 0)
                    workers = (able_workers - worker_diff) / SimpleAgentResourceAcquisitionSystem.farms_per_patch
                    able_workers = worker_diff

                    if len(farmableLand) == 0:
                        break

                    # Remove patches of land randomly
                    patchID = farmableLand.pop(self.model.random.randrange(0, len(farmableLand)))
                    hPos = (household[PositionComponent].x, household[PositionComponent].y)
                    coords = position_cells[patchID]
                    dst = max(abs(coords[0] - hPos[0]), abs(coords[1] - hPos[1]))

                    mad = SimpleAgentResourceAcquisitionSystem.max_acquisition_distance
                    dst_penalty = 1.0 if dst <= mad else 1.0 / (dst - mad)

                    new_resources = resource_cells[patchID] * workers
                    resource_cells[patchID] -= new_resources
                    new_resources *= dst_penalty

                    household[ResourceComponent].resources += new_resources

                    log_string += 'HOUSEHOLD.FARM: {} {} {}\n'.format(household.id, patchID, new_resources)

        # Update Environment Dataframe
        self.model.environment.cells.update({'resources': resource_cells, 'isOwned': owned_cells})
        # Log Events
        self.logger.info(log_string)
        self.logger.info('SYS.TIME: {} {}\n'.format(self.id, time.time() - start_time))

