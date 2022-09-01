import logging
import math
import NeoCOOP
import numpy as np

from ECAgent.Core import *
from ECAgent.Environments import *
from ECAgent.Decode import *
from ECAgent.Collectors import Collector

from Logging import ILoggable

# Cython Modules
from CythonFunctions import CSoilMoistureSystemFunctions, CVegetationGrowthSystemFunctions, CGlobalEnvironmentCurves


def lerp(start, end, percentage):
    return start + (end - start) * percentage


class GlobalEnvironmentComponent(Component):
    """ This Environment Layer Component is responsible for tracking generated rainfall and temperature data. """

    def __init__(self, agent, model: Model, start_temp: [int], end_temp: [int], start_rainfall: [int],
                 end_rainfall: [int], start_solar: [int], end_solar: [int] , start_flood: [int], end_flood: [int],
                 soil_depth: float):
        super().__init__(agent, model)

        self.start_temp = start_temp
        self.end_temp = end_temp

        self.start_rainfall = start_rainfall
        self.end_rainfall = end_rainfall

        self.start_solar = start_solar
        self.end_solar = end_solar

        self.start_flood = start_flood
        self.end_flood = end_flood

        self.soil_depth = soil_depth

        self.temp = -1
        self.rainfall = -1
        self.solar = -1
        self.flood = -1

        # For easier streaming of objects in memory
        del self.model
        del self.agent


class SoilMoistureComponent(Component):
    """This Environment Layer Component just keeps track of the constants necessary for calculating P.E.T."""

    def __init__(self, agent, model: Model, L: int, N: int, I: float):
        super().__init__(agent, model)

        self.L = L
        self.N = N
        self.I = I  # Heat Index
        self.alpha = (0.000000675 * I * I * I) - (0.0000771 * I * I) - (0.01792 * I) + 0.49239

        # For streaming processes
        del self.model
        del self.agent


class VegetationGrowthComponent(Component):
    """ This Environment Layer Component just keeps track of constants needed to calculate vegetation growth. """

    def __init__(self, agent, model: Model, init_pop: int, carry_pop: int, growth_rate: float, decay_rate: float,
                 ideal_moisture: float):
        super().__init__(agent, model)

        self.init_pop = init_pop
        self.carry_pop = carry_pop
        self.growth_rate = growth_rate
        self.decay_rate = decay_rate
        self.ideal_moisture = ideal_moisture

        # For Streaming Purposes
        del self.model
        del self.agent


class GlobalEnvironmentSystem(System, IDecodable, ILoggable):

    MONTH_MULTIPLIER = 1

    """ This System calculates the global rainfall and temperature values every iteration."""
    def __init__(self, id: str, model: Model, start_temp: [int], end_temp: [int], start_rainfall: [int],
                 end_rainfall: [int], start_solar : [int], end_solar : [int], start_flood : [int], end_flood : [int],
                 soil_depth: float, temperature_dict: dict, rainfall_dict: dict, solar_dict : dict, flood_dict : dict,
                 interpolater_range: int, priority=0, frequency=1, start=0, end=maxsize):

        System.__init__(self,id, model, priority, frequency, start, end)
        IDecodable.__init__(self)
        ILoggable.__init__(self, 'model.GES')

        self.temperature_dict = temperature_dict
        self.rainfall_dict = rainfall_dict
        self.solar_dict = solar_dict
        self.flood_dict = flood_dict
        self.interpolator_range = interpolater_range

        if 'frequency' in self.temperature_dict:
            self.temperature_dict['frequency'] = GlobalEnvironmentSystem.convert_to_freq(temperature_dict['frequency'], self.interpolator_range)

        if 'frequency' in self.rainfall_dict:
            self.rainfall_dict['frequency'] = GlobalEnvironmentSystem.convert_to_freq(rainfall_dict['frequency'], self.interpolator_range)

        if 'frequency' in self.solar_dict:
            self.solar_dict['frequency'] = GlobalEnvironmentSystem.convert_to_freq(solar_dict['frequency'], self.interpolator_range)

        if self.flood_dict is not None and 'frequency' in self.flood_dict:
            self.flood_dict['frequency'] = GlobalEnvironmentSystem.convert_to_freq(flood_dict['frequency'], self.interpolator_range)

        model.environment.addComponent(GlobalEnvironmentComponent(model.environment, model, start_temp, end_temp,
                                                                  start_rainfall, end_rainfall, start_solar,
                                                                  end_solar, start_flood, end_flood, soil_depth))


    @staticmethod
    def convert_to_freq(f, total_duration):
        return 2 * math.pi * (total_duration / f)

    @staticmethod
    def decode(params: dict):

        if 'month_multiplier' in params:
            GlobalEnvironmentSystem.MONTH_MULTIPLIER = params['month_multiplier']

        return GlobalEnvironmentSystem(params['id'], params['model'], params['start_temp'], params['end_temp'],
                                       params['start_rainfall'], params['end_rainfall'], params['start_solar'],
                                       params['end_solar'],
                                       params['start_flood'] if 'start_flood' in params else None,
                                       params['end_flood'] if 'end_flood' in params else None,
                                       params['soil_depth'], params['temperature_dict'],
                                       params['rainfall_dict'], params['solar_dict'],
                                       params['flood_dict'] if 'flood_dict' in params else None,
                                       params['interpolator_range'], priority=params['priority'])

    @staticmethod
    def calcMinMaxGlobalVals(startArr, endArr, percentage, func_dict: dict):
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
            if min < 0.1:
                min = 0.0

            max = CGlobalEnvironmentCurves.linear_modified_dsinusoidal_lerp(startArr[1], endArr[1], percentage,
                                                     func_dict['frequency'], func_dict['k'], func_dict['m'])

            if max < 0.1:
                max = 0.0

        else:
            min = CGlobalEnvironmentCurves.linear_lerp(startArr[0], endArr[0], percentage)
            max = CGlobalEnvironmentCurves.linear_lerp(startArr[1], endArr[1], percentage)

        return min, max

    def execute(self):

        logging.debug("Generating Global Data Variables...")

        env_comp = self.model.environment.getComponent(GlobalEnvironmentComponent)

        percentage = min(self.model.systemManager.timestep / self.interpolator_range, 1.0)

        # Calculate the rainfall, temperature, solar and flood values for each month

        # Set Temperature
        env_comp.temp = self.model.random.uniform(*GlobalEnvironmentSystem.calcMinMaxGlobalVals(env_comp.start_temp, env_comp.end_temp, percentage,
                                                                                                self.temperature_dict))

        # Set Rainfall
        env_comp.rainfall = self.model.random.uniform(*GlobalEnvironmentSystem.calcMinMaxGlobalVals(env_comp.start_rainfall, env_comp.end_rainfall,
                                                                                                    percentage, self.rainfall_dict))

        # Set Solar
        env_comp.solar = self.model.random.uniform(*GlobalEnvironmentSystem.calcMinMaxGlobalVals(env_comp.start_solar, env_comp.end_solar,
                                                                                                 percentage, self.solar_dict))

        # Set Flood
        if self.model.enable_flood:
            env_comp.flood = self.model.random.uniform(*GlobalEnvironmentSystem.calcMinMaxGlobalVals(env_comp.start_flood, env_comp.end_flood,
                                                                                                 percentage, self.flood_dict))
            self.logger.info('GES:  {} {} {} {}'.format(
                str(env_comp.temp),
                str(env_comp.rainfall),
                str(env_comp.solar),
                str(env_comp.flood)
            ))
        else:
            self.logger.info('GES:  {} {} {}'.format(
                str(env_comp.temp),
                str(env_comp.rainfall),
                str(env_comp.solar)
            ))

    def __str__(self):
        return 'Global_Properties:\n\nTemperatures: {}C\nRainfall: {}mm\n Solar: {}MJ/m^2'.format(
            self.model.environment.getComponent(GlobalEnvironmentComponent).temp,
            self.model.environment.getComponent(GlobalEnvironmentComponent).rainfall,
            self.model.environment.getComponent(GlobalEnvironmentComponent).solar
        )


class SoilMoistureSystem(System, IDecodable):
    """ This system is responsible for calculating the available soil moisture in each cell"""

    MOISTURE_KEY = 'moisture'
    EET_KEY = 'eet'

    def __init__(self, id: str, model: Model, L: int, N: int, I: float, priority=0, frequency=1, start=0, end=maxsize):
        super().__init__(id, model, priority, frequency, start, end)

        model.environment.addComponent(SoilMoistureComponent(model.environment, model, L, N, I))

        depth = self.model.environment[GlobalEnvironmentComponent].soil_depth
        def moisture_generator(pos, cells):
            cellID = discreteGridPosToID(pos[0], pos[1], model.environment.width)
            return 0.0 #CSoilMoistureSystemFunctions.wfc(cells[NeoCOOP.NeoCOOP.SAND_KEY][cellID], cells[NeoCOOP.NeoCOOP.CLAY_KEY][cellID])

        def eet_generator(pos, cells):
            return 0.0

        # Generate the initial moisture based on the soil sand content
        model.environment.addCellComponent(SoilMoistureSystem.MOISTURE_KEY, moisture_generator)
        model.environment.addCellComponent(SoilMoistureSystem.EET_KEY, eet_generator)
        self.lastAvgMoisture = 0.0


    @staticmethod
    def decode(params: dict):

        return SoilMoistureSystem(params['id'], params['model'], params['L'], params['N'], params['I'],
                                priority=params['priority'])

    def get_soil_moisture(self, unq_id: int):
        return self.model.environment.cells[SoilMoistureSystem.MOISTURE_KEY][unq_id]

    def execute(self):

        if not self.model.enable_flood:
            output = CSoilMoistureSystemFunctions.SMProcess(self.model.environment.cells[SoilMoistureSystem.MOISTURE_KEY].to_numpy(),
                    self.model.environment.cells[NeoCOOP.NeoCOOP.SAND_KEY].to_numpy(), self.model.environment.cells[NeoCOOP.NeoCOOP.CLAY_KEY].to_numpy(),
                        self.model.environment[SoilMoistureComponent], self.model.environment[GlobalEnvironmentComponent])
        else:
            output = CSoilMoistureSystemFunctions.SMProcess_with_Flood(self.model.environment.cells[SoilMoistureSystem.MOISTURE_KEY].to_numpy(),
                    self.model.environment.cells[NeoCOOP.NeoCOOP.SAND_KEY].to_numpy(), self.model.environment.cells[NeoCOOP.NeoCOOP.CLAY_KEY].to_numpy(),
                    self.model.environment.cells[NeoCOOP.NeoCOOP.HEIGHT_KEY].to_numpy(), self.model.environment.cells[NeoCOOP.NeoCOOP.FLOOD_KEY].to_numpy(),
                        self.model.environment[SoilMoistureComponent], self.model.environment[GlobalEnvironmentComponent])

        self.model.environment.cells.update({SoilMoistureSystem.MOISTURE_KEY: output[0],
                                             SoilMoistureSystem.EET_KEY: output[1]})


class VegetationGrowthSystem(System, IDecodable):

    VEGETATION_KEY = 'vegetation'

    def __init__(self, id: str, model: Model, init_pop: int, carry_pop: int, growth_rate: float, decay_rate: float,
                 ideal_moisture, priority=0, frequency=1, start=0, end=maxsize):

        super().__init__(id, model, priority=priority, frequency=frequency, start=start, end=end)

        model.environment.addComponent(VegetationGrowthComponent(self.model.environment, model, init_pop,
                                                                 carry_pop, growth_rate, decay_rate, ideal_moisture))

        # Create a random range of values for the initial vegetation population
        max_carry = min(carry_pop, int(init_pop + (init_pop * init_pop/carry_pop)))
        min_carry = max(0, int(init_pop - (init_pop * init_pop/carry_pop)))

        def vegetation_generator(pos, cells):
            return self.model.random.uniform(min_carry, max_carry)
        # Generate Initial vegetation layer
        model.environment.addCellComponent(VegetationGrowthSystem.VEGETATION_KEY, vegetation_generator)

    def execute(self):

        outputs = CVegetationGrowthSystemFunctions.VGProcess(self.model.environment.cells[VegetationGrowthSystem.VEGETATION_KEY].to_numpy(),
                                                        self.model.environment.cells[SoilMoistureSystem.EET_KEY].to_numpy(),
                                                        self.model.environment.cells[NeoCOOP.NeoCOOP.SLOPE_KEY].to_numpy(),
                                                        self.model.environment[VegetationGrowthComponent],
                                                        self.model.environment[SoilMoistureComponent],
                                                        self.model.environment[GlobalEnvironmentComponent],
                                                        self.model.cell_size ** 2)

        self.model.environment.cells.update({VegetationGrowthSystem.VEGETATION_KEY: outputs})

        logging.debug('...Vegetation System...')
        logging.debug('Vegetation: {} Mean Moisture: {}'.format(np.mean(self.model.environment.cells[VegetationGrowthSystem.VEGETATION_KEY]),
                                                        np.mean(self.model.environment.cells[SoilMoistureSystem.MOISTURE_KEY])))

    @staticmethod
    def decode(params: dict):
        return VegetationGrowthSystem(params['id'], params['model'], params['init_pop'], params['carry_pop'],
                                      params['growth_rate'], params['decay_rate'], params['ideal_moisture'],
                                      priority=params['priority'])


class VegetationSnapshotCollector(Collector):

    def __init__(self, id: str, model, file_name, frequency: int = 1):
        super().__init__(id, model, frequency=frequency)

        self.file_name = file_name
        self.headers = [x for x in [SoilMoistureSystem.MOISTURE_KEY, VegetationGrowthSystem.VEGETATION_KEY, 'isOwned', 'isSettlement']
                        if x in self.model.environment.cells]

    def collect(self):
        self.model.environment.cells.to_csv(self.file_name + '/iteration_{}.csv'.format(self.model.systemManager.timestep),
                                 mode='w', index=True, header=True, columns=self.headers)

