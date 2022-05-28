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
                 end_rainfall: [int], start_solar: [int], end_solar: [int] , soil_depth: float):
        super().__init__(agent, model)

        self.start_temp = start_temp
        self.end_temp = end_temp
        self.start_rainfall = start_rainfall
        self.end_rainfall = end_rainfall
        self.start_solar = start_solar
        self.end_solar = end_solar

        self.soil_depth = soil_depth

        self.temp = []
        self.rainfall = []
        self.solar = []

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
    """ This System calculates the global rainfall and temperature values every iteration."""
    def __init__(self, id: str, model: Model, start_temp: [int], end_temp: [int], start_rainfall: [int],
                 end_rainfall: [int], start_solar, end_solar, soil_depth: float,
                 temperature_dict: dict, rainfall_dict: dict, solar_dict : dict,
                 interpolater_range: int, priority=0, frequency=1, start=0, end=maxsize):

        System.__init__(self,id, model, priority, frequency, start, end)
        IDecodable.__init__(self)
        ILoggable.__init__(self, 'model.GES')

        self.temperature_dict = temperature_dict
        self.rainfall_dict = rainfall_dict
        self.solar_dict = solar_dict
        self.interpolator_range = interpolater_range

        if 'frequency' in self.temperature_dict:
            self.temperature_dict['frequency'] = GlobalEnvironmentSystem.convert_to_freq(temperature_dict['frequency'], self.interpolator_range)

        if 'frequency' in self.rainfall_dict:
            self.rainfall_dict['frequency'] = GlobalEnvironmentSystem.convert_to_freq(rainfall_dict['frequency'], self.interpolator_range)

        if 'frequency' in self.solar_dict:
            self.solar_dict['frequency'] = GlobalEnvironmentSystem.convert_to_freq(solar_dict['frequency'], self.interpolator_range)

        model.environment.addComponent(GlobalEnvironmentComponent(model.environment, model, start_temp, end_temp,
                                                                  start_rainfall, end_rainfall, start_solar,
                                                                  end_solar, soil_depth))


    @staticmethod
    def convert_to_freq(f, total_duration):
        return 2 * math.pi * (total_duration / f)

    @staticmethod
    def decode(params: dict):
        return GlobalEnvironmentSystem(params['id'], params['model'], params['start_temp'], params['end_temp'],
                                       params['start_rainfall'], params['end_rainfall'], params['start_solar'],
                                       params['end_solar'], params['soil_depth'], params['temperature_dict'],
                                       params['rainfall_dict'], params['solar_dict'], params['interpolator_range'],
                                       priority=params['priority'])

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

        env_comp = self.model.environment.getComponent(GlobalEnvironmentComponent)

        env_comp.temp.clear()
        env_comp.rainfall.clear()
        env_comp.solar.clear()

        percentage = min(self.model.systemManager.timestep / self.interpolator_range, 1.0)

        # Set Temperature
        min_t, max_t = GlobalEnvironmentSystem.calcMinMaxGlobalVals(env_comp.start_temp, env_comp.end_temp, percentage,
                                                                    self.temperature_dict)

        # Set Rainfall
        min_r, max_r = GlobalEnvironmentSystem.calcMinMaxGlobalVals(env_comp.start_rainfall, env_comp.end_rainfall,
                                                                    percentage, self.rainfall_dict)

        # Set Solar
        min_s, max_s = GlobalEnvironmentSystem.calcMinMaxGlobalVals(env_comp.start_solar, env_comp.end_solar,
                                                                    percentage, self.rainfall_dict)

        # Calculate the rainfall and temperature values for each month
        for i in range(12):
            env_comp.temp.append(self.model.random.uniform(min_t, max_t))
            env_comp.rainfall.append(self.model.random.uniform(min_r, max_r))
            env_comp.solar.append(self.model.random.uniform(min_s, max_s))

        logging.debug(self)
        self.logger.info('GES:  {} {} {}'.format(
            str(np.mean(self.model.environment.getComponent(GlobalEnvironmentComponent).temp)),
            str(np.mean(self.model.environment.getComponent(GlobalEnvironmentComponent).rainfall)),
            str(np.mean(self.model.environment.getComponent(GlobalEnvironmentComponent).solar))
        ))

    def __str__(self):
        return 'Global_Properties:\n\nTemperatures: {}C\nRainfall: {}mm\n Solar: {}MJ/m^2'.format(
            self.model.environment.getComponent(GlobalEnvironmentComponent).temp,
            self.model.environment.getComponent(GlobalEnvironmentComponent).rainfall,
            self.model.environment.getComponent(GlobalEnvironmentComponent).solar
        )


class SoilMoistureSystem(System, IDecodable):
    """ This system is responsible for calculating the availalbe soil moisture in each cell"""
    def __init__(self, id: str, model: Model, L: int, N: int, I: float, priority=0, frequency=1, start=0, end=maxsize):
        super().__init__(id, model, priority, frequency, start, end)

        model.environment.addComponent(SoilMoistureComponent(model.environment, model, L, N, I))

        def moisture_generator(pos, cells):
            cellID = discreteGridPosToID(pos[0], pos[1], model.environment.width)
            return CSoilMoistureSystemFunctions.wfc(model.environment.getComponent(GlobalEnvironmentComponent).soil_depth,
                      cells['sand_content'][cellID])

        # Generate the initial moisture based on the soil sand content
        model.environment.addCellComponent('moisture', moisture_generator)

        self.lastAvgMoisture = 0.0


    @staticmethod
    def decode(params: dict):
        return SoilMoistureSystem(params['id'], params['model'], params['L'], params['N'], params['I'],
                                priority=params['priority'])

    def get_soil_moisture(self, unq_id: int):
        return self.model.environment.cells['moisture'][unq_id]

    def execute(self):

        if NeoCOOP.NeoCOOP.pool is None:
            outputs = [CSoilMoistureSystemFunctions.SMProcess(self.model.environment.cells,
                                                  self.model.environment[SoilMoistureComponent],
                                                  self.model.environment[GlobalEnvironmentComponent])]
        else:

            dfs = np.array_split(self.model.environment.cells[['pos', 'height','moisture', 'sand_content']],
                                 self.model.pool_count)
            sm_comp = [self.model.environment[SoilMoistureComponent] for i in range(self.model.pool_count)]
            gec = [self.model.environment[GlobalEnvironmentComponent] for i in range(self.model.pool_count)]

            outputs = NeoCOOP.NeoCOOP.pool.starmap(CSoilMoistureSystemFunctions.SMProcess, zip(dfs, sm_comp, gec))

        final_list = np.concatenate(outputs)

        self.model.environment.cells.update({'moisture': final_list})


class VegetationGrowthSystem(System, IDecodable):

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
        model.environment.addCellComponent('vegetation', vegetation_generator)

    def execute(self):

        if NeoCOOP.NeoCOOP.pool is None:
            outputs = [CVegetationGrowthSystemFunctions.VGProcess(self.model.environment.cells,
                                                        self.model.environment[VegetationGrowthComponent],
                                                        self.model.environment[SoilMoistureComponent],
                                                        self.model.environment[GlobalEnvironmentComponent],
                                                        self.model.cellSize ** 2,
                                                        self.model.random)]
        else:

            dfs = []
            vg_comp = []
            sm_comp = []
            gec = []
            randoms = []

            edge = self.model.environment.width * self.model.environment.height
            division = edge // self.model.pool_count

            sub_df = self.model.environment.cells[['pos', 'height', 'moisture', 'vegetation', 'isOwned',
                                                   'slope']]

            for t in range(self.model.pool_count):
                t_min = division * t
                t_max = division * (t + 1) if t + 1 != self.model.pool_count else edge

                dfs.append(dfs.append(sub_df.iloc[t_min:t_max]))
                vg_comp.append(self.model.environment[VegetationGrowthComponent])
                sm_comp.append(self.model.environment[SoilMoistureComponent])
                gec.append(self.model.environment[GlobalEnvironmentComponent])
                randoms.append(self.model.random)

            outputs = NeoCOOP.NeoCOOP.pool.starmap(CVegetationGrowthSystemFunctions.VGProcess,
                                              zip(dfs, vg_comp, sm_comp, gec, randoms))

        final_veg_list = np.concatenate([o[0] for o in outputs])
        final_soil_list = np.concatenate([o[1] for o in outputs])

        self.model.environment.cells.update({'moisture': final_soil_list,
                                             'vegetation': final_veg_list})

        logging.debug('...Vegetation System...')
        logging.debug('Vegetation: {} Mean Moisture: {}'.format(np.mean(self.model.environment.cells['vegetation']),
                                                        np.mean(self.model.environment.cells['moisture'])))

    @staticmethod
    def decode(params: dict):
        return VegetationGrowthSystem(params['id'], params['model'], params['init_pop'], params['carry_pop'],
                                      params['growth_rate'], params['decay_rate'], params['ideal_moisture'],
                                      priority=params['priority'])


class VegetationSnapshotCollector(Collector):

    def __init__(self, id: str, model, file_name, frequency: int = 1):
        super().__init__(id, model, frequency=frequency)

        self.file_name = file_name
        self.headers = [x for x in ['moisture', 'vegetation', 'isOwned', 'isSettlement', 'resources']
                        if x in self.model.environment.cells]

    def collect(self):
        self.model.environment.cells.to_csv(self.file_name + '/iteration_{}.csv'.format(self.model.systemManager.timestep),
                                 mode='w', index=True, header=True, columns=self.headers)

