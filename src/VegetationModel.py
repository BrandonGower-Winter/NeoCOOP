import math
import numpy as np

from ECAgent.Core import *
from ECAgent.Environments import *
from ECAgent.Decode import *
from ECAgent.Collectors import Collector

from main import EgyptModel
from Logging import ILoggable

# Cython Modules
from CythonFunctions import CSoilMoistureSystemFunctions, CVegetationGrowthSystemFunctions, CGlobalEnvironmentCurves


def lerp(start, end, percentage):
    return start + (end - start) * percentage


class GlobalEnvironmentComponent(Component):

    def __init__(self, agent, model: Model, start_temp: [int], end_temp: [int], start_rainfall: [int],
                 end_rainfall: [int], start_flood: [int], end_flood: [int], soil_depth: float):
        super().__init__(agent, model)

        self.start_temp = start_temp
        self.end_temp = end_temp
        self.start_rainfall = start_rainfall
        self.end_rainfall = end_rainfall
        self.start_flood = start_flood
        self.end_flood = end_flood

        self.soil_depth = soil_depth

        self.temp = []
        self.rainfall = []
        self.flood = -1

        # For easier streaming of objects in memory
        del self.model
        del self.agent


class SoilMoistureComponent(Component):

    def __init__(self, agent, model: Model, L: int, N: int, I: float, flood_cell_divide: int):
        super().__init__(agent, model)

        self.L = L
        self.N = N
        self.I = I  # Heat Index
        self.alpha = (0.000000675 * I * I * I) - (0.0000771 * I * I) - (0.01792 * I) + 0.49239

        # Get avg water cell height
        isWaterArr = model.environment.cells['isWater'].tolist()
        total = 0.0
        count = 0

        self.avg_water_heights = []
        self.flood_cell_divide = flood_cell_divide
        self.flood_divide_ratio = self.model.environment.width // flood_cell_divide

        for y_block in range(math.ceil(self.model.environment.height / flood_cell_divide)):
            for x_block in range(math.ceil(self.model.environment.width / flood_cell_divide)):

                total = 0.0
                count = 0

                for x in range(flood_cell_divide):
                    x_coord = x + (x_block * flood_cell_divide)
                    for y in range(flood_cell_divide):
                        y_coord = y + (y_block * flood_cell_divide)

                        unq_id = discreteGridPosToID(x_coord, y_coord, model.environment.width)

                        if isWaterArr[unq_id]:
                            total += model.environment.cells['height'][unq_id]
                            count +=1

                self.avg_water_heights.append(0.0 if count == 0 else total / count)

        # Fill other cells using cells that have water
        has_changes = True
        while has_changes:
            has_changes = False

            for i in range(len(self.avg_water_heights)):
                if self.avg_water_heights[i] == 0.0:
                    # Derive flood height using neighbouring cells

                    neighbours = [self.avg_water_heights[i - self.flood_divide_ratio] if i >= self.flood_divide_ratio else 0.0,
                                  self.avg_water_heights[i + self.flood_divide_ratio] if i < len(self.avg_water_heights) - self.flood_divide_ratio else 0.0,
                                  self.avg_water_heights[i - 1] if i % self.flood_divide_ratio != 0 else 0.0,
                                  self.avg_water_heights[i + 1] if (i+1) % self.flood_divide_ratio != 0 else 0.0]

                    valid_cells = len([x for x in neighbours if x != 0.0])
                    self.avg_water_heights[i] = sum(neighbours) / valid_cells if valid_cells != 0 else 0.0

                    if self.avg_water_heights[i] != 0.0:
                        has_changes = True

        # For streaming processes
        del self.model
        del self.agent

    def avgWaterHeight(self, pos: (int, int)):
        return self.avg_water_heights[discreteGridPosToID(pos[0] // self.flood_cell_divide,
                                                          pos[1] // self.flood_cell_divide, self.flood_divide_ratio)]


class VegetationGrowthComponent(Component):

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

    def __init__(self, id: str, model: Model, start_temp: [int], end_temp: [int], start_rainfall: [int],
                 end_rainfall: [int], start_flood: [int], end_flood: [int], soil_depth: float,
                 temperature_dict: dict, rainfall_dict: dict, flood_dict: dict, interpolater_range: int,
                 priority=0, frequency=1, start=0, end=maxsize):

        System.__init__(self,id, model, priority, frequency, start, end)
        IDecodable.__init__(self)
        ILoggable.__init__(self, 'model.GES')

        self.temperature_dict = temperature_dict
        self.rainfall_dict = rainfall_dict
        self.flood_dict = flood_dict
        self.interpolator_range = interpolater_range

        model.environment.addComponent(GlobalEnvironmentComponent(model.environment, model, start_temp, end_temp,
                                                                  start_rainfall, end_rainfall, start_flood, end_flood,
                                                                  soil_depth))

    @staticmethod
    def decode(params: dict):
        return GlobalEnvironmentSystem(params['id'], params['model'], params['start_temp'], params['end_temp'],
                                       params['start_rainfall'], params['end_rainfall'], params['start_flood'],
                                       params['end_flood'], params['soil_depth'], params['temperature_dict'],
                                       params['rainfall_dict'], params['flood_dict'], params['interpolator_range'],
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

        percentage = min(self.model.systemManager.timestep / self.interpolator_range, 1.0)

        # Set Temperature
        min_t, max_t = GlobalEnvironmentSystem.calcMinMaxGlobalVals(env_comp.start_temp, env_comp.end_temp, percentage,
                                                                    self.temperature_dict)

        # Set Rainfall
        min_r, max_r = GlobalEnvironmentSystem.calcMinMaxGlobalVals(env_comp.start_rainfall, env_comp.end_rainfall,
                                                                    percentage, self.rainfall_dict)
        # Set Flooding
        min_f, max_f = GlobalEnvironmentSystem.calcMinMaxGlobalVals(env_comp.start_flood, env_comp.end_flood,
                                                                    percentage, self.flood_dict)
        env_comp.flood = self.model.random.uniform(min_f, max_f)

        for i in range(12):

            env_comp.temp.append(self.model.random.uniform(min_t, max_t))
            env_comp.rainfall.append(self.model.random.uniform(min_r, max_r))

        logging.debug(self)
        self.logger.info('GES:  {} {} {}'.format(
            str(np.mean(self.model.environment.getComponent(GlobalEnvironmentComponent).temp)),
            str(np.mean(self.model.environment.getComponent(GlobalEnvironmentComponent).rainfall)),
            str(self.model.environment.getComponent(GlobalEnvironmentComponent).flood)
        ))

    def __str__(self):
        return 'Global_Properties:\n\nTemperatures: {}C\nRainfall: {}mm\nFlood: {}m\n'.format(
            self.model.environment.getComponent(GlobalEnvironmentComponent).temp,
            self.model.environment.getComponent(GlobalEnvironmentComponent).rainfall,
            self.model.environment.getComponent(GlobalEnvironmentComponent).flood)


class SoilMoistureSystem(System, IDecodable):

    def __init__(self, id: str, model: Model, L: int, N: int, I: float, flood_cell_size: int,
                 priority=0, frequency=1, start=0, end=maxsize):
        super().__init__(id, model, priority, frequency, start, end)

        model.environment.addComponent(SoilMoistureComponent(model.environment, model, L, N, I, flood_cell_size))

        def moisture_generator(pos, cells):
            cellID = discreteGridPosToID(pos[0], pos[1], model.environment.width)
            return 0.0 if cells['isWater'][cellID] else SoilMoistureSystem.wfc(model.environment.getComponent(GlobalEnvironmentComponent
                                                                            ).soil_depth, cells['sand_content'][cellID]) * min(1.0,
                        math.pow(cells['height'][cellID] / model.environment.getComponent(SoilMoistureComponent).avgWaterHeight(pos), 2))

        model.environment.addCellComponent('moisture', moisture_generator)

        self.lastAvgMoisture = 0.0


    @staticmethod
    def decode(params: dict):
        return SoilMoistureSystem(params['id'], params['model'], params['L'], params['N'], params['I'],
                                  params['flood_cell_divide'], priority=params['priority'])

    @staticmethod
    def thornthwaite(day_length : int, days: int, avg_temp: int, heat_index : float, alpha: float):
        return 16 * (day_length/12) * (days/30) * math.pow(10*avg_temp/heat_index, alpha)

    @staticmethod
    def calcRdr(sand_content, soil_m, soil_depth):
        a = SoilMoistureSystem.alpha(sand_content)
        b = SoilMoistureSystem.beta(sand_content)
        return (1 + a)/(1 + a * math.pow(soil_m/soil_depth, b))

    @staticmethod
    def alpha(sand_content):
        clay = 100 - sand_content
        sand_sqrd = math.pow(sand_content, 2)
        return math.exp(-4.396 - 0.0715 * clay - 0.000488 * sand_sqrd -
                        0.00004258 * sand_sqrd * clay) * 100

    @staticmethod
    def beta(sand_content):
        clay = 100 - sand_content
        return -3.140 - 0.000000222 * math.pow(clay, 2) - 0.00003484 * math.pow(sand_content, 2) * clay

    @staticmethod
    def wfc(soil_depth, sand_content):
        return soil_depth * lerp(0.3, 0.7, 1 - (sand_content/100.0))

    def get_soil_moisture(self, unq_id: int):
        sm_comp = self.model.environment.getComponent(SoilMoistureComponent)
        global_env_comp = self.model.environment.getComponent(GlobalEnvironmentComponent)

        if self.is_flooded(unq_id):
            return SoilMoistureSystem.wfc(global_env_comp.soil_depth, self.model.environment.cells['sand_content'][unq_id])
        else:
            return self.model.environment.cells['moisture'][unq_id]

    @staticmethod
    def is_flooded(height, pos, sm_comp, global_env_comp):
        return height < global_env_comp.flood + sm_comp.avgWaterHeight(pos)

    @staticmethod
    def SMProcess(df, sm_comp, global_env_comp) -> [float]:

        soil_vals = df['moisture'].to_numpy()
        for row in df.itertuples():
            if row.isWater:
                continue

            if SoilMoistureSystem.is_flooded(row.height, row.pos, sm_comp, global_env_comp):
                soil_vals[row.Index] = CSoilMoistureSystemFunctions.wfc(global_env_comp.soil_depth, row.sand_content)
                continue

            for i in range(12):
                PET = CSoilMoistureSystemFunctions.thornthwaite(
                    sm_comp.L, sm_comp.N, global_env_comp.temp[i], sm_comp.I, sm_comp.alpha)

                if PET > global_env_comp.rainfall[i]:
                    rdr = CSoilMoistureSystemFunctions.calcRdr(row.sand_content,
                                       soil_vals[row.Index] + global_env_comp.rainfall[i], global_env_comp.soil_depth)
                    moisture = soil_vals[row[0]] - (PET - global_env_comp.rainfall[i]) * rdr
                    soil_vals[row.Index] = moisture if moisture > 0 else 0

                else:
                    wfc = CSoilMoistureSystemFunctions.wfc(global_env_comp.soil_depth, row.sand_content)

                    moisture = soil_vals[row.Index] + (
                                global_env_comp.rainfall[i] - PET)
                    soil_vals[row.Index] = moisture if moisture < wfc else wfc

        return soil_vals

    def execute(self):

        if EgyptModel.pool is None:
            outputs = [CSoilMoistureSystemFunctions.SMProcess(self.model.environment.cells,
                                                  self.model.environment[SoilMoistureComponent],
                                                  self.model.environment[GlobalEnvironmentComponent])]
        else:

            dfs = np.array_split(self.model.environment.cells[['pos', 'height','moisture', 'isWater', 'sand_content']],
                                 self.model.pool_count)
            sm_comp = [self.model.environment[SoilMoistureComponent] for i in range(self.model.pool_count)]
            gec = [self.model.environment[GlobalEnvironmentComponent] for i in range(self.model.pool_count)]

            outputs = EgyptModel.pool.starmap(CSoilMoistureSystemFunctions.SMProcess, zip(dfs, sm_comp, gec))

        final_list = np.concatenate(outputs)

        self.model.environment.cells.update({'moisture': final_list})


class VegetationGrowthSystem(System, IDecodable):

    def __init__(self, id: str, model: Model, init_pop: int, carry_pop: int, growth_rate: float, decay_rate: float,
                 ideal_moisture, priority=0, frequency=1, start=0, end=maxsize):

        super().__init__(id, model, priority=priority, frequency=frequency, start=start, end=end)

        model.environment.addComponent(VegetationGrowthComponent(self.model.environment, model, init_pop,
                                                                 carry_pop, growth_rate, decay_rate, ideal_moisture))

        # Create a random range of values for the inititial vegetation population
        max_carry = min(carry_pop, int(init_pop + (init_pop * init_pop/carry_pop)))
        min_carry = max(0, int(init_pop - (init_pop * init_pop/carry_pop)))

        def vegetation_generator(pos, cells):
            cellID = discreteGridPosToID(pos[0], pos[1], model.environment.width)
            return 0.0 if cells['isWater'][cellID] else self.model.random.uniform(min_carry, max_carry)

        model.environment.addCellComponent('vegetation', vegetation_generator)

    @staticmethod
    def Logistic_Growth(pop: float, carry_cap: int, growth_rate: float):
        return growth_rate * pop * ((carry_cap - pop)/carry_cap) if carry_cap > pop else 0

    @staticmethod
    def decay(val: float, rate: float):
        return val * rate

    @staticmethod
    def waterPenalty(moisture: float, moisture_ideal: float):

        if moisture < moisture_ideal:
            return moisture/moisture_ideal, 0.0
        else:
            return 1.0, moisture - moisture_ideal

    @staticmethod
    def tOpt(temp: float):
        return -0.0005 * math.pow(temp - 20.0, 2) + 1

    @staticmethod
    def tempPenalty(temperature: float, random):
        topt = VegetationGrowthSystem.tOpt(temperature)
        return random.uniform(0.8 - 0.0005 * math.pow(topt, 2), 0.8 + 0.02 * topt)

    def execute(self):

        if EgyptModel.pool is None:
            outputs = [CVegetationGrowthSystemFunctions.VGProcess(self.model.environment.cells,
                                                        self.model.environment[VegetationGrowthComponent],
                                                        self.model.environment[SoilMoistureComponent],
                                                        self.model.environment[GlobalEnvironmentComponent],
                                                        self.model.random)]
        else:

            dfs = []
            vg_comp = []
            sm_comp = []
            gec = []
            randoms = []

            edge = self.model.environment.width * self.model.environment.height
            division = edge // self.model.pool_count

            sub_df = self.model.environment.cells[['pos', 'height', 'moisture', 'isWater', 'vegetation', 'isOwned',
                                                   'slope']]

            for t in range(self.model.pool_count):
                t_min = division * t
                t_max = division * (t + 1) if t + 1 != self.model.pool_count else edge

                dfs.append(dfs.append(sub_df.iloc[t_min:t_max]))
                vg_comp.append(self.model.environment[VegetationGrowthComponent])
                sm_comp.append(self.model.environment[SoilMoistureComponent])
                gec.append(self.model.environment[GlobalEnvironmentComponent])
                randoms.append(self.model.random)

            outputs = EgyptModel.pool.starmap(CVegetationGrowthSystemFunctions.VGProcess,
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


class SoilContentSystem(System, IDecodable):
    def __init__(self, id: str, model: Model, sand_content_range, degradation_factor, priority=0, frequency=1, start=0, end=maxsize):
        super().__init__(id, model, priority, frequency, start, end)

        self.sand_content_range = sand_content_range
        self.degradation_factor = degradation_factor

    def execute(self):

        sand_cells = self.model.environment.cells['sand_content'].tolist()
        vegetation_cells = self.model.environment.cells['vegetation']

        for i in range(len(sand_cells)):
            sand_cells[i] = lerp(self.sand_content_range[0], self.sand_content_range[1],
                                  1.0 - vegetation_cells[i]/self.model.environment.getComponent(VegetationGrowthComponent).carry_pop)

        # For each cell
        # If flooded: leave alone
        # If farmed: degrade by degradation factor
        # If neither:

        self.model.environment.cells.update({'sand_content': sand_cells})

    @staticmethod
    def decode(params: dict):
        return SoilContentSystem(params['id'], params['model'], params['sand_content_range'], params['degradation_factor'],
                                 priority=params['priority'])


class VegetationCollector(Collector, IDecodable):

    def __init__(self, id: str, model: Model):
        super().__init__(id, model)

        self.records.append([])
        self.records.append([])

    def collect(self):

        wh = self.model.environment.width * self.model.environment.height
        self.records[1].append(np.mean(
            [
                self.model.environment.cells['vegetation'][i] for i in range(wh)
                if not self.model.environment.cells['isWater'][i]
            ]
        ))
        self.records[0].append(np.mean([
                self.model.environment.cells['moisture'][i] for i in range(wh)
                if not self.model.environment.cells['isWater'][i]
            ]
        ))

    @staticmethod
    def decode(params: dict):
        return VegetationCollector(params['id'], params['model'])


class VegetationSnapshotCollector(Collector):

    def __init__(self, id: str, model, file_name, frequency: int = 1):
        super().__init__(id, model, frequency=frequency)

        self.file_name = file_name
        self.headers = ['moisture', 'vegetation', 'isOwned', 'isSettlement']

    def collect(self):
        self.model.environment.cells.to_csv(self.file_name + '/iteration_{}.csv'.format(self.model.systemManager.timestep),
                                 mode='w', index=True, header=True, columns=self.headers)

