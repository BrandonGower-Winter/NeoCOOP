import numpy as np
cimport numpy as np

from libc.math cimport cos, exp

# General Functions

cdef float lerp( float a, float b, float t):
    return a + (b - a) * t

cdef unsigned int discreteGridPosToID(unsigned int x, unsigned int y, unsigned int width):
    return (y * width) + x

cdef int absolute(int x):
    return x if x > 0 else x * -1

#######################################################################################################################

cdef class CSoilMoistureSystemFunctions:

    @staticmethod
    def thornthwaite(int day_length, int days, int avg_temp, float heat_index, float alpha):
        return 16.0 * (day_length / 12.0) * (days / 30.0) * ((10.0 * avg_temp / heat_index) ** alpha)

    @staticmethod
    def calcRdr(int sand_content, float soil_m, float soil_depth):
        cdef float a, b

        a = CSoilMoistureSystemFunctions.alpha(sand_content)
        b = CSoilMoistureSystemFunctions.beta(sand_content)

        return (1 + a) / (1 + a * ((soil_m / soil_depth) ** b))

    @staticmethod
    cdef float alpha(int sand_content):
        cdef int clay, sand_sqrd

        clay = 100 - sand_content
        sand_sqrd = sand_content ** 2

        return (2.71828 ** (-4.396 - 0.0715 * clay - 0.000488 * sand_sqrd - 0.00004258 * sand_sqrd * clay)) * 100

    @staticmethod
    cdef float beta(int sand_content):
        cdef int clay

        clay = 100 - sand_content
        return -3.140 - 0.000000222 * (clay ** 2) - 0.00003484 * (sand_content ** 2) * clay

    @staticmethod
    def wfc(float soil_depth, int sand_content):
        return soil_depth * lerp(0.3, 0.7, 1 - (sand_content / 100.0))

    @staticmethod
    def is_flooded(height, pos, sm_comp, global_env_comp):
        if global_env_comp.flood == 0:
            return False
        else:
            return height < global_env_comp.flood + sm_comp.avgWaterHeight(pos)

    @staticmethod
    def SMProcess(df, sm_comp, global_env_comp) -> [float]:
        cdef int i
        cdef float PET

        # COMPUTE PET Values
        cdef np.ndarray[double] sand_content = df['sand_content'].to_numpy(dtype=np.double)
        cdef np.ndarray[double] clay_content = 100 - sand_content
        cdef np.ndarray[double] sand_sqrd = sand_content ** 2
        cdef np.ndarray[double] alpha = (2.71828 ** (-4.396 - 0.0715 * clay_content - 0.000488 * sand_sqrd - 0.00004258 * sand_sqrd * clay_content)) * 100
        cdef np.ndarray[double] beta = -3.140 - 0.000000222 * (clay_content ** 2) - 0.00003484 * sand_sqrd * clay_content

        cdef np.ndarray[double] heightmap = df['height'].to_numpy()
        cdef np.ndarray[double] soil_vals = df['moisture'].to_numpy()

        cdef np.ndarray is_flooded = np.full(len(soil_vals), False, dtype=np.bool)

        if global_env_comp.flood != 0:
            water_heights = np.array([sm_comp.avgWaterHeight(pos) for pos in df['pos']], dtype=np.double)
            is_flooded = heightmap < (water_heights + global_env_comp.flood)

        cdef np.ndarray[double] flooded_vals = global_env_comp.soil_depth * (0.3 + (0.4 * (1 - sand_content/100.0)))
        cdef np.ndarray[double] non_flooded_vals = soil_vals.copy()
        # For Each Month
        for i in range(12):

            # Calculate Potential Evaporation
            PET = CSoilMoistureSystemFunctions.thornthwaite(sm_comp.L, sm_comp.N, global_env_comp.temp[i],
                                                            sm_comp.I, sm_comp.alpha)

            # Calculate new soil values
            if PET > global_env_comp.rainfall[i]: # When Rainfall is lower than PET
                non_flooded_vals = np.maximum(non_flooded_vals - (PET - global_env_comp.rainfall[i]) *
                        ((1 + alpha) / (1 + alpha * ((soil_vals / global_env_comp.soil_depth) ** beta))), 0)

            else: # When Rainfall is higher than PET
                non_flooded_vals = non_flooded_vals + (global_env_comp.rainfall[i] - PET)
                non_flooded_vals = np.minimum(non_flooded_vals, flooded_vals)

        # Apply new soil values
        soil_vals = np.where(is_flooded, flooded_vals, non_flooded_vals)
        return soil_vals

#######################################################################################################################

cdef class CVegetationGrowthSystemFunctions:

    @staticmethod
    def Logistic_Growth(float pop, int carry_cap, float growth_rate):
        return growth_rate * pop * ((carry_cap - pop) / carry_cap) if carry_cap > pop else 0

    @staticmethod
    def decay(float val, float rate):
        return val * rate

    @staticmethod
    def waterPenalty(float moisture, float moisture_ideal):

        if moisture < moisture_ideal:
            return moisture / moisture_ideal, 0.0
        else:
            return 1.0, moisture - moisture_ideal

    @staticmethod
    cdef float tOpt(float temp):
        return -0.0005 * ((temp - 20.0) ** 2) + 1

    @staticmethod
    def tempPenalty(float temperature, random):
        cdef float topt

        topt = CVegetationGrowthSystemFunctions.tOpt(temperature)
        return random.uniform(0.8 - 0.0005 * (topt ** 2.0), 0.8 + 0.02 * topt)

    @staticmethod
    def VGProcess(df, vg_comp, sm_comp, ge_comp, random) -> ([float], [float]):

        cdef float r, topt


        cdef np.ndarray[double] veg_cells = df['vegetation'].to_numpy()
        cdef np.ndarray[double] moist_cells = df['moisture'].to_numpy()
        cdef np.ndarray[double] slope_penalty_cells = df['slope'].to_numpy()

        cdef np.ndarray[double] rs = np.zeros(len(veg_cells))

        # Get the available moisture
        cdef np.ndarray[double] moist_avail = moist_cells - (veg_cells / vg_comp.carry_pop * vg_comp.ideal_moisture)

        # Calculate water penalties
        np.clip(moist_avail / vg_comp.ideal_moisture , 0.0, 1.0, out=rs)

        # Calculate remaining moisture
        np.clip(moist_avail - vg_comp.ideal_moisture, 0.0, None, out=moist_cells)

        # Calculate the temperature penalty
        topt = CVegetationGrowthSystemFunctions.tOpt(np.mean(ge_comp.temp))
        rs *= random.uniform(0.8 - 0.0005 * (topt ** 2.0), 0.8 + 0.02 * topt)

        # Apply slope penalty
        rs *= slope_penalty_cells

        # Update Vegetation Values:
        cdef float filter_cond = (0.001 * vg_comp.carry_pop)
        _filter = [veg_cells <= filter_cond, veg_cells > filter_cond]
        choicelist = [veg_cells + rs * vg_comp.growth_rate,
                      np.clip(veg_cells - (veg_cells * vg_comp.decay_rate) + (veg_cells * rs * vg_comp.growth_rate),
                              0.0, vg_comp.carry_pop)]
        veg_cells = np.select(_filter, choicelist)

        return veg_cells, moist_cells

#######################################################################################################################

cdef class CAgentResourceConsumptionSystemFunctions:

    @staticmethod
    cdef (float, float) consume(float resources, float required_resources):
        cdef float hunger, remaining_resources

        # This is actually the inverse of hunger with 1.0 being completely 'full' and zero being 'starving'
        hunger = min(1.0, resources / required_resources)
        remaining_resources = max(0.0, resources - required_resources)

        return remaining_resources, hunger

    @staticmethod
    def ARCProcess(object resComp) -> float:
        cdef float req_res, rem_res, hunger

        req_res = resComp.required_resources()
        rem_res, hunger = CAgentResourceConsumptionSystemFunctions.consume(resComp.resources, req_res)
        resComp.hunger = hunger
        resComp.satisfaction += hunger
        resComp.resources = rem_res

        return req_res * hunger

#######################################################################################################################

cdef class CAgentResourceAcquisitionFunctions:
    @staticmethod
    def num_to_farm(float threshold, int maxFarm, object random):
        cdef int numToFarm, index
        numToFarm = 0
        for index in range(maxFarm):
            if random.random() < threshold:
                numToFarm += 1
        return numToFarm

    @staticmethod
    def num_to_farm_phouse(float threshold, int maxFarm, object random, float farm_utility, float forage_utility):
        cdef int numToFarm, index
        cdef bint is_farm_max

        numToFarm = 0
        for index in range(maxFarm):
            is_farm_max = farm_utility > forage_utility
            # A satisfied house has a hunger of 1.0
            if random.random() < threshold:
                if is_farm_max:
                    numToFarm += 1
            else:
                numToFarm += random.randint(0, 1)

        return numToFarm

    @staticmethod
    def generateNeighbours(int xPos, int yPos, int width, int height, int radius, np.ndarray owned_cells,
                           np.ndarray settlement_cells, np.ndarray water_cells):
        cdef int x, y, id
        cdef list toReturn

        toReturn = []
        for x in range(max(xPos - radius, 0), min(xPos + radius, width)):
            for y in range(max(yPos - radius, 0), min(yPos + radius, height)):

                id = discreteGridPosToID(x, y, width)

                if owned_cells[id] == -1 and settlement_cells[id] == -1 and not water_cells[id]:
                    toReturn.append(id)

        return toReturn

    @staticmethod
    def forage(int patch_id, int workers, np.ndarray vegetation_cells, int consumption_rate, float forage_multiplier,
               int per_patch) -> float:
        cdef float veg_diff, farmed_res

        veg_diff = max(vegetation_cells[patch_id]
                       - consumption_rate * (workers / per_patch), 0.0)

        farmed_res = vegetation_cells[patch_id] - veg_diff
        vegetation_cells[patch_id] = veg_diff
        return farmed_res * forage_multiplier

    @staticmethod
    def farm(int patch_id, int workers, (int, int) house_pos, (int, int) coords, float temperature, int max_acquisition_distance,
             int moisture_consumption_rate, int crop_gestation_period, int farming_production_rate, int farms_per_patch,
             np.ndarray height_cells, np.ndarray moisture_cells, np.ndarray slope_cells,
             object sm_comp, object ge_comp, object random) -> float:
        # Calculate penalties

        cdef int dst
        cdef float dst_penalty, tmp_penalty, wtr_penalty, moisture_remain, crop_yield

        dst = max(absolute(coords[0] - house_pos[0]), absolute(coords[1] - house_pos[1]))
        dst_penalty = 1.0 if dst <= max_acquisition_distance else 1.0 / (dst - max_acquisition_distance)

        tmp_penalty = CVegetationGrowthSystemFunctions.tempPenalty(temperature, random)

        wtr_penalty, moisture_remain = CVegetationGrowthSystemFunctions.waterPenalty(moisture_cells[patch_id],
                                                                                     moisture_consumption_rate / crop_gestation_period)
        # Calculate Crop Yield

        if not CSoilMoistureSystemFunctions.is_flooded(height_cells[patch_id], coords, sm_comp, ge_comp):
            # Adjust soil moisture if cell not flooded
            moisture_cells[patch_id] = moisture_remain

        crop_yield = farming_production_rate * wtr_penalty * tmp_penalty * slope_cells[patch_id] * (workers / farms_per_patch)

        return int(crop_yield * dst_penalty)


#######################################################################################################################

cdef class CGlobalEnvironmentCurves:

    @staticmethod
    def linear_lerp(float a, float b, float t):
        return lerp(a, b, t)

    @staticmethod
    cdef float cosine(float t, float frequency):
        return 0.5 * cos( frequency * t) + 0.5

    @staticmethod
    def cosine_lerp(float a, float b, float t, float frequency):
        return lerp(a, b, 1.0 - CGlobalEnvironmentCurves.cosine(t, frequency))


    @staticmethod
    cdef float exponential(float t, float k):
        return exp(-k * t)

    @staticmethod
    def exponential_lerp(float a, float b, float t, float k):
        return lerp(a, b, 1.0 - CGlobalEnvironmentCurves.exponential(t, k))

    @staticmethod
    cdef float dampening_sinusoidal(float t, float frequency, float k):
        return CGlobalEnvironmentCurves.cosine(t, frequency) * CGlobalEnvironmentCurves.exponential(t, k)

    @staticmethod
    def dampening_sinusoidal_lerp(float a, float b, float t, float frequency, float k):
        return lerp(a, b, 1.0 - CGlobalEnvironmentCurves.dampening_sinusoidal(t, frequency, k))

    @staticmethod
    cdef float linear_modified_dsinusoidal(float t, float frequency, float k, float m):
        cdef float res

        res = CGlobalEnvironmentCurves.dampening_sinusoidal(t, frequency, k)

        return res + (m * (1-t) * (1-res))

    @staticmethod
    def linear_modified_dsinusoidal_lerp(float a, float b, float t, float frequency, float k, float m):
        return lerp(a, b, 1.0 - CGlobalEnvironmentCurves.linear_modified_dsinusoidal(t, frequency, k, m))


cdef class CAgentUtilityFunctions:

    @staticmethod
    def xtent(float w, float d, float b, float m):
        return (w ** b) - m * d

    @staticmethod
    def xtent_distribution(np.ndarray ws, np.ndarray ds, float b, float m):
        cdef np.ndarray result
        result = (ws ** b) - m * ds

        return result / np.abs(result).sum()
        #return (result - result.min()) / (result.max() - result.min())
