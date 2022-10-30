import numpy as np
cimport numpy as np

from libc.math cimport cos, sin, exp, log10

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
    def wfc(int sand_content, int clay_content):
        # Function returns m3 (of moisture) / m3 (of soil)
        cdef float wfc = 0.332 - 0.0007251 * sand_content * 0.1276 * log10(clay_content)
        return  wfc

    @staticmethod
    def SMProcess(soil_vals, sand_content, clay_content, sm_comp, global_env_comp) -> [float]:
        cdef int i
        cdef float PET
        cdef np.ndarray[double] EET
        cdef np.ndarray[double] RDR

        # COMPUTE PET Values
        cdef np.ndarray[double] sand_sqrd = sand_content ** 2
        cdef np.ndarray[double] alpha = (2.71828 ** (-4.396 - 0.0715 * clay_content - 0.000488 * sand_sqrd - 0.00004258 * sand_sqrd * clay_content)) * 100
        cdef np.ndarray[double] beta = -3.140 - 0.000000222 * (clay_content ** 2) - 0.00003484 * sand_sqrd * clay_content

        cdef np.ndarray[double] max_vals = (0.332 - 0.0007251 * sand_content * 0.1276 * np.log10(clay_content))

        # Calculate Potential Evaporation
        PET = CSoilMoistureSystemFunctions.thornthwaite(sm_comp.L, sm_comp.N, global_env_comp.temp,
                                                        sm_comp.I, sm_comp.alpha) / 1000.0

        # Calculate new soil values
        if PET > global_env_comp.rainfall: # When Rainfall is lower than PET

            RDR = ((1 + alpha) / (1 + alpha * (soil_vals ** np.maximum(beta, 0.0))))

            EET = np.minimum(global_env_comp.rainfall +  (PET - global_env_comp.rainfall) * RDR,
                             global_env_comp.rainfall + soil_vals - 0.17)

            soil_vals = np.maximum(soil_vals - (PET - global_env_comp.rainfall) * RDR, 0)

        else: # When Rainfall is higher than PET
            soil_vals = soil_vals + (global_env_comp.rainfall - PET)
            soil_vals = np.minimum(soil_vals, max_vals)
            EET = np.full(len(soil_vals), PET)


        return soil_vals, EET / PET


    @staticmethod
    def SMProcess_with_Flood(soil_vals, sand_content, clay_content, height_cells, flood_heights, rain_mask, sm_comp, global_env_comp) -> [float]:
        cdef int i
        cdef float PET
        cdef np.ndarray[double] EET
        cdef np.ndarray[double] RDR

        # COMPUTE PET Values
        cdef np.ndarray[double] sand_sqrd = sand_content ** 2
        cdef np.ndarray[double] alpha = (2.71828 ** (-4.396 - 0.0715 * clay_content - 0.000488 * sand_sqrd - 0.00004258 * sand_sqrd * clay_content)) * 100
        cdef np.ndarray[double] beta = -3.140 - 0.000000222 * (clay_content ** 2) - 0.00003484 * sand_sqrd * clay_content
        cdef np.ndarray[double] rain_cells = rain_mask * global_env_comp.rainfall

        clay_content[clay_content < 0.00000001] = 0.00000001 # Prevent division by zero warnings
        cdef np.ndarray[double] max_vals = 0.332 - 0.0007251 * sand_content * 0.1276 * np.where(clay_content != 0.0, np.log10(clay_content), 0.0)

        # Calculate Potential Evaporation
        PET = CSoilMoistureSystemFunctions.thornthwaite(sm_comp.L, sm_comp.N, global_env_comp.temp,
                                                        sm_comp.I, sm_comp.alpha) / 1000.0

        cdef np.ndarray is_flooded = height_cells < global_env_comp.flood + flood_heights

        soil_vals = np.where(is_flooded, max_vals, soil_vals)

        # Calculate new soil values
        RDR = ((1 + alpha) / (1 + alpha * (soil_vals ** np.maximum(beta, 0.0))))
        soil_vals = np.where(PET > rain_cells, np.maximum(soil_vals - (PET - rain_cells) * RDR, 0), np.minimum(soil_vals + (rain_cells - PET), max_vals))

        EET = np.where(PET > rain_cells, np.minimum(rain_cells +  (PET - rain_cells) * RDR, rain_cells + soil_vals - 0.17), np.full(len(soil_vals), PET))
        EET = np.where(is_flooded, PET, EET)


        return soil_vals, EET / PET

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
            return 0.5 + moisture / moisture_ideal, 0.0
        else:
            return 1.0, moisture - moisture_ideal

    @staticmethod
    cdef float tOpt(float temp):
        return 1.1814 / (1 + exp(0.2 * (10.0 - temp))) / (1 + exp(0.3 * (-30.0 + temp)))

    @staticmethod
    def tempPenalty(float temperature, random):
        return CVegetationGrowthSystemFunctions.tOpt(temperature)

    @staticmethod
    def VGProcess(veg_cells, eet_cells, slope_cells, vg_comp, sm_comp, ge_comp, area) -> ([float], [float]):

        cdef float r
        cdef np.ndarray[double] rs = np.zeros(len(veg_cells))

        # Calculate water penalties
        np.clip(0.5 + 0.5 * eet_cells , 0.0, 1.0, out=rs)

        # Calculate the temperature penalty
        rs *=  CVegetationGrowthSystemFunctions.tOpt(ge_comp.temp)

        # Apply slope penalty
        veg_cells = rs * slope_cells

        return veg_cells

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
    def ARCProcess(object resComp, float storageEfficiency) -> float:
        cdef float curr_res, req_res, rem_res, hunger, consumed, i_left
        cdef int i

        curr_res = resComp.resources
        req_res = resComp.required_resources()
        rem_res, hunger = CAgentResourceConsumptionSystemFunctions.consume(curr_res, req_res)
        resComp.hunger = hunger
        resComp.iter_since_last_move += 1
        resComp.satisfaction += hunger
        resComp.resources = rem_res * storageEfficiency

        if rem_res == 0.0:
            for i in range(len(resComp.storage_decay)):
                resComp.storage_decay[i] = 0
        else:
            consumed = curr_res - rem_res
            for i in range(len(resComp.storage_decay)):

                i_left = max(0.0, resComp.storage_decay[i] - consumed)
                consumed -= resComp.storage_decay[i]
                resComp.storage_decay[i] = i_left

                if consumed < 0.005:
                    break


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
                           np.ndarray settlement_cells):
        cdef int x, y, id
        cdef list toReturn

        toReturn = []
        for x in range(max(xPos - radius, 0), min(xPos + radius, width)):
            for y in range(max(yPos - radius, 0), min(yPos + radius, height)):

                id = discreteGridPosToID(x, y, width)

                if owned_cells[id] == -1 and settlement_cells[id] == -1:
                    toReturn.append(id)

        return toReturn

    @staticmethod
    def generateBorderCells(int xPos, int yPos, int width, int height, int radius):

        cdef int x, y
        cdef list toReturn = []

        cdef int x_min = max(xPos - radius, 0)
        cdef int x_max = min(xPos + radius, width - 1)
        cdef int y_min = max(yPos - radius, 0)
        cdef int y_max = min(yPos + radius, height - 1)

        # Top Row
        toReturn += [discreteGridPosToID(x , y_min, width) for x in range(x_min, x_max + 1)]

        # Middle Rows
        for y in range(y_min + 1, y_max):
            toReturn += [discreteGridPosToID(x_min, y, width), discreteGridPosToID(x_max, y, width)]

        # Bottom Row
        toReturn += [discreteGridPosToID(x, y_max, width) for x in range(x_min, x_max + 1)]

        return toReturn

    @staticmethod
    def forage(int patch_id, int workers, int consumption_rate, float forage_multiplier, int farms_per_patch, np.ndarray veg_cells) -> float:
        return veg_cells[patch_id] * consumption_rate * forage_multiplier * (workers / farms_per_patch)


    @staticmethod
    def farm(int patch_id, int workers, (int, int) house_pos, (int, int) coords, int max_acquisition_distance,
             int farming_production_rate, int farms_per_patch, np.ndarray veg_cells) -> float:

        # Calculate penalties
        cdef int dst
        cdef float dst_penalty, crop_yield

        dst = max(absolute(coords[0] - house_pos[0]), absolute(coords[1] - house_pos[1]))
        dst_penalty = 1.0 if dst <= max_acquisition_distance else 1.0 / (dst - max_acquisition_distance)

        crop_yield = farming_production_rate * veg_cells[patch_id] * (workers / farms_per_patch)

        return crop_yield * dst_penalty


#######################################################################################################################

cdef class CGlobalEnvironmentCurves:

    @staticmethod
    def linear_lerp(float a, float b, float t):
        return lerp(a, b, t)

    @staticmethod
    cdef float cosine(float t, float frequency):
        return 0.5 * sin( frequency * t) + 0.5

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

        cdef float sum = result[result > 0].sum()
        return result / sum if sum > 0.0 else result * 0.0
        #return (result - result.min()) / (result.max() - result.min())
