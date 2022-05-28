import json
import logging
import math
import numpy as np
import statistics
import time

from ECAgent.Core import *
from ECAgent.Decode import IDecodable
from ECAgent.Environments import PositionComponent, discreteGridPosToID
from ECAgent.Collectors import Collector, FileCollector
from VegetationModel import SoilMoistureComponent, GlobalEnvironmentComponent

from Logging import ILoggable

# Cython Functions
from CythonFunctions import CAgentResourceConsumptionSystemFunctions, CAgentResourceAcquisitionFunctions, CAgentUtilityFunctions


class Individual:
    """ This class stores the id and age of an individual living in a household"""
    def __init__(self, id, age=0):
        self.id = id
        self.age = age


class ResourceComponent(Component):
    """ This component stores the resources and relevant properties of the Household agent"""
    carrying_capacity = 0
    consumption_rate = 0
    child_factor = 0.0
    age_of_maturity = 0
    vision_square = 0
    move_lookback = 3

    def __init__(self, agent, model: Model):
        super().__init__(agent, model)

        self.occupants = {}
        self.occupant_counter = 0
        self.resources = 0
        # Not really a measure of hunger but a measure of how much of the total a family resources requirement was met
        self.hunger = 0
        self.satisfaction = 0.0
        self.ownedLand = []
        self.storage_decay = []
        self.a_workers = 0

    def add_occupant(self, id, age: int = 0):
        self.occupants[id] = Individual(id, age)
        if age >= ResourceComponent.age_of_maturity:
            self.a_workers += 1
        self.occupant_counter += 1

    def move_occupant(self, id, other_household):
        other_household[ResourceComponent].occupants[id] = self.occupants[id]
        self.occupants.pop(id)

    def claim_land(self, id):
        self.ownedLand.append(id)

    def release_land(self, id):
        self.ownedLand.remove(id)

    def able_workers(self):
        """ Returns the number of household occupants that have reached the age of maturity """
        return self.a_workers

    def update_able_workers(self):
        """ updates able workers count """
        self.a_workers = len([o for o in self.occupants if self.occupants[o].age >= ResourceComponent.age_of_maturity])

    def average_age(self):
        return statistics.mean([self.occupants[o].age for o in self.occupants
                                if self.occupants[o].age >= ResourceComponent.age_of_maturity])

    def get_next_id(self):
        return '%d_%d' % (self.agent.id, self.occupant_counter)

    def required_resources(self):
        """ Returns the amount of resources required by the Household """
        occu_length = len(self.occupants)
        return ResourceComponent.consumption_rate * (self.able_workers() + (ResourceComponent.child_factor * (occu_length - self.able_workers())))

    def excess_resources(self):
        """ Returns the amount of excess resources a Household has. """
        return max(0.0, self.resources - self.required_resources())


class HouseholdRelationshipComponent(Component):
    """ This component holds the necessary properties for managing a households relationships with other Households."""
    load_difference = 0

    def __init__(self, agent, model: Model, settlementID : int):
        super().__init__(agent, model)

        self.settlementID = settlementID
        self.load = 0

        # Randomize the resource trading personalities
        self.peer_resource_transfer_chance = self.model.random.random()
        self.sub_resource_transfer_chance = self.model.random.random()

        self.lookback_sids = []

    def is_aquaintance(self, h):
        """Returns true if household h is in the same settlement as household h"""
        return self.settlementID == h[HouseholdRelationshipComponent].settlementID

    def is_auth(self, h):
        """Returns true if household self has an authority relationship over household h"""

        s_status = self.agent.social_status()
        if h is self.agent or s_status <= 0:
            return False

        h_status = h.social_status()
        return self.is_aquaintance(h) and ((s_status - h_status)/max(s_status, h_status)
                                           > HouseholdRelationshipComponent.load_difference)

    def is_sub(self, h):
        """Returns true if household self has a subordinate relationship with household h"""
        # Household self is a subordinate if the household h has an auth relationship
        return h[HouseholdRelationshipComponent].is_auth(self.agent)

    def is_peer(self, h):
        """Returns true if household self has a peer relationship with household h"""

        if h is self.agent or self.is_auth(h):
            return False

        h_status = h.social_status()
        s_status = self.agent.social_status()

        if h_status <= 0 and s_status <= 0:
            return True  # Ensure no division by error

        return self.is_aquaintance(h) and abs(s_status - h_status)/max(s_status, h_status) \
               < HouseholdRelationshipComponent.load_difference


class HouseholdPreferenceComponent(Component):
    """ This component holds all the necessary properties required for Household Utility-based decision making."""
    learning_rate_range = [0.0, 1.0]

    def __init__(self, agent, model: Model, init_preference: float = 0.0):
        super().__init__(agent, model)

        self.forage_utility = init_preference
        self.farm_utility = 0.0
        self.learning_rate = model.random.uniform(HouseholdPreferenceComponent.learning_rate_range[0],
                                                  HouseholdPreferenceComponent.learning_rate_range[1])

        self.prev_hunger = 1.0


class IEComponent(Component):
    """ This component holds all of the necessary properties required for the Cultural Algorithm """

    mutation_rate = 0.05
    conformity_range = [0.0, 1.0]

    b = 1.5
    m = 0.5

    def __init__(self, agent: Agent, model: Model):
        super().__init__(agent, model)

        self.conformity = self.model.random.uniform(IEComponent.conformity_range[0], IEComponent.conformity_range[1])


class Household(Agent, IDecodable):
    """ This is the base household agent class """

    def __init__(self, id: str, model: Model, settlementID: int):
        super().__init__(id, model)

        self.addComponent(ResourceComponent(self, model))
        self.addComponent(HouseholdRelationshipComponent(self, model, settlementID))

    @staticmethod
    def decode(params: dict):

        ResourceComponent.age_of_maturity = params['age_of_maturity']
        ResourceComponent.consumption_rate = params['consumption_rate']
        ResourceComponent.carrying_capacity = params['carrying_capacity']
        ResourceComponent.vision_square = params['vision_square']
        ResourceComponent.child_factor = params['child_factor']

        HouseholdRelationshipComponent.load_difference = params['load_difference']

        agent = Household(params['agent_index'], params['model'], -1)
        for i in range(params['init_occupants']):
            age = agent.model.random.randrange(params['init_age_range'][0], params['init_age_range'][1])
            agent[ResourceComponent].add_occupant(agent[ResourceComponent].get_next_id(), age)
        return agent

    def __str__(self):
        return 'Household {}:\n\tOccupants: {}\n\tResources: {}\n\tLoad: {}'.format(
            self.id, len(self[ResourceComponent].occupants), self[ResourceComponent].resources, self[HouseholdRelationshipComponent].load)

    def social_status(self):
        """ Returns the agents social status which is the sum of its resources and load. """
        return self[ResourceComponent].resources + self[HouseholdRelationshipComponent].load

    def jsonify(self) -> dict:
        created_dict = {}
        created_dict['id'] = self.id

        created_dict['occupants'] = len(self[ResourceComponent].occupants)
        created_dict['able_workers'] = self[ResourceComponent].able_workers()
        created_dict['resources'] = self[ResourceComponent].resources
        created_dict['hunger'] = self[ResourceComponent].hunger
        created_dict['satisfaction'] = self[ResourceComponent].satisfaction
        created_dict['owned_land'] = self[ResourceComponent].ownedLand

        created_dict['settlement_id'] = self[HouseholdRelationshipComponent].settlementID
        created_dict['load'] = self[HouseholdRelationshipComponent].load
        created_dict['peer_chance'] = self[HouseholdRelationshipComponent].peer_resource_transfer_chance
        created_dict['sub_chance'] = self[HouseholdRelationshipComponent].sub_resource_transfer_chance

        return created_dict


class PreferenceHousehold(Household):
    """This is the base class for giving a Household Utility-based Decision Making """
    def __init__(self, id: str, model: Model, settlementID: int, init_preference):
        super().__init__(id, model, settlementID)

        self.addComponent(HouseholdPreferenceComponent(self, model, init_preference))

    @staticmethod
    def decode(params: dict):
        ResourceComponent.age_of_maturity = params['age_of_maturity']
        ResourceComponent.consumption_rate = params['consumption_rate']
        ResourceComponent.carrying_capacity = params['carrying_capacity']
        ResourceComponent.vision_square = params['vision_square']
        ResourceComponent.child_factor = params['child_factor']

        HouseholdRelationshipComponent.load_difference = params['load_difference']

        HouseholdPreferenceComponent.learning_rate_range = params['learning_rate_range']

        agent = PreferenceHousehold(params['agent_index'], params['model'], -1, params['init_preference'])
        for i in range(params['init_occupants']):
            age = agent.model.random.randrange(params['init_age_range'][0], params['init_age_range'][1])
            agent[ResourceComponent].add_occupant(agent[ResourceComponent].get_next_id(), age)
        return agent

    def __str__(self):
        return super().__str__() + '\n\tFarming Preference(f/F): ({}, {})'.format(
            self[HouseholdPreferenceComponent].forage_utility, self[HouseholdPreferenceComponent].farm_utility)

    def jsonify(self) -> dict:
        created_dict = super().jsonify()

        created_dict['forage_utility'] = self[HouseholdPreferenceComponent].forage_utility
        created_dict['farm_utility'] = self[HouseholdPreferenceComponent].farm_utility
        created_dict['prev_hunger'] = self[HouseholdPreferenceComponent].prev_hunger
        created_dict['learning_rate'] = self[HouseholdPreferenceComponent].learning_rate

        return created_dict


class IEHousehold(PreferenceHousehold):
    """ This is the Household class for a Utility-based Household that exchanges genetic information using a
        Genetic Algorithm and Cultural Algorithm. """
    def __init__(self, id: str, model: Model, settlementID: int, init_preference):
        super().__init__(id, model, settlementID, init_preference)

        self.addComponent(IEComponent(self, self.model))

    @staticmethod
    def decode(params: dict):
        ResourceComponent.age_of_maturity = params['age_of_maturity']
        ResourceComponent.consumption_rate = params['consumption_rate']
        ResourceComponent.carrying_capacity = params['carrying_capacity']
        ResourceComponent.vision_square = params['vision_square']
        ResourceComponent.child_factor = params['child_factor']

        HouseholdRelationshipComponent.load_difference = params['load_difference']

        HouseholdPreferenceComponent.learning_rate_range = params['learning_rate_range']
        IEComponent.conformity_range = params['conformity_range']
        IEComponent.b = params['b']
        IEComponent.m = params['m']
        IEComponent.mutation_rate = params['mutation_rate']

        agent = IEHousehold(params['agent_index'], params['model'], -1, params['init_preference'])
        for i in range(params['init_occupants']):
            age = agent.model.random.randrange(params['init_age_range'][0], params['init_age_range'][1])
            agent[ResourceComponent].add_occupant(agent[ResourceComponent].get_next_id(), age)
        return agent

    def jsonify(self) -> dict:
        created_dict = super().jsonify()

        created_dict['conformity'] = self[IEComponent].conformity

        return created_dict


class Settlement:
    """ This is a simple container class for keeping track of which households belong to which settlements. """
    def __init__(self, id):
        self.id = id
        self.pos = []
        self.occupants = []

    def jsonify(self):
        return {
            'id': self.id,
            'pos': self.pos,
            'occupants': self.occupants
        }


class SettlementRelationshipComponent(Component):
    """ This is an environment layer component that allows the model to track of all of the settlements currently
    occupying the environment """
    def __init__(self, agent: Agent, model: Model, yrs_per_move, cell_capacity):
        super().__init__(agent, model)
        self.settlement_count = 0
        self.settlements = {}

        self.yrs_per_move = yrs_per_move
        self.cell_capacity = cell_capacity

    def create_settlement(self) -> int:
        self.settlements[self.settlement_count] = Settlement(self.settlement_count)
        id = self.settlement_count
        self.settlement_count += 1
        return id

    def remove_settlement(self, settlementID):
        # Remove all settlement spots
        for unq_id in self.settlements[settlementID].pos:
            self.model.environment.cells.at[unq_id, 'isSettlement'] = -1
        self.settlements.pop(settlementID)

    def add_household_to_settlement(self, household: Household, settlementID):

        if len(self.settlements[settlementID].occupants) > 0 and len(self.settlements[settlementID].occupants) % self.cell_capacity == 0:
            # Acquire more land from a neighbouring cell
            plot_choices = []
            for coords in [self.model.environment.cells['pos'][x] for x in self.settlements[settlementID].pos]:
                plot_choices += [p for p in self.model.environment.getNeighbours(coords)
                                 if p not in plot_choices and self.model.environment.cells['isSettlement'][p] == -1]

            choice = self.model.random.choice(plot_choices)
            self.settlements[settlementID].pos.append(choice)
            self.model.environment.cells.at[choice, 'isSettlement'] = settlementID

            # Remove house from owners list if settlement expands
            if self.model.environment.cells['isOwned'][choice] != -1:
                rem_h = self.model.environment.getAgent(self.model.environment.cells['isOwned'][choice])
                rem_h[ResourceComponent].release_land(choice)
                self.model.environment.cells.at[choice, 'isOwned'] = -1

        self.settlements[settlementID].occupants.append(household.id)
        household[HouseholdRelationshipComponent].settlementID = settlementID

    def remove_household(self, household: Household):
        sID = household[HouseholdRelationshipComponent].settlementID
        self.settlements[sID].occupants.remove(household.id)

        # Purge Settlement if it doesn't exist
        if len(self.settlements[sID].occupants) == 0:
            self.remove_settlement(sID)

        household[HouseholdRelationshipComponent].settlementID = -1

    def move_household(self, household: Household, new_settlementID):
        self.remove_household(household)
        self.add_household_to_settlement(household, new_settlementID)

    def merge_settlements(self, s1, s2):
        """Merges Settlement S2 into Settlement S1"""

        # Move all houses
        for houseID in self.settlements[s2].occupants:
            self.move_household(self.model.environment.getAgent(houseID), s1)

        # Delete Settlement s2
        self.remove_settlement(s2)

    def getSettlementWealth(self, settlementID):
        return sum([self.model.environment.getAgent(h)[ResourceComponent].resources for h in self.settlements[settlementID].occupants])

    def getSettlementExcessWealth(self, settlementID):
        return sum([self.model.environment.getAgent(h)[ResourceComponent].excess_resources()
                    for h in self.settlements[settlementID].occupants])

    def getSettlementLoad(self, settlementID):
        return sum([self.model.environment.getAgent(h)[HouseholdRelationshipComponent].load for h in
                    self.settlements[settlementID].occupants])

    def getSettlementSocialStatus(self, settlementID):
        return sum([self.model.environment.getAgent(h).social_status() for h in
                    self.settlements[settlementID].occupants])

    def getAverageSettlementSocialStatus(self, settlementID):
        return self.getSettlementSocialStatus(settlementID) / len(self.settlements[settlementID].occupants)

    def getSettlementPopulation(self, settlementID):
        return sum([len(self.model.environment.getAgent(h)[ResourceComponent].occupants) for h in
                    self.settlements[settlementID].occupants])

    def getSettlementFarmUtility(self, settlementID):
        return sum([self.model.environment.getAgent(h)[HouseholdPreferenceComponent].farm_utility for h in
                    self.settlements[settlementID].occupants
                    if self.model.environment.getAgent(h).hasComponent(HouseholdPreferenceComponent)]
                   ) / len(self.settlements[settlementID].occupants)

    def getSettlementForageUtility(self, settlementID):
        return sum([self.model.environment.getAgent(h)[HouseholdPreferenceComponent].forage_utility for h in
                    self.settlements[settlementID].occupants
                    if self.model.environment.getAgent(h).hasComponent(HouseholdPreferenceComponent)]
                   ) / len(self.settlements[settlementID].occupants)

    def getAverageSettlementWealth(self, settlementID):
        return self.getSettlementWealth(settlementID) / len(self.settlements[settlementID].occupants)

    def getEmptySettlementNeighbours(self, settlementID, target_count, start_count=1, exclude=[]):
        """ Returns a list of grid positions that neighbour the settlement and are unoccupied. """
        emptyCells = []
        count = start_count

        while len(emptyCells) < target_count and count <= self.model.systemManager.systems['RAS'].max_acquisition_distance:
            coords = self.model.environment.cells['pos'][self.settlements[settlementID].pos[0]]

            candidate_cells = CAgentResourceAcquisitionFunctions.generateBorderCells(coords[0], coords[1],
                                                                                     self.model.environment.width, self.model.environment.height, count)
            emptyCells += [p for p in candidate_cells
                           if p not in emptyCells and self.model.environment.cells['isSettlement'][p] == -1
                           and self.model.environment.cells['isOwned'][p] == -1
                           and p not in exclude]

            count += 1
        return emptyCells, count

    def get_all_auth(self, h: Household):
        """Returns all of the households which have an auth relationship over household h"""
        return [self.model.environment.getAgent(x) for x in self.settlements[h[HouseholdRelationshipComponent].settlementID].occupants
                if self.model.environment.getAgent(x)[HouseholdRelationshipComponent].is_auth(h)]

    def get_all_auth_with_excess(self, h: Household):
        """Returns all of the households which have an auth relationship over household h and have excess resources"""
        return [self.model.environment.getAgent(x) for x in self.settlements[h[HouseholdRelationshipComponent].settlementID].occupants
                if self.model.environment.getAgent(x)[HouseholdRelationshipComponent].is_auth(h)
                and self.model.environment.getAgent(x)[ResourceComponent].excess_resources() > 0]

    def get_all_sub(self, h: Household):
        """Returns all households which have a subordinate relationship over household h"""
        return [self.model.environment.getAgent(x) for x in self.settlements[h[HouseholdRelationshipComponent].settlementID].occupants
                if self.model.environment.getAgent(x)[HouseholdRelationshipComponent].is_sub(h)]

    def get_all_peer(self, h: Household):
        """Returns all of the households which have a peer relationship over household h"""
        return [self.model.environment.getAgent(x) for x in
                self.settlements[h[HouseholdRelationshipComponent].settlementID].occupants
                if self.model.environment.getAgent(x)[HouseholdRelationshipComponent].is_peer(h)]

    def get_all_peer_with_excess(self, h: Household):
        """Returns all of the households which have a peer relationship over household h and excess resources"""
        return [self.model.environment.getAgent(x) for x in
                self.settlements[h[HouseholdRelationshipComponent].settlementID].occupants
                if self.model.environment.getAgent(x)[HouseholdRelationshipComponent].is_peer(h)
                and self.model.environment.getAgent(x)[ResourceComponent].excess_resources() > 0]

    def create_belief_space(self, sID):
        """ Creates a belief space """
        # Get Weights
        households = [self.model.environment.getAgent(h) for h in self.settlements[sID].occupants
                      if self.model.environment.getAgent(h).hasComponent(HouseholdPreferenceComponent)]
        ws = np.array([h.social_status() for h in households])

        forage, farm, learning_rate, conformity, peer, sub = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        if AgentIEAdaptationSystem.influence_type == 'BST':
            i = np.argmax(ws)
            forage = households[i][HouseholdPreferenceComponent].forage_utility
            farm = households[i][HouseholdPreferenceComponent].farm_utility
            learning_rate = households[i][HouseholdPreferenceComponent].learning_rate
            conformity = households[i][IEComponent].conformity
            peer = households[i][HouseholdRelationshipComponent].peer_resource_transfer_chance
            sub = households[i][HouseholdRelationshipComponent].sub_resource_transfer_chance
        else:
            if ws.sum() == 0.0:
                ws = np.ones((len(households)))  # If the village has no one with any social status, they are all equal
                ws = ws / len(households)
            else:
                ws = ws / ws.sum()

            for i in range(len(households)):
                forage += households[i][HouseholdPreferenceComponent].forage_utility * ws[i]
                farm += households[i][HouseholdPreferenceComponent].farm_utility * ws[i]
                learning_rate += households[i][HouseholdPreferenceComponent].learning_rate * ws[i]
                conformity += households[i][IEComponent].conformity * ws[i]
                peer += households[i][HouseholdRelationshipComponent].peer_resource_transfer_chance * ws[i]
                sub += households[i][HouseholdRelationshipComponent].sub_resource_transfer_chance * ws[i]

        return BeliefSpace(forage, farm, learning_rate, conformity, peer, sub)

    def get_learning_rate_std(self, sID):
        """Returns the Standard Deviation of the House Learning Rate for all Households that are in settlement sID """
        if len(self.settlements[sID].occupants) < 2:
            return 0.0
        return np.std(
            [self.model.environment.getAgent(h)[HouseholdPreferenceComponent].learning_rate
             for h in self.settlements[sID].occupants
             if self.model.environment.getAgent(h).hasComponent(HouseholdPreferenceComponent)]
        )

    def get_forage_utility_std(self, sID):
        """Returns the Standard Deviation of the House Forage Utility for all Households that are in settlement sID """
        vals = [max(self.model.environment.getAgent(h)[HouseholdPreferenceComponent].forage_utility, 0.0)
                for h in self.settlements[sID].occupants
                if self.model.environment.getAgent(h).hasComponent(HouseholdPreferenceComponent)]

        return np.std(vals) if sum(vals) > 0.0 and len(vals) > 1 else 0.0

    def get_farm_utility_std(self, sID):
        """Returns the Standard Deviation of the House Farm Utility for all Households that are in settlement sID """
        vals = [max(self.model.environment.getAgent(h)[HouseholdPreferenceComponent].farm_utility, 0.0)
                for h in self.settlements[sID].occupants
                if self.model.environment.getAgent(h).hasComponent(HouseholdPreferenceComponent)]

        return np.std(vals) if sum(vals) > 0.0 and len(vals) > 1 else 0.0

    def get_peer_transfer_std(self, sID):
        """Returns the Standard Deviation of the House Peer Resource Transfer Chance for all Households
        that are in settlement sID """
        if len(self.settlements[sID].occupants) < 2:
            return 0.0
        return np.std(
            [self.model.environment.getAgent(h)[HouseholdRelationshipComponent].peer_resource_transfer_chance
             for h in self.settlements[sID].occupants
             if self.model.environment.getAgent(h).hasComponent(HouseholdRelationshipComponent)]
        )

    def get_sub_transfer_std(self, sID):
        """Returns the Standard Deviation of the House Sub Resource Transfer Chance for all Households
        that are in settlement sID """
        if len(self.settlements[sID].occupants) < 2:
            return 0.0
        return np.std(
            [self.model.environment.getAgent(h)[HouseholdRelationshipComponent].sub_resource_transfer_chance
             for h in self.settlements[sID].occupants
             if self.model.environment.getAgent(h).hasComponent(HouseholdRelationshipComponent)]
        )

    def get_conformity_std(self, sID):
        """Returns the Standard Deviation of the House Conformity for all Households that are in settlement sID """

        if len(self.settlements[sID].occupants) < 2:
            return 0.0

        return np.std(
            [self.model.environment.getAgent(h)[IEComponent].conformity
             for h in self.settlements[sID].occupants
             if self.model.environment.getAgent(h).hasComponent(IEComponent)]
        )


class AgentResourceAcquisitionSystem(System, IDecodable, ILoggable):
    """ This system is responsible for executing the Agent Resource Acquisition process. """
    farms_per_patch = 0
    land_buffer = 0
    max_acquisition_distance = 0

    moisture_consumption_rate = 0
    crop_gestation_period = 0
    farming_production_rate = 0
    forage_consumption_rate = 0
    forage_production_multiplier = 0.0

    storage_yrs = 1

    @staticmethod
    def decode(params: dict):
        AgentResourceAcquisitionSystem.farms_per_patch = params['farms_per_patch']
        AgentResourceAcquisitionSystem.land_buffer = params['land_buffer']
        AgentResourceAcquisitionSystem.max_acquisition_distance = params['max_acquisition_distance']
        AgentResourceAcquisitionSystem.moisture_consumption_rate = params['moisture_consumption_rate']
        AgentResourceAcquisitionSystem.crop_gestation_period = params['crop_gestation_period']
        AgentResourceAcquisitionSystem.farming_production_rate = params['farming_production_rate']
        AgentResourceAcquisitionSystem.forage_consumption_rate = params['forage_consumption_rate']
        AgentResourceAcquisitionSystem.forage_production_multiplier = params['forage_production_multiplier']
        AgentResourceAcquisitionSystem.storage_yrs = params['storage_yrs']
        return AgentResourceAcquisitionSystem(params['id'], params['model'], params['priority'])

    def __init__(self, id: str, model: Model,priority):

        System.__init__(self, id, model, priority=priority)
        IDecodable.__init__(self)
        ILoggable.__init__(self, 'model.RAS')

        def owned_generator(pos, cells):
            return -1
        # And isOwned environment layer
        model.environment.addCellComponent('isOwned', owned_generator)

    def acquire_land(self, household: Household, target: int, last_neighbour_count):
        new_land = int(target * AgentResourceAcquisitionSystem.land_buffer) - len(household[ResourceComponent].ownedLand)

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
        vegetation_cells = self.model.environment.cells['vegetation'].to_numpy()
        moisture_cells = self.model.environment.cells['moisture'].to_numpy()
        height_cells = self.model.environment.cells['height'].to_numpy()
        position_cells = self.model.environment.cells['pos']  # Not passed to Cython so it doesn't need to be np.array
        slope_cells = self.model.environment.cells['slope'].to_numpy()

        # This is just for dynamic programming purposes
        settlement_forage_land = {}
        settlement_neighbour_count = {}

        is_land_available = True
        if len(self.model.environment.getAgents()) > 0.9 * (self.model.environment.width * self.model.environment.height):
            is_land_available = len([0 for x in range(len(owned_cells)) if owned_cells[x] == -1
                                     and settlement_cells[x] == -1]) > 0

        log_string = ''

        def getVegetation(location):  # Function used to sort land patches by vegetation density.
            return vegetation_cells[location]

        for household in self.model.environment.getAgents():
            # Get Settlement ID
            sID = household[HouseholdRelationshipComponent].settlementID

            # Decay storage_yrs harvest
            if len(household[ResourceComponent].storage_decay) == AgentResourceAcquisitionSystem.storage_yrs:
                household[ResourceComponent].resources -= household[ResourceComponent].storage_decay.pop()

            # Determine how many patches a household can farm
            able_workers = household[ResourceComponent].able_workers()
            max_farm = math.ceil(able_workers / AgentResourceAcquisitionSystem.farms_per_patch)

            farm_threshold = household[HouseholdPreferenceComponent].prev_hunger + (1.0 * self.model.systemManager.timestep / self.model.iterations)

            numToFarm = CAgentResourceAcquisitionFunctions.num_to_farm_phouse(farm_threshold, max_farm, self.model.random,
                                                                              household[HouseholdPreferenceComponent].farm_utility, household[HouseholdPreferenceComponent].forage_utility)


            numToForage = max_farm - numToFarm
            hPos = (household[PositionComponent].x, household[PositionComponent].y)

            # If ownedLand < patches to farm allocate more land to farm
            if len(household[ResourceComponent].ownedLand) < numToFarm * AgentResourceAcquisitionSystem.land_buffer:
                if is_land_available:
                    settlement_neighbour_count[sID] = self.acquire_land(household, numToFarm,
                                                                    settlement_neighbour_count[sID] if sID in settlement_neighbour_count else 1)

                if len(household[ResourceComponent].ownedLand) < numToFarm:
                    numToForage += (numToFarm - len(household[ResourceComponent].ownedLand))
                    numToFarm -= len(household[ResourceComponent].ownedLand)

            # Farm numToFarm Cells
            totalFarm = 0
            if numToFarm > 0:

                # Select land patches
                farmableLand = [x for x in household[ResourceComponent].ownedLand]

                for i in range(numToFarm):

                    worker_diff = max(able_workers - AgentResourceAcquisitionSystem.farms_per_patch, 0)
                    workers = able_workers - worker_diff
                    able_workers = worker_diff

                    if len(farmableLand) == 0:
                        break

                    # Remove patches of land randomly
                    patchID = farmableLand.pop(self.model.random.randrange(0, len(farmableLand)))
                    new_resources = CAgentResourceAcquisitionFunctions.farm(patchID, workers, hPos, position_cells[patchID],
                                                                            sum(self.model.environment[GlobalEnvironmentComponent].temp) / 12.0,
                                                                            AgentResourceAcquisitionSystem.max_acquisition_distance,
                                                                            AgentResourceAcquisitionSystem.moisture_consumption_rate,
                                                                            AgentResourceAcquisitionSystem.crop_gestation_period,
                                                                            AgentResourceAcquisitionSystem.farming_production_rate,
                                                                            AgentResourceAcquisitionSystem.farms_per_patch,
                                                                            height_cells, moisture_cells, slope_cells,
                                                                            self.model.environment[SoilMoistureComponent],
                                                                            self.model.environment[GlobalEnvironmentComponent],
                                                                            self.model.random)

                    household[ResourceComponent].resources += new_resources
                    totalFarm += new_resources

                    log_string += 'HOUSEHOLD.FARM: {} {} {}\n'.format(household.id, patchID, new_resources)

            # Forage numToForage Cells
            totalForage = 0
            if numToForage > 0 and is_land_available:
                if sID not in settlement_forage_land:
                    settlement_forage_land[sID] = CAgentResourceAcquisitionFunctions.generateNeighbours(hPos[0], hPos[1],
                                                                                                        self.model.environment.width, self.model.environment.height,
                                                                                                        AgentResourceAcquisitionSystem.max_acquisition_distance, owned_cells,
                                                                                                        settlement_cells)

                foragableLand = settlement_forage_land[sID]

                if len(foragableLand) < numToForage:
                    numToForage = len(foragableLand)

                # Sort Foraging land by how much vegetation it has
                foragableLand.sort(key=getVegetation, reverse=True)

                # Forage these patches
                for iForage in range(numToForage):
                    worker_diff = max(able_workers - AgentResourceAcquisitionSystem.farms_per_patch, 0)
                    workers = able_workers - worker_diff
                    able_workers = worker_diff

                    new_resources = CAgentResourceAcquisitionFunctions.forage(foragableLand[iForage], workers,
                                                                              vegetation_cells, AgentResourceAcquisitionSystem.forage_consumption_rate,
                                                                              AgentResourceAcquisitionSystem.forage_production_multiplier,
                                                                              AgentResourceAcquisitionSystem.farms_per_patch)

                    # Update Household Resources
                    household[ResourceComponent].resources += new_resources
                    totalForage += new_resources

                    log_string += 'HOUSEHOLD.FORAGE: {} {} {}\n'.format(household.id, foragableLand[iForage], new_resources)

            # Add newly acquired resources to decay storage to keep track of them.
            household[ResourceComponent].storage_decay.append(totalFarm + totalForage)

            # Adjust the farm_preference based on number of resources acquired
            AgentResourceAcquisitionSystem.adjust_farm_preference(household, totalForage + totalFarm,
                                                                  totalFarm/numToFarm if numToFarm != 0 else 0.0, totalForage/numToForage if numToForage != 0 else 0.0)

        # Update Environment Dataframe
        self.model.environment.cells.update({'vegetation': vegetation_cells, 'moisture': moisture_cells,
                                             'isOwned': owned_cells})
        # Log Events
        self.logger.info(log_string)
        self.logger.info('SYS.TIME: {} {}\n'.format(self.id, time.time() - start_time))

    @staticmethod
    def adjust_farm_preference(household: PreferenceHousehold, acquired_resources, farm_res_avg, forage_res_avg):
        # Set new prev_hunger
        household[HouseholdPreferenceComponent].prev_hunger = acquired_resources / household[ResourceComponent].required_resources()

        # Adjust forage utility
        household[HouseholdPreferenceComponent].forage_utility += household[HouseholdPreferenceComponent].learning_rate * (
                forage_res_avg - household[HouseholdPreferenceComponent].forage_utility
        )
        # Adjust Farm utility
        household[HouseholdPreferenceComponent].farm_utility += household[HouseholdPreferenceComponent].learning_rate * (
                farm_res_avg - household[HouseholdPreferenceComponent].farm_utility
        )


class AgentResourceTransferSystem(System, IDecodable, ILoggable):
    """ This System is Responsible for Handling the resource transfer requests of hungry Households """
    def __init__(self, id: str, model: Model, priority, load_decay):
        System.__init__(self, id, model, priority=priority)
        IDecodable.__init__(self)
        ILoggable.__init__(self, 'model.ARTS')

        self.load_decay = load_decay

    @staticmethod
    def decode(params: dict):
        return AgentResourceTransferSystem(params['id'], params['model'], params['priority'], params['load_decay'])

    def execute(self):
        start_time = time.time()
        log_string = ''

        # Decay Load
        for h in self.model.environment.getAgents():
            h[HouseholdRelationshipComponent].load *= self.load_decay

        sr_comp = self.model.environment[SettlementRelationshipComponent]

        # Precalc excess wealth
        excess_wealth = {}
        for sid in sr_comp.settlements:
            excess_wealth[sid] = sr_comp.getSettlementExcessWealth(sid)

        # For each settlement with excess wealth
        for settlement in [sr_comp.settlements[s] for s in sr_comp.settlements if excess_wealth[s] > 0]:

            # Separate the households into two lists (the have's and have-not's)
            excess_households = []
            peer_households = []
            sub_households = []
            poor_households = []

            for household in [self.model.environment.getAgent(h) for h in settlement.occupants]:
                if household[ResourceComponent].resources < household[ResourceComponent].required_resources():
                    poor_households.append(household)
                else:
                    # Here we check if an agent is willing to grant resource transfer requests to subordinates or peers
                    excess_households.append(household)
                    # Sub Check
                    if self.model.random.random() < household[HouseholdRelationshipComponent].sub_resource_transfer_chance:
                        sub_households.append(household)
                    # Peer Check
                    if self.model.random.random() < household[HouseholdRelationshipComponent].peer_resource_transfer_chance:
                        peer_households.append(household)

            for household in poor_households:
                resources_needed = household[ResourceComponent].required_resources() - household[ResourceComponent].resources

                # Get auth relationships as primary providers
                providers = [auth for auth in sub_households if auth[HouseholdRelationshipComponent].is_auth(household)]

                # Get required resources
                # Get help from superiors randomly
                while len(providers) != 0 and resources_needed > 0:
                    provider = self.model.random.choice(providers)
                    providers.remove(provider)
                    resource_given = AgentResourceTransferSystem.ask_for_resources(provider, resources_needed)
                    household[ResourceComponent].resources += resource_given

                    if resource_given > 0:
                        log_string += 'HOUSEHOLD.RESOURCES.TRANSFER.AUTH: {} {} {}\n'.format(
                            household.id, provider.id, resource_given)
                    else:  # Household has not more resources left so we remove it to save on computation time later
                        excess_households.remove(provider)
                        sub_households.remove(provider)
                        if provider in peer_households:
                            peer_households.remove(provider)

                    # Update amount of resources needed
                    resources_needed -= resource_given

                # Note: If an agent has no load and still requires resources, that means that it's peers
                # and subordinates (of which there will be none) will also have no resources to give and, therefore,
                # we won't have to check any transfer requests.
                if resources_needed > 0 and household[HouseholdRelationshipComponent].load > 0:
                    # Get peers as secondary providers
                    providers = [peer for peer in peer_households if peer[HouseholdRelationshipComponent].is_peer(household)]

                    while len(providers) != 0 and resources_needed > 0:
                        provider = self.model.random.choice(providers)
                        providers.remove(provider)
                        resource_given = AgentResourceTransferSystem.ask_for_resources(provider, resources_needed)
                        household[ResourceComponent].resources += resource_given

                        if resource_given > 0:
                            log_string += 'HOUSEHOLD.RESOURCES.TRANSFER.PEER: {} {} {}\n'.format(
                                household.id, provider.id, resource_given)
                        else:  # Household has not more resources left so we remove it to save on computation time later
                            excess_households.remove(provider)
                            peer_households.remove(provider)

                            if provider in sub_households:
                                sub_households.remove(provider)

                        # Update amount of resources needed
                        resources_needed -= resource_given

                if resources_needed > 0 and household[HouseholdRelationshipComponent].load > 0:
                    # Get subordinates as tertiary providers
                    providers = [sub for sub in excess_households if sub[HouseholdRelationshipComponent].is_sub(household)]

                    while len(providers) != 0 and resources_needed > 0:
                        provider = self.model.random.choice(providers)
                        providers.remove(provider)

                        # Subordinates cannot say no to giving away excess resources

                        resource_given = AgentResourceTransferSystem.ask_for_resources(provider, resources_needed)
                        household[ResourceComponent].resources += resource_given

                        if resource_given > 0:
                            log_string += 'HOUSEHOLD.RESOURCES.TRANSFER.SUB: {} {} {}\n'.format(
                                household.id, provider.id, resource_given)
                        else:  # Household has not more resources left so we remove it to save on computation time later
                            excess_households.remove(provider)

                            if provider in sub_households:
                                sub_households.remove(provider)

                            if provider in peer_households:
                                peer_households.remove(provider)

                        # Update amount of resources needed
                        resources_needed -= resource_given

        self.logger.info(log_string)
        self.logger.info('SYS.TIME: {} {}\n'.format(self.id, time.time() - start_time))

    @staticmethod
    def ask_for_resources(h: Household, amount: int) -> int:

        # Return nothing if h does not have any resources to spare
        excess = h[ResourceComponent].excess_resources()
        if excess == 0:
            return 0

        # If excess does not cover the entire amount asked for
        if excess <= amount:
            h[ResourceComponent].resources -= excess
            h[HouseholdRelationshipComponent].load += excess

            return excess
        else:
            h[ResourceComponent].resources -= amount
            h[HouseholdRelationshipComponent].load += amount

            return amount


class AgentResourceConsumptionSystem(System, IDecodable, ILoggable):
    """ This System is Responsible for a Household's Consumption of Resources """

    def __init__(self, id: str, model: Model,priority):
        System.__init__(self, id, model, priority=priority)
        IDecodable.__init__(self)
        ILoggable.__init__(self, 'model.RCS')

    @staticmethod
    def decode(params: dict):
        return AgentResourceConsumptionSystem(params['id'], params['model'], params['priority'])

    @staticmethod
    def consume(resources, required_resources) -> (int, float):
        # This is actually the inverse of hunger with 1.0 being completely 'full' and zero being 'starving'
        hunger = min(1.0, resources / required_resources)
        remaining_resources = max(0, resources - required_resources)

        return remaining_resources, hunger

    @staticmethod
    def ARCProcess(resComp: ResourceComponent) -> float:

        req_res = resComp.required_resources()
        rem_res, hunger = AgentResourceConsumptionSystem.consume(resComp.resources, req_res)
        resComp.hunger = hunger
        resComp.satisfaction += hunger
        resComp.resources = rem_res

        return req_res * hunger

    def execute(self):
        start_time = time.time()
        to_log = ''
        for stats in [(a.id, CAgentResourceConsumptionSystemFunctions.ARCProcess(a[ResourceComponent]))
                      for a in self.model.environment.getAgents()]:
            to_log += 'HOUSEHOLD.CONSUME: {} {}\n'.format(stats[0], stats[1])

        self.logger.info(to_log)
        self.logger.info('SYS.TIME: {} {}\n'.format(self.id, time.time() - start_time))


class AgentPopulationSystem(System, IDecodable, ILoggable):
    """This system is responsible for managing agent reproduction, death and aging"""

    max_lookback_length = 3

    def __init__(self, id: str, model: Model, priority, birth_rate, death_rate, yrs_per_move, num_settlements,
                 cell_capacity):
        System.__init__(self, id, model, priority=priority)
        IDecodable.__init__(self)
        ILoggable.__init__(self, 'model.APS')

        self.birth_rate = birth_rate
        self.death_rate = death_rate

        self.model.environment.addComponent(SettlementRelationshipComponent(self.model.environment, model, yrs_per_move,
                                                                            cell_capacity))

        self.settlement_move_locs = {}
        self.neighbouring_settlements = {}
        self.new_settlements = []

        self.num_households = 0

        # Add Settlement Cell Map
        def settlement_generator(pos, cells):
            return -1

        # Add is settlement environment layer
        model.environment.addCellComponent('isSettlement', settlement_generator)

        # Create the settlements
        for i in range(num_settlements):
            self.model.environment[SettlementRelationshipComponent].create_settlement()

    def split_household(self, household: Household):

        self.logger.info('HOUSEHOLD.SPLIT: {}'.format(household.id))

        new_household = IEHousehold(self.num_households, self.model, -1, 0.0)

        self.model.environment.addAgent(new_household)
        # Add to settlement
        sID = household[HouseholdRelationshipComponent].settlementID
        self.model.environment[SettlementRelationshipComponent].add_household_to_settlement(new_household, sID)

        # Set household position
        h_pos = self.model.environment.cells['pos'][
            self.model.environment[SettlementRelationshipComponent].settlements[sID].pos[-1]]

        new_household[PositionComponent].x = h_pos[0]
        new_household[PositionComponent].y = h_pos[1]

        # Split Resources
        half_res = household[ResourceComponent].resources / 2.0
        household[ResourceComponent].resources -= half_res
        new_household[ResourceComponent].resources += half_res

        # Copy across hunger and satisfaction values
        new_household[ResourceComponent].hunger = household[ResourceComponent].hunger
        new_household[ResourceComponent].satisfaction = household[ResourceComponent].satisfaction

        # Split land
        num_to_split = len(household[ResourceComponent].ownedLand) // 2

        while num_to_split > 0:
            land_id = self.model.random.choice(
                household[ResourceComponent].ownedLand
            )

            household[ResourceComponent].release_land(land_id)
            new_household[ResourceComponent].claim_land(land_id)

            # Update environment layer
            self.model.environment.cells.at[land_id, 'isOwned'] = new_household.id

            num_to_split -= 1

        # Split Occupants
        able_count = household[ResourceComponent].able_workers()
        child_count = (len(household[ResourceComponent].occupants) - able_count) // 2
        able_count = able_count // 2

        able_individuals = [o for o in household[ResourceComponent].occupants
                            if household[ResourceComponent].occupants[o].age >= ResourceComponent.age_of_maturity]

        children = [o for o in household[ResourceComponent].occupants
                    if o not in able_individuals]

        # Move adults
        while able_count > 0:
            id = self.model.random.choice(
                able_individuals
            )

            household[ResourceComponent].move_occupant(id, new_household)
            able_individuals.remove(id)
            able_count -= 1

        # Move children
        while child_count > 0:
            id = self.model.random.choice(
                children
            )

            household[ResourceComponent].move_occupant(id, new_household)
            children.remove(id)
            child_count -= 1

        sr_comp = self.model.environment[SettlementRelationshipComponent]
        # Determine Parents using neighbouring settlements and households
        # Settlements get a penalty associated with the xtent formula to promote within settlement gene crossover
        parents = [household, self.determine_parent(sID, new_household)]

        # Determine Gene Array
        parent_indices = [self.model.random.randint(0, 1) for _ in range(6)]
        # And if mutation should occur
        mutate_arr = [self.model.random.random() for _ in range(6)]

        # Set Household Resource Trading Personality Types with mutation
        # Peer Resource Transfer
        if mutate_arr[0] > IEComponent.mutation_rate:
            new_household[HouseholdRelationshipComponent].peer_resource_transfer_chance = parents[
                parent_indices[0]][HouseholdRelationshipComponent].peer_resource_transfer_chance
        else:
            new_household[HouseholdRelationshipComponent].peer_resource_transfer_chance = self.model.random.random()
            self.logger.info('MUTATE.HOUSEHOLD.PEER_TRANSFER: {}'.format(new_household.id))

        # Sub Resource Transfer
        if mutate_arr[1] > IEComponent.mutation_rate:
            new_household[HouseholdRelationshipComponent].sub_resource_transfer_chance = parents[
                parent_indices[1]][HouseholdRelationshipComponent].sub_resource_transfer_chance
        else:
            new_household[HouseholdRelationshipComponent].sub_resource_transfer_chance = self.model.random.random()
            self.logger.info('MUTATE.HOUSEHOLD.SUB_TRANSFER: {}'.format(new_household.id))

        # Set Preference
        new_household[HouseholdPreferenceComponent].prev_hunger = household[
            HouseholdPreferenceComponent].prev_hunger

        # Forage Utility
        if mutate_arr[2] > IEComponent.mutation_rate:
            new_household[HouseholdPreferenceComponent].forage_utility = parents[parent_indices[2]][
                HouseholdPreferenceComponent].forage_utility
        else:
            new_household[HouseholdPreferenceComponent].forage_utility = self.model.random.gauss(
                parents[parent_indices[2]][HouseholdPreferenceComponent].forage_utility,
                sr_comp.get_forage_utility_std(sID))

            self.logger.info('MUTATE.HOUSEHOLD.FORAGE_UTILITY: {}'.format(new_household.id))

        # Farm Utility
        if mutate_arr[3] > IEComponent.mutation_rate:
            new_household[HouseholdPreferenceComponent].farm_utility = parents[parent_indices[3]][
                HouseholdPreferenceComponent].farm_utility
        else:
            new_household[HouseholdPreferenceComponent].farm_utility = self.model.random.gauss(
                parents[parent_indices[3]][HouseholdPreferenceComponent].farm_utility,
                sr_comp.get_farm_utility_std(sID))

            self.logger.info('MUTATE.HOUSEHOLD.FARM_UTILITY: {}'.format(new_household.id))

        # Learning Rate
        if mutate_arr[4] > IEComponent.mutation_rate:
            new_household[HouseholdPreferenceComponent].learning_rate = parents[parent_indices[4]][
                HouseholdPreferenceComponent].learning_rate
        else:
            new_household[HouseholdPreferenceComponent].learning_rate = self.model.random.gauss(
                parents[parent_indices[4]][HouseholdPreferenceComponent].learning_rate,
                sr_comp.get_learning_rate_std(sID))

            if new_household[HouseholdPreferenceComponent].learning_rate < 0.01:
                new_household[HouseholdPreferenceComponent].learning_rate = 0.01

            self.logger.info('MUTATE.HOUSEHOLD.STUBBORNESS: {}'.format(new_household.id))

        # Conformity
        if mutate_arr[5] > IEComponent.mutation_rate:
            new_household[IEComponent].conformity = parents[parent_indices[5]][IEComponent].conformity
        else:
            new_household[IEComponent].conformity = self.model.random.gauss(
                parents[parent_indices[5]][IEComponent].conformity,
                sr_comp.get_conformity_std(sID))

            if new_household[IEComponent].conformity < 0.01:
                new_household[IEComponent].conformity = 0.01

            self.logger.info('MUTATE.HOUSEHOLD.CONFORMITY: {}'.format(new_household.id))

        self.num_households += 1

        household[ResourceComponent].update_able_workers()
        new_household[ResourceComponent].update_able_workers()

        self.logger.info('CREATE.HOUSEHOLD: {} {} {}'.format(new_household.id, sID, self.model.environment[
            SettlementRelationshipComponent].settlements[sID].pos[-1]))

    def execute(self):
        start_time = time.time()
        log_string = ''

        self.settlement_move_locs = {}
        self.neighbouring_settlements = {}
        self.new_settlements.clear()

        toRem = []
        for household in self.model.environment.getAgents():

            # Reallocation check
            if self.model.systemManager.timestep != 0 and self.model.systemManager.timestep % self.model.environment[SettlementRelationshipComponent].yrs_per_move == 0:
                if self.model.random.random() > (household[ResourceComponent].satisfaction / (self.model.environment[SettlementRelationshipComponent].yrs_per_move-1)):

                    logging.debug('Moving Household: ' + str(household.id))
                    self.reallocate_agent(household)

                household[ResourceComponent].satisfaction = 0.0  # Reset hunger decision every 'yrs_per_move' steps

            # Birth Chance
            for i in range(household[ResourceComponent].able_workers()):
                if self.model.random.random() <= self.birth_rate:
                    household[ResourceComponent].add_occupant(household[ResourceComponent].get_next_id())
                    log_string += 'HOUSEHOLD.BIRTH: {}\n'.format(household.id)

            # Split household if household reaches capacity
            if len(household[ResourceComponent].occupants) > ResourceComponent.carrying_capacity:
                logging.debug('Splitting Household: ' + str(household.id))
                self.split_household(household)


            # Death Chance
            occuRem = []
            for occupant in household[ResourceComponent].occupants:
                # Random death rate
                if self.model.random.random() * household[ResourceComponent].hunger < self.death_rate:
                    occuRem.append(occupant)
                else:  # If occupant does not die, they age by 1
                    household[ResourceComponent].occupants[occupant].age += 1

            for o in occuRem:
                household[ResourceComponent].occupants.pop(o)
                log_string += 'HOUSEHOLD.DEATH: {}\n'.format(household.id)

            household[ResourceComponent].update_able_workers()

            if household[ResourceComponent].able_workers() == 0:  # Add households to the delete list
                toRem.append(household)

        # Delete empty households here
        for household in toRem:
            # Remove all land from ownership
            self.model.environment.cells.loc[(self.model.environment.cells.isOwned == household.id), 'isOwned'] = -1
            self.model.environment[SettlementRelationshipComponent].remove_household(household)
            # Delete Agent
            self.model.environment.removeAgent(household.id)

            if len(household[ResourceComponent].occupants) > 0:
                log_string += 'REMOVE.HOUSEHOLD.ORPHANED: {}\n'.format(household.id)
            else:
                log_string += 'REMOVE.HOUSEHOLD.EMPTY: {}\n'.format(household.id)

        self.logger.info(log_string)
        self.logger.info('SYS.TIME: {} {}\n'.format(self.id, time.time() - start_time))

    def reallocate_agent(self, household: Household):
        # Get rid of all land ownership that household has
        self.model.environment.cells.loc[(self.model.environment.cells.isOwned == household.id), 'isOwned'] = -1
        household[ResourceComponent].ownedLand.clear()

        old_sid = household[HouseholdRelationshipComponent].settlementID

        if old_sid != -1:
            self.model.environment[SettlementRelationshipComponent].remove_household(household)

        # Add old settlement to agent's lookback memory
        household[HouseholdRelationshipComponent].lookback_sids.append(old_sid)

        # If agent's lookback memory has reached capacity, pop the oldest memory off the list
        if len(household[HouseholdRelationshipComponent].lookback_sids) > AgentPopulationSystem.max_lookback_length:
            household[HouseholdRelationshipComponent].lookback_sids.pop(0)

        if old_sid not in self.model.environment[SettlementRelationshipComponent].settlements:
            self.logger.info('REMOVE.SETTLEMENT.ABANDONED: {}'.format(old_sid))

        if old_sid not in self.neighbouring_settlements:
            self.neighbouring_settlements[old_sid] = []
            for settlement in [self.model.environment[SettlementRelationshipComponent].settlements[s] for s in
                           self.model.environment[SettlementRelationshipComponent].settlements]:
                dist = (household[PositionComponent].x - self.model.environment.cells['pos'][settlement.pos[0]][0]) ** 2 \
                       + (household[PositionComponent].y - self.model.environment.cells['pos'][settlement.pos[0]][
                    1]) ** 2

                if dist < ResourceComponent.vision_square:
                    self.neighbouring_settlements[old_sid].append(settlement)

        # Check for settlement with most wealth (ignoring lookback sids)
        mostWealth = 0
        most_id = -1
        for settlement in [s for s in self.neighbouring_settlements[old_sid]
                           if s.id not in household[HouseholdRelationshipComponent].lookback_sids
                           and s.id in self.model.environment[SettlementRelationshipComponent].settlements]:

            avgWealth = self.model.environment[SettlementRelationshipComponent].getAverageSettlementSocialStatus(settlement.id)
            if avgWealth > mostWealth:
                mostWealth = avgWealth
                most_id = settlement.id

        if mostWealth >= household[ResourceComponent].required_resources():
            # Move to this settlement
            self.model.environment.getComponent(SettlementRelationshipComponent).add_household_to_settlement(household, most_id)
            new_x, new_y = self.model.environment.cells['pos'][self.model.environment[SettlementRelationshipComponent].settlements[most_id].pos[-1]]
            household[PositionComponent].x = new_x
            household[PositionComponent].y = new_y

            self.logger.info('HOUSEHOLD.MOVE.SETTLEMENT: {} {} {} {}'.format(household.id, old_sid, most_id,
                                                                             self.model.environment[SettlementRelationshipComponent].settlements[most_id].pos[-1]))

        else:
            # Assign new household position if agent chooses to not move to an existing settlement
            # This checks to see if someone else from the settlement has moved and, if so, the household follows them,
            # else a new settlement may be created. This creates grouped migration.
            if old_sid not in self.settlement_move_locs:

                # First check to see if household moves a one of the newly created settlements,
                valid_new_settlements = []
                for settlement in [self.model.environment[SettlementRelationshipComponent].settlements[s]
                                   for s in self.new_settlements]:
                    dist = (household[PositionComponent].x - self.model.environment.cells['pos'][settlement.pos[0]][
                        0]) ** 2 \
                           + (household[PositionComponent].y - self.model.environment.cells['pos'][settlement.pos[0]][
                        1]) ** 2

                    if dist < ResourceComponent.vision_square:
                        valid_new_settlements.append(settlement)

                if len(valid_new_settlements) > 0 and self.model.random.random() > 1.0 / len(valid_new_settlements):
                    # Choose one of the new settlements
                    self.settlement_move_locs[old_sid] = self.model.random.choice(valid_new_settlements).pos[0]
                else:
                    # Choose to create a new settlement
                    hPos = (household[PositionComponent].x, household[PositionComponent].y)
                    possible_locs = [x for x in self.model.environment.getNeighbours(hPos,
                                                          radius=int(ResourceComponent.vision_square ** .5))
                                                          if self.model.environment.cells['isOwned'][x] == -1
                                                          and self.model.environment.cells['isSettlement'][x] == -1]

                    if len(possible_locs) > 0:
                        self.settlement_move_locs[old_sid] = self.model.random.choice(possible_locs)
                        new_unq_id = self.settlement_move_locs[old_sid]

                        # Create a new Settlement
                        sttlID = self.model.environment.getComponent(
                            SettlementRelationshipComponent).create_settlement()
                        self.model.environment[SettlementRelationshipComponent].settlements[sttlID].pos.append(
                            new_unq_id)
                        self.new_settlements.append(sttlID)
                        self.logger.info('CREATE.SETTLEMENT: {} {}'.format(sttlID, new_unq_id))
                        self.model.environment.cells.at[new_unq_id, 'isSettlement'] = sttlID
                    else:  # Household will stay put in the event that their is absolutely no where for them to go.
                        self.settlement_move_locs[old_sid] = self.model.environment[
                            SettlementRelationshipComponent].settlements[old_sid].pos[0]

            # Move House and add it to settlement
            new_x, new_y = self.model.environment.cells['pos'][self.settlement_move_locs[old_sid]]
            # Must cast to int() to make it a python object so that it can be JSON serialized
            sttlID = int(self.model.environment.cells['isSettlement'][self.settlement_move_locs[old_sid]])
            household[PositionComponent].x = new_x
            household[PositionComponent].y = new_y
            self.model.environment.getComponent(SettlementRelationshipComponent).add_household_to_settlement(household, sttlID)
            self.logger.info('HOUSEHOLD.MOVE.RANDOM: {} {} {} {}'.format(household.id, old_sid, sttlID, self.settlement_move_locs[old_sid]))

    def determine_parent(self, sID, new_household: IEHousehold):

        # Get all households in settlement sID
        households = [self.model.environment.getAgent(h) for h in self.model.environment[
            SettlementRelationshipComponent].settlements[sID].occupants
                      if self.model.environment.getAgent(h).social_status != 0.0
                      and self.model.environment.getAgent(h) != new_household]

        # Get all Settlements using Xtent
        sr_comp = self.model.environment[SettlementRelationshipComponent]

        pos_series = self.model.environment.cells['pos']

        ss = [s for s in sr_comp.settlements if s != sID]
        ws = np.array([sr_comp.getSettlementSocialStatus(s) for s in ss])

        self_pos = pos_series[sr_comp.settlements[sID].pos[0]]
        pos_ids = [sr_comp.settlements[s].pos[0] for s in ss]
        # Calculate the distances
        ds = np.sqrt(np.array([((pos_series[pos][0] - self_pos[0]) ** 2 + (pos_series[pos][1] - self_pos[1]) ** 2)
                               for pos in pos_ids])) * self.model.cellSize

        xtent_dst = CAgentUtilityFunctions.xtent_distribution(ws, ds, IEComponent.b, IEComponent.m)

        # Add Other Set of Households to possible parent candidates probabilistically
        for i in range(len(xtent_dst)):
            if self.model.random.random() < xtent_dst[i]:
                # Add all households
                households.extend([self.model.environment.getAgent(h)
                                   for h in sr_comp.settlements[ss[i]].occupants])

        # Check to see if all households do not have any resource, if so, randomly choose
        if len(households) == 0:
            return self.model.random.choice([self.model.environment.getAgent(h) for h in self.model.environment[
                SettlementRelationshipComponent].settlements[sID].occupants
                                             if self.model.environment.getAgent(h) != new_household])

        # Get status distribution
        h_status = [h.social_status() for h in households if h.social_status]
        h_status_total = sum(h_status)

        h_weighted_distribution = []

        if h_status_total == 0:
            return self.model.random.choice(households)

        for status in h_status:

            if len(h_weighted_distribution) == 0:
                h_weighted_distribution.append(status / h_status_total)
            else:
                h_weighted_distribution.append(h_weighted_distribution[-1] + status / h_status_total)

        random_val = self.model.random.random()
        for i in range(len(h_weighted_distribution)):

            if h_weighted_distribution[i] < random_val:
                return households[i]

        return households[-1]  # Default to last agent if value is not lower than any of the agents in the distribution

    @staticmethod
    def decode(params: dict):
        return AgentPopulationSystem(params['id'], params['model'], params['priority'], params['birth_rate'],
                                     params['death_rate'], params['yrs_per_move'], params['init_settlements'],
                                     params['cell_capacity'])


class BeliefSpace:
    """ This is a container class for belief space construction and influence """
    def __init__(self, forage_utility, farm_utility, learning_rate, conformity, peer_transfer, sub_transfer):
        self.forage_utility = forage_utility
        self.farm_utility = farm_utility
        self.learning_rate = learning_rate
        self.conformity = conformity
        self.peer_transfer = peer_transfer
        self.sub_transfer = sub_transfer

    def influence(self, bs, dst_penalty: float):
        self.forage_utility += (bs.forage_utility - self.forage_utility) * self.conformity * dst_penalty
        self.farm_utility += (bs.farm_utility - self.farm_utility) * self.conformity * dst_penalty
        self.learning_rate += (bs.learning_rate - self.learning_rate) * self.conformity * dst_penalty
        self.peer_transfer += (bs.peer_transfer - self.peer_transfer) * self.conformity * dst_penalty
        self.sub_transfer += (bs.sub_transfer - self.sub_transfer) * self.conformity * dst_penalty

        # Do Conformity Last so it doesn't affect the other results.
        self.conformity += (bs.conformity - self.conformity) * self.conformity * dst_penalty

    def jsonify(self):
        return {
            'forage_utility': self.forage_utility,
            'farm_utility': self.farm_utility,
            'learning_rate': self.learning_rate,
            'conformity': self.conformity,
            'peer_transfer': self.peer_transfer,
            'sub_transfer': self.sub_transfer
        }

    def duplicate(self):
        return BeliefSpace(self.forage_utility, self.farm_utility, self.learning_rate, self.conformity,
                           self.peer_transfer, self.sub_transfer)


class AgentIEAdaptationSystem(System, IDecodable, ILoggable):
    """ This System Manages the Cultural Algorithm that allows for peer influenced agent adaptation """
    influence_rate = 0.05
    novelty_rate = 0.05
    influence_type = 'AVG'
    persist_belief_space = False

    def __init__(self,id: str, model: Model, priority: int, frequency: int = 1):
        System.__init__(self, id, model, priority=priority, frequency=frequency)
        IDecodable.__init__(self)
        ILoggable.__init__(self, 'model.IEAS')
        self.belief_spaces = {}

    def execute(self):
        start_time = time.time()
        log_string = ''

        belief_spaces = {}

        sr_comp = self.model.environment[SettlementRelationshipComponent]

        # Calculate Belief Space details for each settlement
        settlements = [sr_comp.settlements[s] for s in sr_comp.settlements]
        wealth_dict = {}
        normative_prob = {}
        pos_dict = {}

        for settlement in settlements:
            bs = sr_comp.create_belief_space(settlement.id)
            if AgentIEAdaptationSystem.persist_belief_space and settlement.id in self.belief_spaces:
                self.belief_spaces[settlement.id].influence(bs, 1.0)
                belief_spaces[settlement.id] = self.belief_spaces[settlement.id]
            else:
                belief_spaces[settlement.id] = bs
            wealth_dict[settlement.id] = sr_comp.getSettlementSocialStatus(settlement.id)
            pos_dict[settlement.id] = sr_comp.settlements[settlement.id].pos[0]

        # Influence Each Settlement's belief space using XTENT
        pos_series = self.model.environment.cells['pos']

        # Calculate Influence Belief Space
        influenced_belief_spaces = {}

        for settlement in settlements:
            ss = [s for s in sr_comp.settlements if s != settlement.id]
            ws = np.array([wealth_dict[s] for s in ss])

            self_pos = pos_series[sr_comp.settlements[settlement.id].pos[0]]
            pos_ids = [pos_dict[s] for s in ss]
            # Calculate the distances
            ds = np.sqrt(np.array([((pos_series[pos][0] - self_pos[0]) ** 2 + (pos_series[pos][1] - self_pos[1]) ** 2)
                                   for pos in pos_ids])) * self.model.cellSize

            xtent_dst = CAgentUtilityFunctions.xtent_distribution(ws, ds, IEComponent.b, IEComponent.m)

            rand_val = self.model.random.random()
            rand_count = 0.0
            for index in range(len(xtent_dst)):
                if xtent_dst[index] > 0.0:
                    rand_count += ws[index]
                    if rand_count > rand_val:
                        influenced_belief_spaces[settlement.id] = belief_spaces[ss[index]]
                        normative_prob[settlement.id] = wealth_dict[settlement.id] / ws[index] if ws[index] > 0 else 1.0
                        break

            if settlement.id not in normative_prob:  # If no settlement can influence s, it influences itself with 100%
                normative_prob[settlement.id] = 1.0

        # Influence Each Household using Updated Settlement belief spaces
        for agent in self.model.environment.getAgents():
            sID = agent[HouseholdRelationshipComponent].settlementID
            if self.model.random.random() < AgentIEAdaptationSystem.influence_rate:
                # Choose which belief space is going to influence the agent
                # First we check for novelty
                if self.model.random.random() < IEComponent.mutation_rate:
                    index = self.model.random.randint(0, 6)

                    if index == 0:  # Peer Transfer
                        agent[HouseholdRelationshipComponent].peer_resource_transfer_chance = self.model.random.random()
                    elif index == 1:  # Sub Transfer
                        agent[HouseholdRelationshipComponent].sub_resource_transfer_chance = self.model.random.random()
                    elif index == 2:  # Forage Utility
                        agent[HouseholdPreferenceComponent].forage_utility = self.model.random.gauss(
                            agent[HouseholdPreferenceComponent].forage_utility, sr_comp.get_forage_utility_std(sID))
                    elif index == 3:  # Farm Utility
                        agent[HouseholdPreferenceComponent].farm_utility = self.model.random.gauss(
                            agent[HouseholdPreferenceComponent].farm_utility, sr_comp.get_farm_utility_std(sID))
                    elif index == 4:  # Learning Rate
                        agent[HouseholdPreferenceComponent].learning_rate = self.model.random.gauss(
                            agent[HouseholdPreferenceComponent].learning_rate, sr_comp.get_learning_rate_std(sID))
                    else:  # Conformity
                        agent[IEComponent].conformity = max(0.01, self.model.random.gauss(
                            agent[IEComponent].conformity, sr_comp.get_conformity_std(sID)))

                    log_string += 'HOUSEHOLD.INFLUENCE.DOMAIN: {}\n'.format(agent.id)
                else:
                    # Determine if NORMATIVE or SPATIAL knowledge sources should be used.
                    if self.model.random.random() < normative_prob[sID]:
                        AgentIEAdaptationSystem.influence_agent(agent, belief_spaces[sID])
                        log_string += 'HOUSEHOLD.INFLUENCE.NORMATIVE: {}\n'.format(agent.id)
                    else:
                        AgentIEAdaptationSystem.influence_agent(agent, influenced_belief_spaces[sID])
                        log_string += 'HOUSEHOLD.INFLUENCE.SPATIAL: {}\n'.format(agent.id)

        self.belief_spaces = belief_spaces
        self.logger.info(log_string)
        self.logger.info('SYS.TIME: {} {}\n'.format(self.id, time.time() - start_time))

    @staticmethod
    def influence_agent(agent: IEHousehold, bs: BeliefSpace):

        conformity = agent[IEComponent].conformity

        agent[HouseholdPreferenceComponent].forage_utility += ( bs.forage_utility - agent[HouseholdPreferenceComponent
        ].forage_utility) * conformity

        agent[HouseholdPreferenceComponent].farm_utility += (bs.farm_utility - agent[HouseholdPreferenceComponent
        ].farm_utility) * conformity

        agent[HouseholdPreferenceComponent].learning_rate += (bs.learning_rate - agent[HouseholdPreferenceComponent
        ].learning_rate) * conformity

        agent[IEComponent].conformity += (bs.conformity - agent[IEComponent].conformity) * conformity

        agent[HouseholdRelationshipComponent].peer_resource_transfer_chance += (bs.peer_transfer - agent[
            HouseholdRelationshipComponent].peer_resource_transfer_chance) * conformity
        agent[HouseholdRelationshipComponent].sub_resource_transfer_chance += (bs.sub_transfer - agent[
            HouseholdRelationshipComponent].sub_resource_transfer_chance) * conformity


    @staticmethod
    def decode(params: dict):
        AgentIEAdaptationSystem.influence_rate = params['influence_rate']
        AgentIEAdaptationSystem.influence_type = params['influence_type']
        AgentIEAdaptationSystem.persist_belief_space = params['persist_belief_space']
        return AgentIEAdaptationSystem(params['id'], params['model'], params['priority'],
                                       frequency=params['frequency'] if 'frequency' in params else 1)


class AgentSnapshotCollector(Collector):

    def __init__(self, id: str, model, file_name, frequency=1):
        super().__init__(id, model, frequency=frequency)

        self.file_name = file_name

    def collect(self):

        toWrite = []

        for household in self.model.environment.getAgents():
            toWrite.append(household.jsonify())

        with open(self.file_name + '/iteration_{}.json'.format(self.model.systemManager.timestep), 'w') as outfile:
            json.dump(toWrite, outfile, indent=4)


class SettlementSnapshotCollector(Collector):

    def __init__(self, id: str, model, file_name, frequency: int = 1):
        super().__init__(id, model, frequency=frequency)

        self.file_name = file_name
        self.IEAS = 'IEAS' in self.model.systemManager.systems

    def collect(self):

        toWrite = []

        for sid in self.model.environment[SettlementRelationshipComponent].settlements:
            srComp = self.model.environment[SettlementRelationshipComponent]
            generated_dict = srComp.settlements[sid].jsonify()
            generated_dict['wealth'] = srComp.getSettlementWealth(sid)
            generated_dict['load'] = srComp.getSettlementLoad(sid)
            generated_dict['population'] = srComp.getSettlementPopulation(sid)
            generated_dict['farm_utility'] = srComp.getSettlementFarmUtility(sid)
            generated_dict['forage_utility'] = srComp.getSettlementForageUtility(sid)

            if self.IEAS and sid in self.model.systemManager.systems['IEAS'].belief_spaces:
                generated_dict['belief_space'] = self.model.systemManager.systems['IEAS'].belief_spaces[sid].jsonify()

            toWrite.append(generated_dict)

        with open(self.file_name + '/iteration_{}.json'.format(self.model.systemManager.timestep), 'w') as outfile:
            json.dump(toWrite, outfile, indent=4)
