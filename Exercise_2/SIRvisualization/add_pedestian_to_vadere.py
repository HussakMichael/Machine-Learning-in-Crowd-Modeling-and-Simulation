"""
This Python-Script adds a Pedestrian to the Rimea Corner test

"""
# authors: Leilani Tam von Burg, Atalay Yirik, Michael Hussak

# general imports
import json
import os


class Attributes:
    """
    Within the Attribute Class, the attributes of a pedestrian can be specified.
    The default initial values are extracted from the vadere source code.
    """
    def __init__(self,
                 id_: int = 1,
                 radius: float = 0.2,
                 density_dependent_speed: bool = False,
                 speed_distribution_mean: float = 1.34,
                 speed_distribution_standard_deviation: float = 0.26,
                 minimum_speed: float = 0.5,
                 maximum_speed: float = 2.2,
                 acceleration: float = 2.0,
                 footstep_history_size: int = 4,
                 search_radius: float = 1.0,
                 walking_direction_calculation: str = "BY_TARGET_CENTER",
                 walking_direction_same_if_angle_less_or_equal: float = 45.0
                 ):
        """

        :param id_: ID of the pedestrian
        :param radius: Radius of the pedestrian
        :param density_dependent_speed: The Density dependent speed of the pedestrian
        :param speed_distribution_mean: The mean speed distribution of the pedestrian
        :param speed_distribution_standard_deviation: The standard deviation of the speed
            distribution of the pedestrian
        :param minimum_speed: The minimum speed of the pedestrian
        :param maximum_speed: The maximum speed of the pedestrian
        :param acceleration: The acceleration of the pedestrian
        :param footstep_history_size: The footstep history size of the pedestrian
        :param search_radius:
        :param walking_direction_calculation: The walking direction calculation further defines, how the direction
            is calculated, default: "BY_TARGET_CENTER", can also be "BY_TARGET_CLOSEST_POINT", "BY_GRADIENT"
        :param walking_direction_same_if_angle_less_or_equal: Defines the angle, when to adjust the walking direction
        """
        self.id_ = id_
        self.radius = radius
        self.density_dependent_speed = density_dependent_speed
        self.speed_distribution_mean = speed_distribution_mean
        self.speed_distribution_standard_deviation = speed_distribution_standard_deviation
        self.minimum_speed = minimum_speed
        self.maximum_speed = maximum_speed
        self.acceleration = acceleration
        self.footstep_history_size = footstep_history_size
        self.search_radius = search_radius
        self.walking_direction_calculation = walking_direction_calculation
        self.walking_direction_same_if_angle_less_or_equal = walking_direction_same_if_angle_less_or_equal

    def to_dict(self) -> dict:
        """
        Function is used to hand over the specified attributes to a json dictionary

        :return: a json dictionary
        """
        json_dict = {
            "id": self.id_,
            "radius": self.radius,
            "densityDependentSpeed": self.density_dependent_speed,
            "speedDistributionMean": self.speed_distribution_mean,
            "speedDistributionStandardDeviation": self.speed_distribution_standard_deviation,
            "minimumSpeed": self.minimum_speed,
            "maximumSpeed": self.maximum_speed,
            "acceleration": self.acceleration,
            "footstepHistorySize": self.footstep_history_size,
            "searchRadius": self.search_radius,
            "walkingDirectionCalculation": self.walking_direction_calculation,
            "walkingDirectionSameIfAngleLessOrEqual": self.walking_direction_same_if_angle_less_or_equal
        }
        return json_dict


class Position:
    """
    This class is used to define the position of a pedestrian
    """
    def __init__(self,
                 x: float,
                 y: float
                 ):
        """

        :param x: X-Coordinate of the Pedestrian
        :param y: Y-Coordinate of the Pedestrian
        """
        self.x = x
        self.y = y

    def to_dict(self) -> dict:
        """
        Function is used to hand over the specified attributes to a json dictionary

        :return: a json dictionary
        """
        json_dict = {
            "x": self.x,
            "y": self.y
        }
        return json_dict


class Velocity:
    """
    This class is used to define the velocity of a pedestrian
    """
    def __init__(self,
                 x: float = 0.0,
                 y: float = 0.0
                 ):
        """

        :param x: X-Velocity of the Pedestrian
        :param y: Y-Velocity of the Pedestrian
        """
        self.x = x
        self.y = y

    def to_dict(self) -> dict:
        """
        Function is used to hand over the specified attributes to a json dictionary

        :return: a json dictionary
        """
        json_dict = {
            "x": self.x,
            "y": self.y
        }
        return json_dict


class Pedestrian:
    """
    The Pedestrian class is builds up the basic structure of a Pedestrians json dict.
    In further steps, the pedestrians attributes, its position and velocity can be defined
    """
    def __init__(self,
                 ):
        self.json_dict = self.to_dict()

    @staticmethod
    def to_dict() -> dict:
        """
        Function is used to hand over the specified attributes to a json dictionary

        :return: a json dictionary
        """
        json_dict = {"attributes": {},
                     "source": None,
                     "targetIds": [],
                     "nextTargetListIndex": 0,
                     "isCurrentTargetAnAgent": False,
                     "position": {},
                     "velocity": {},
                     "freeFlowSpeed": 1.3,
                     "followers": [],
                     "idAsTarget": -1,
                     "isChild": False,
                     "isLikelyInjured": False,
                     "psychologyStatus": {
                         "mostImportantStimulus": None,
                         "threatMemory": {
                             "allThreats": [],
                             "latestThreatUnhandled": False
                         },
                         "selfCategory": "TARGET_ORIENTED",
                         "groupMembership": "OUT_GROUP",
                         "knowledgeBase": {
                             "knowledge": []
                         }
                     },
                     "groupIds": [],
                     "groupSizes": [],
                     "trajectory": {
                         "footSteps": []
                     },
                     "modelPedestrianMap": {},
                     "type": "PEDESTRIAN"
                     }
        return json_dict

    def add(self,
            attributes: Attributes,
            position: Position,
            velocity: Velocity,
            target_ids=None
            ) -> None:
        """
        By calling this function, the attributes, the position and the velocity as well as the target IDs of
        a pedestrian can be specified.
        These variables will be added to the pedestrians json dict.

        :param attributes: specified attributes of the pedestrian
        :param position: specified position of the pedestrian
        :param velocity: specified velocity of the pedestrian
        :param target_ids: specified target IDs of the pedestrian
        :return:
        """

        self.json_dict["position"] = position.to_dict()
        self.json_dict["attributes"] = attributes.to_dict()
        self.json_dict["velocity"] = velocity.to_dict()
        if target_ids is None:
            target_ids = []
        self.json_dict["targetIds"] = target_ids


class Scenario:
    """
    By initializing the Scenario class with the scenario path, the scenario is directly loaded.
    """
    def __init__(self,
                 scenario_path,
                 output_path
                 ):
        """

        :param scenario_path: Path to the scenario to be loaded
        :param output_path: Path, where modified scenario is saved to
        """
        self.output_path = output_path
        self.scenario = self.load_scenario(scenario_path)

    @staticmethod
    def load_scenario(scenario_path) -> json:
        """
        This function is used to load the scenario

        :param scenario_path: Path to the scenario to be loaded
        :return: the scenario as a json dict
        """
        with open(scenario_path) as json_file:
            return json.load(json_file)

    def add_pedestrian_to_scenario(self, pedestrian: Pedestrian):
        """

        :param pedestrian:
        :return:
        """
        # in case pedestrian has no predefined target
        if len(pedestrian.json_dict["targetIds"]) == 0:
            for target in self.scenario["scenario"]["topography"]["targets"]:
                pedestrian.json_dict["targetIds"].append(target["id"])

        # appending pedestrian to dynamic elements of scenario
        self.scenario["scenario"]["topography"]["dynamicElements"].append(pedestrian.json_dict)

    def save_scenario(self):
        # changing name to show that it is the modified scenario
        self.scenario["name"] = self.scenario["name"] + "_modified"
        with open(self.output_path, 'w') as outfile:
            json.dump(self.scenario, outfile)


# loading a scenario
scen = Scenario(scenario_path=r"./scenarios/rimea_06_corner.scenario",
                output_path=r"./scenarios/rimea_06_corner_modified.scenario")

# creating a Pedestrian element with predefined attributes and velocity and a specified position
attr = Attributes()
velo = Velocity()
pos = Position(x=11.5, y=2)
ped = Pedestrian()

ped.add(attributes=attr, position=pos, velocity=velo)

scen.add_pedestrian_to_scenario(pedestrian=ped)
scen.save_scenario()

os.system('java -jar ./vadere/vadere-console.jar scenario-run --scenario-file "./scenarios/rimea_06_corner_modified.scenario" --output-dir="./output"')