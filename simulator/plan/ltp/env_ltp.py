from prediction.M2I.predictor import M2IPredictor
import numpy as np
import math
import logging
import copy
import random
import time

import interactive_sim.envs.util as utils
import plan.helper as plan_helper

S0 = 2
T = 0.25 #1.5  # reaction time when following
DELTA = 4  # the power term in IDM
PLANNING_HORIZON = 5  # in frames
PREDICTION_HTZ = 10  # prediction_htz
T_HEADWAY = 0.2
A_SPEEDUP_DESIRE = 0.3  # A
A_SLOWDOWN_DESIRE = 1.5  # B
XPT_SHRESHOLD = 0.7
MINIMAL_DISTANCE_PER_STEP = 0.05
MINIMAL_DISTANCE_TO_TRAVEL = 4
# MINIMAL_DISTANCE_TO_RESCALE = -999 #0.1
REACTION_AFTER = 200  # in frames
MINIMAL_SCALE = 0.3
MAX_DEVIATION_FOR_PREDICTION = 4
TRAFFIC_LIGHT_COLLISION_SIZE = 2

MINIMAL_SPEED_TO_TRACK_ORG_GOAL = 5
MINIMAL_DISTANCE_TO_GOAL = 15

OFF_ROAD_DIST = 30

PRINT_TIMER = False
DRAW_CBC_PTS = False

class EnvLTPlanner:
    def __init__(self):
        self.planning_interval = env_config.env.planning_interval
        self.scenario_frame_number = 0
        self.online_predictor = predictor

        self.current_on_road = True
        # self.data = None
        self.dataset = dataset
        self.online_predictor.dataset = dataset

        self.valid_lane_types = [1, 2] if self.dataset == 'Waymo' else [0, 11]
        self.vehicle_types = [1] if self.dataset == 'Waymo' else [0, 7]  # Waymo: Unset=0, Vehicle=1, Pedestrian=2, Cyclist=3, Other=4

    def reset(self, *args, **kwargs):
        print('env planner reset')

    def step(self, data_dic):
        return data_dic