from prediction.M2I.predictor import M2IPredictor
import numpy as np
import math
import logging
import copy
import random
import time

from plan.env_planner import EnvPlanner, Agent, SudoInterpolator
import interactive_sim.envs.util as utils
import plan.helper as plan_helper


S0 = 3
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
PRINT_TIMER = False

class EgoLTPlanner(EnvPlanner):
    def reset(self, *args, **kwargs):
        print('ego planner reset')

    def step(self, data_dic):
        agents = data_dic['agent']
        ego = agents['ego']
        ego_pose = agents['ego']['pose']
        ego_pose.append(ego_pose[-1])
        data_dic['agent']['ego']['pose'] = ego_pose
        return data_dic
