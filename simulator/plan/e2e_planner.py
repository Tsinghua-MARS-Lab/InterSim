from prediction.M2I.predictor import M2IPredictor
import numpy as np
import math
import logging
import interactive_sim.envs.util as utils
import copy
import random
from plan.env_planner import EnvPlanner, Agent, SudoInterpolator
import time

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
PRINT_TIMER = False

def get_angle(x, y):
    return math.atan2(y, x)

def euclidean_distance(pt1, pt2):
    x_1, y_1 = pt1
    x_2, y_2 = pt2
    return math.sqrt((x_1-x_2)**2+(y_1-y_2)**2)

def get_angle_of_a_line(pt1, pt2):
    # angle from horizon to the right, counter-clockwise,
    x1, y1 = pt1
    x2, y2 = pt2
    angle = math.atan2(y2 - y1, x2 - x1)
    return angle


def calculate_yaw_from_states(trajectory, default_yaw):
    time_frames, _ = trajectory.shape
    pred_yaw = np.zeros([time_frames])
    for i in range(time_frames - 1):
        pose_p = trajectory[i+1]
        pose = trajectory[i]
        delta_x = pose_p[0] - pose[0]
        delta_y = pose_p[1] - pose[1]
        dis = np.sqrt(delta_x*delta_x + delta_y*delta_y)
        if dis > 1:
            angel = get_angle(delta_x, delta_y)
            pred_yaw[i] = angel
            default_yaw = angel
        else:
            pred_yaw[i] = default_yaw
    return pred_yaw


def change_axis(yaw):
    return - yaw - math.pi/2


class E2EPlanner(EnvPlanner):
    """
    BasePlanner should not use the gt future trajectory information for planning.
    Using the gt trajectory for collision avoidance is not safe.
    The future trajectory might be changed after the ego planner's planning by the env planner.
    The BasePlanner has its own predictor which does not share information with the predictor of the EnvPlanner
    The BasePlanner is used to control the ego agent only.
    The BasePlanner is derived from the EnvPlanner, change predefined functions to build/test your own planner.
    """

    def plan_ego(self, current_state, current_frame_idx):
        self.scenario_frame_number = current_frame_idx
        frame_diff = self.scenario_frame_number - self.planning_from
        current_state['predicting']['emergency_stopping'] = False

        if frame_diff >= 0 and frame_diff % self.planning_interval == 0:
            # load scenario data
            if current_state is None:
                print("[E2E Planner]: Invalid State for planner")
                return

            ego_agent_id = current_state['predicting']['ego_id'][1]
            self.online_predictor.marginal_predict(current_frame=self.scenario_frame_number,
                                                   selected_agent_ids=ego_agent_id,
                                                   current_data=current_state)

            total_time_frame = current_state['agent'][ego_agent_id]['pose'].shape[0]
            goal_point = current_state['predicting']['goal_pts'][ego_agent_id]
            my_current_pose = current_state['agent'][ego_agent_id]['pose'][current_frame_idx - 1]
            my_current_v_per_step = euclidean_distance(
                current_state['agent'][ego_agent_id]['pose'][current_frame_idx - 1, :2],
                current_state['agent'][ego_agent_id]['pose'][current_frame_idx - 6, :2]) / 5

            if ego_agent_id in current_state['predicting']['route']:
                current_route = current_state['predicting']['route'][ego_agent_id]
                init_route = False
            else:
                current_route = []
                init_route = True
            my_traj, current_route = self.get_reroute_traj(current_state=current_state,
                                                           agent_id=ego_agent_id,
                                                           current_frame_idx=current_frame_idx,
                                                           dynamic_turnings=True,
                                                           current_route=current_route)
            if init_route:
                current_state['predicting']['route'][ego_agent_id] = current_route
                # check if original end point fits current route
                goal_pt, goal_yaw = self.online_predictor.data['predicting']['goal_pts'][ego_agent_id]

                current_lanes, current_closest_pt_indices, dist_to_lane, distance_threshold = self.find_closes_lane(
                    current_state=current_state,
                    agent_id=ego_agent_id,
                    my_current_v_per_step=my_current_v_per_step,
                    my_current_pose=[goal_pt[0], goal_pt[1], 0, goal_yaw],
                    return_list=True)
                goal_fit = False
                for current_lane in current_lanes:
                    if current_lane in current_route:
                        goal_fit = True
                        break
                current_state['predicting']['goal_fit'] = goal_fit
                # current_state['predicting']['mark_pts'] = [goal_pt]

            if not self.current_on_road:
                print("TEST OFF ROAD!!!!!!!!")

            planning_results_dictionary = self.online_predictor.data['predicting']['marginal_trajectory'][ego_agent_id]
            total_k = planning_results_dictionary['rst'].shape[0]
            closest_distance_to_route = 999999
            closest_idx = None
            for k in range(total_k):
                traj = planning_results_dictionary['rst'][k]
                ending_pt = traj[-1, :2]
                ending_yaw = utils.get_angle_of_a_line(traj[-2, :2], traj[-1, :2])
                current_lanes, current_closest_pt_indices, dist_to_lane, distance_threshold = self.find_closes_lane(
                    current_state=current_state,
                    agent_id=ego_agent_id,
                    my_current_v_per_step=my_current_v_per_step,
                    my_current_pose=[ending_pt[0], ending_pt[1], 0, ending_yaw],
                    return_list=True)
                if dist_to_lane < closest_distance_to_route:
                    closest_distance_to_route = dist_to_lane
                    closest_idx = k

                current_state['predicting']['trajectory_to_mark'].append(traj)

            traj_from_e2e = planning_results_dictionary['rst'][closest_idx]
            my_interpolator = SudoInterpolator(traj_from_e2e, my_current_pose)
            planed_traj = self.get_trajectory_from_interpolator(my_interpolator=my_interpolator,
                                                                my_current_speed=my_current_v_per_step)

            # current_state['predicting']['trajectory_to_mark'].append(planed_traj)

            # if agent_id == 181:
            #     for each_traj in prediction_traj_dic_m[agent_id]['rst']:
            #         current_state['predicting']['trajectory_to_mark'].append(each_traj)

            planning_horizon, _ = planed_traj.shape
            current_state['agent'][ego_agent_id]['pose'][current_frame_idx:planning_horizon + current_frame_idx, :] = planed_traj[:total_time_frame - current_frame_idx, :]
            # print("testing traj: ", ego_agent_id, current_state['agent'][ego_agent_id]['pose'][current_frame_idx-5:current_frame_idx+20, :2])
        return current_state


