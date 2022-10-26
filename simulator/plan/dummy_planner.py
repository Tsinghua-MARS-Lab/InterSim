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
T = 0.25  # 1.5  # reaction time when following
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
    return utils.normalize_angle(- yaw - math.pi/2)


class DummyPlanner(EnvPlanner):
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
                return

            ego_agent_id = current_state['predicting']['ego_id'][1]
            total_time_frame = current_state['agent'][ego_agent_id]['pose'].shape[0]
            goal_point = current_state['predicting']['goal_pts'][ego_agent_id]
            my_current_pose = current_state['agent'][ego_agent_id]['pose'][current_frame_idx - 1]
            my_current_v_per_step = euclidean_distance(current_state['agent'][ego_agent_id]['pose'][current_frame_idx - 1, :2],
                                                       current_state['agent'][ego_agent_id]['pose'][current_frame_idx - 6, :2]) / 5

            if ego_agent_id in current_state['predicting']['route']:
                current_route = current_state['predicting']['route'][ego_agent_id]
                init_route = False
            else:
                current_route = []
                init_route = True
            _, current_route = self.get_reroute_traj(current_state=current_state,
                                                     agent_id=ego_agent_id,
                                                     current_frame_idx=current_frame_idx,
                                                     dynamic_turnings=True,
                                                     current_route=current_route)

            current_yaw = my_current_pose[3]
            ending_pt = [my_current_pose[0] - np.sin(change_axis(
                current_yaw)) * 1000, my_current_pose[1] - np.cos(change_axis(current_yaw)) * 1000]
            my_interpolator = SudoInterpolator(
                np.array([my_current_pose[:2], ending_pt]), my_current_pose)

            planed_traj = self.get_trajectory_from_interpolator(my_interpolator=my_interpolator,
                                                                my_current_speed=my_current_v_per_step,
                                                                desired_speed=7)

            current_state['predicting']['trajectory_to_mark'].append(
                planed_traj)

            # if agent_id == 181:
            #     for each_traj in prediction_traj_dic_m[agent_id]['rst']:
            #         current_state['predicting']['trajectory_to_mark'].append(each_traj)

            planning_horizon, _ = planed_traj.shape
            current_state['agent'][ego_agent_id]['pose'][current_frame_idx:planning_horizon +
                                                         current_frame_idx, :] = planed_traj[:total_time_frame - current_frame_idx, :]
            # print("testing traj: ", ego_agent_id, current_state['agent'][ego_agent_id]['pose'][current_frame_idx-5:current_frame_idx+20, :2])
        return current_state


class SudoInterpolator:
    def __init__(self, trajectory, current_pose):
        self.trajectory = trajectory
        self.current_pose = current_pose

    def interpolate(self, distance: float, starting_from=None, debug=False):
        if starting_from is not None:
            assert False, 'not implemented'
        else:
            pose = self.trajectory.copy()
        if distance <= MINIMAL_DISTANCE_PER_STEP:
            return self.current_pose
        total_frame, _ = pose.shape
        # assert distance > 0, distance
        distance_input = distance
        for i in range(total_frame):
            if i == 0:
                pose1 = self.current_pose[:2]
                pose2 = pose[0, :2]
            else:
                pose1 = pose[i - 1, :2]
                pose2 = pose[i, :2]
            next_step = euclidean_distance(pose1, pose2)
            if debug:
                print(
                    f"{i} {next_step} {distance} {total_frame} {self.current_pose}")
            if next_step >= MINIMAL_DISTANCE_PER_STEP:
                if distance > next_step and i != total_frame - 1:
                    distance -= next_step
                    continue
                else:
                    return self.get_state_from_poses(pose1, pose2, distance, next_step)
                    # x = (pose2[0] - pose1[0]) * distance / next_step + pose1[0]
                    # y = (pose2[1] - pose1[1]) * distance / next_step + pose1[1]
                    # yaw = utils.normalize_angle(get_angle_of_a_line(pt1=pose1, pt2=pose2))
                    # return [x, y, 0, yaw]
        if distance_input > distance:
            # hide it outshoot
            # logging.warning(f'Over shooting while planning!!!')
            return self.get_state_from_poses(pose1, pose2, distance, next_step)
        else:
            # return current pose if trajectory not moved at all
            pose1 = self.current_pose[:2]
            pose2 = pose[0, :2]
            return self.get_state_from_poses(pose1, pose2, 0, 0.01)

    def get_state_from_poses(self, pose1, pose2, mul, divider):
        x = (pose2[0] - pose1[0]) * mul / (divider + 0.0001) + pose1[0]
        y = (pose2[1] - pose1[1]) * mul / (divider + 0.0001) + pose1[1]
        yaw = utils.normalize_angle(get_angle_of_a_line(pt1=pose1, pt2=pose2))
        return [x, y, 0, yaw]

    def get_distance_with_index(self, index: int):
        distance = 0
        if index != 0:
            pose = self.trajectory.copy()
            total_frame, _ = pose.shape
            for i in range(total_frame):
                if i >= index != -1:
                    # pass -1 to travel through all indices
                    break
                elif i == 0:
                    step = euclidean_distance(
                        self.current_pose[:2], pose[i, :2])
                else:
                    step = euclidean_distance(pose[i, :2], pose[i-1, :2])
                if step > MINIMAL_DISTANCE_PER_STEP:
                    distance += step
        return distance

    def get_speed_with_index(self, index: int):
        if index != 0:
            p_t = self.trajectory[index, :2]
            p_t1 = self.trajectory[index - 1, :2]
            speed_per_step = utils.euclidean_distance(p_t, p_t1)
            return speed_per_step
        else:
            return None


class Agent:
    def __init__(self,
                 # init location, angle, velocity
                 x=0.0, y=0.0, yaw=0.0, vx=0.01, vy=0, length=4.726, width=1.842, agent_id=None, color=None):
        self.x = x  # px
        self.y = y
        self.yaw = change_axis(yaw)
        self.vx = vx  # px/frame
        self.vy = vy
        self.length = length  # px
        self.width = width  # px
        self.agent_polys = []
        self.crashed = False
        self.agent_id = agent_id
        self.color = color
