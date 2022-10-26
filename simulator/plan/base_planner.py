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


class BasePlanner(EnvPlanner):
    """
    BasePlanner should not use the gt future trajectory information for planning.
    Using the gt trajectory for collision avoidance is not safe.
    The future trajectory might be changed after the ego planner's planning by the env planner.
    The BasePlanner has its own predictor which does not share information with the predictor of the EnvPlanner
    The BasePlanner is used to control the ego agent only.
    The BasePlanner is derived from the EnvPlanner, change predefined functions to build/test your own planner.
    """

    def plan_ego(self, current_state, current_frame_idx):
        # self.scenario_frame_number = current_frame_idx
        # frame_diff = self.scenario_frame_number - self.planning_from
        current_state['predicting']['emergency_stopping'] = False

        if self.is_planning(current_frame_idx):
        # if frame_diff >= 0 and frame_diff % self.planning_interval == 0:
            # load scenario data
            if current_state is None:
                return

            planner_tic = time.perf_counter()
            if 'planner_timer' not in current_state:
                current_state['planner_timer'] = []
                current_state['predict_timer'] = []

            ego_agent_id = current_state['predicting']['ego_id'][1]
            current_state['agent'][ego_agent_id]['action'] = 'follow'
            total_time_frame = current_state['agent'][ego_agent_id]['pose'].shape[0]
            goal_point = current_state['predicting']['goal_pts'][ego_agent_id]
            my_current_pose = current_state['agent'][ego_agent_id]['pose'][current_frame_idx - 1]
            my_current_v_per_step = euclidean_distance(current_state['agent'][ego_agent_id]['pose'][current_frame_idx - 1, :2],
                                                       current_state['agent'][ego_agent_id]['pose'][current_frame_idx - 6, :2]) / 5
            my_target_speed = 7  # change this to the speed limit of the current lane
            if my_current_v_per_step > 10:
                my_current_v_per_step = 1
            elif my_current_v_per_step < 0.01:
                my_current_v_per_step = 0

            if PRINT_TIMER:
                last_tic = time.perf_counter()

            # past_goal = self.check_past_goal(traj=current_state['agent'][ego_agent_id]['pose'],
            #                                  current_idx=current_frame_idx,
            #                                  current_state=current_state,
            #                                  agent_id=ego_agent_id)
            if True: # (self.follow_gt_first and (past_goal or current_frame_idx > 70)) or my_current_v_per_step <= 1:  # agent_id == ego_id:
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
                                                               current_route=current_route,
                                                               is_ego=True)
                if init_route and len(current_route) > 2:
                    current_state['predicting']['route'][ego_agent_id] = current_route
                    # check if original end point fits current route
                    goal_pt, goal_yaw = self.online_predictor.data['predicting']['goal_pts'][ego_agent_id]

                    current_lanes, current_closest_pt_indices, dist_to_lane, distance_threshold = self.find_closes_lane(
                        current_state=current_state,
                        agent_id=ego_agent_id,
                        my_current_v_per_step=my_current_v_per_step,
                        my_current_pose=[goal_pt[0], goal_pt[1], 0, goal_yaw],
                        return_list=True)

                    if dist_to_lane is None:
                        dist_to_lane = 99999

                    # check goal fitted on current route
                    goal_fit = False
                    if dist_to_lane < 1:
                        goal_fit = True
                    # goal_fit = False
                    # for current_lane in current_lanes:
                    #     if current_lane in current_route:
                    #         goal_fit = True
                    #         break
                    current_state['predicting']['goal_fit'] = goal_fit
                    current_state['predicting']['mark_pts'] = [goal_pt]
                # else:
                #     self.find_closest_index_on_route(route=current_route,
                #                                      my_current_pose=[])

            else:
                my_traj = current_state['agent'][ego_agent_id]['pose'][current_frame_idx-1:, :2]

            if not self.current_on_road:
                print("TEST OFF ROAD!!!!!!!!")

            if PRINT_TIMER:
                print(f"Time spent on ego reroute:  {time.perf_counter() - last_tic:04f}s")
                last_tic = time.perf_counter()

            my_interpolator = SudoInterpolator(my_traj.copy(), my_current_pose)
            my_interpolated_trajectory = self.get_trajectory_from_interpolator(my_interpolator=my_interpolator,
                                                                               my_current_speed=my_current_v_per_step,
                                                                               agent_id=ego_agent_id)
            my_traj = my_interpolated_trajectory[:, :2]
            my_interpolator = SudoInterpolator(my_traj.copy(), my_current_pose)

            other_agent_traj = []
            other_agent_ids = []
            # prior for ped and cyclist
            prior_agent_traj = []
            prior_agent_ids = []

            constant_v = False

            if self.predict_env_for_ego_collisions:
                if not constant_v:
                    # predict on all agents for detection checking
                    self.online_predictor.marginal_predict(current_frame=current_frame_idx, selected_agent_ids='all', current_data=current_state)
                    for each_agent_id in self.online_predictor.data['predicting']['marginal_trajectory'].keys():
                        if each_agent_id == ego_agent_id:
                            continue
                        # if k = 1
                        k = 6
                        # k = 1
                        for n in range(k):
                            pred_traj = self.online_predictor.data['predicting']['marginal_trajectory'][each_agent_id]['rst'][n]
                            total_frames_in_pred = pred_traj.shape[0]
                            pred_traj_with_yaw = np.ones((total_frames_in_pred, 4)) * -1
                            pred_traj_with_yaw[:, :2] = pred_traj[:, :]
                            for t in range(total_frames_in_pred):
                                if t == total_frames_in_pred - 1:
                                    pred_traj_with_yaw[t, 3] = pred_traj_with_yaw[t-1, 3]
                                else:
                                    pred_traj_with_yaw[t, 3] = utils.get_angle_of_a_line(pt1=pred_traj[t, :2], pt2=pred_traj[t+1, :2])
                            other_agent_traj.append(pred_traj_with_yaw)
                            other_agent_ids.append(each_agent_id)
                else:
                    # use constant velocity
                    for each_agent_id in current_state['agent']:
                        if each_agent_id == ego_agent_id:
                            continue
                        varies = [1, 0.5, 0.9, 1.1, 1.5, 2.0]
                        # varies = [1]
                        for v in varies:
                            delta_x = (current_state['agent'][each_agent_id]['pose'][current_frame_idx - 1, 0] - current_state['agent'][each_agent_id]['pose'][current_frame_idx - 6, 0]) / 5
                            delta_y = (current_state['agent'][each_agent_id]['pose'][current_frame_idx - 1, 1] - current_state['agent'][each_agent_id]['pose'][current_frame_idx - 6, 1]) / 5
                            pred_traj_with_yaw = np.ones((80, 4)) * -1
                            pred_traj_with_yaw[:, 3] = current_state['agent'][each_agent_id]['pose'][current_frame_idx - 1, 3]
                            for t in range(80):
                                pred_traj_with_yaw[t, 0] = current_state['agent'][each_agent_id]['pose'][current_frame_idx - 1, 0] + t * delta_x * v
                                pred_traj_with_yaw[t, 1] = current_state['agent'][each_agent_id]['pose'][current_frame_idx - 1, 1] + t * delta_y * v
                            # always yield with constant v
                            prior_agent_traj.append(pred_traj_with_yaw)
                            prior_agent_ids.append(each_agent_id)
            else:
                for each_agent in current_state['agent']:

                    if each_agent == ego_agent_id:
                        continue
                    # check distance
                    if euclidean_distance(current_state['agent'][ego_agent_id]['pose'][current_frame_idx - 1, :2],
                                          current_state['agent'][each_agent]['pose'][current_frame_idx - 1, :2]) > 500 and current_state['agent'][ego_agent_id]['pose'][current_frame_idx - 1, 0] != -1:  # 20m for 1 second on 70km/h
                        continue

                    # check if is a steady agent, parking steady agents stays where they are
                    assert current_frame_idx >= 10, current_frame_idx
                    if euclidean_distance(current_state['agent'][each_agent]['pose'][current_frame_idx - 1, :2],
                                          current_state['agent'][each_agent]['pose'][current_frame_idx - 10, :2]) < 3:
                        steady_in_past = True
                    else:
                        steady_in_past = False

                    # 'predict' its trajectory by following lanes
                    if int(current_state['agent'][each_agent]['type']) not in self.vehicle_types:
                        if current_state['agent'][each_agent]['pose'][current_frame_idx - 1, 0] == -1.0 or current_state['agent'][each_agent]['pose'][current_frame_idx - 6, 0] == -1.0:
                            continue
                        # for non-vehicle types agent
                        delta_x = (current_state['agent'][each_agent]['pose'][current_frame_idx - 1, 0] - current_state['agent'][each_agent]['pose'][current_frame_idx - 6, 0]) / 5
                        delta_y = (current_state['agent'][each_agent]['pose'][current_frame_idx - 1, 1] - current_state['agent'][each_agent]['pose'][current_frame_idx - 6, 1]) / 5
                        # traj_with_yaw = np.ones_like(current_state['agent'][each_agent]['pose'][current_frame_idx:, :]) * -1
                        varies = [1, 0.5, 0.9, 1.1, 1.5, 2.0]
                        # varies = [1]
                        for mul in varies:
                        # for m in range(3):
                            traj_with_yaw = np.ones((80, 4)) * -1
                            traj_with_yaw[:, 3] = current_state['agent'][each_agent]['pose'][current_frame_idx - 1, 3]
                            traj_with_yaw[0, :] = current_state['agent'][each_agent]['pose'][current_frame_idx, :]
                            # if m == 0:
                            #     # add multiple trajectory to be conservative
                            #     d = 1.1
                            # elif m == 1:
                            #     d = 0.9
                            # else:
                            #     d = 0.5

                            for i in range(39):
                                # traj_with_yaw[i + 1, 0] = traj_with_yaw[i, 0] + min(0.5, delta_x * mul **i)
                                # traj_with_yaw[i + 1, 1] = traj_with_yaw[i, 1] + min(0.5, delta_y * mul **i)
                                traj_with_yaw[i + 1, 0] = traj_with_yaw[i, 0] + min(0.5, delta_x * mul)
                                traj_with_yaw[i + 1, 1] = traj_with_yaw[i, 1] + min(0.5, delta_y * mul)

                            # prior ped
                            # prior_agent_ids.append(each_agent)
                            # prior_agent_traj.append(traj_with_yaw)
                            # pred ped

                            other_agent_ids.append(each_agent)
                            other_agent_traj.append(traj_with_yaw)
                        # current_state['predicting']['trajectory_to_mark'].append(traj_with_yaw)
                    else:
                        if current_state['agent'][each_agent]['pose'][current_frame_idx - 1, 0] == -1.0 or current_state['agent'][each_agent]['pose'][current_frame_idx - 6, 0] == -1.0 or current_state['agent'][each_agent]['pose'][current_frame_idx - 11, 0] == -1.0:
                            continue

                        # for vehicle types agents
                        each_agent_current_pose = current_state['agent'][each_agent]['pose'][current_frame_idx - 1]
                        each_agent_current_v_per_step = euclidean_distance(
                            current_state['agent'][each_agent]['pose'][current_frame_idx - 1, :2],
                            current_state['agent'][each_agent]['pose'][current_frame_idx - 6, :2]) / 5
                        each_agent_current_a_per_step = (euclidean_distance(
                            current_state['agent'][each_agent]['pose'][current_frame_idx - 1, :2],
                            current_state['agent'][each_agent]['pose'][current_frame_idx - 6, :2]) / 5 - euclidean_distance(
                            current_state['agent'][each_agent]['pose'][current_frame_idx - 6, :2],
                            current_state['agent'][each_agent]['pose'][current_frame_idx - 11, :2]) / 5) / 5
                        if each_agent_current_v_per_step > 10:
                            each_agent_current_v_per_step = 1
                        # get the route for each agent, you can use your prediction model here
                        if each_agent_current_v_per_step < 0.25:
                            each_agent_current_v_per_step = 0

                        if each_agent_current_a_per_step > 0.5:
                            each_agent_current_a_per_step = 0.3

                        # 1. find the closest lane
                        current_lanes, current_closest_pt_indices, dist_to_lane, _ = self.find_closes_lane(
                            current_state=current_state,
                            agent_id=each_agent,
                            my_current_v_per_step=each_agent_current_v_per_step,
                            my_current_pose=each_agent_current_pose,
                            no_unparallel=False,
                            return_list=True)

                        # detect parking vehicles
                        if each_agent_current_v_per_step < 0.05 and (dist_to_lane is None or dist_to_lane > 2) and steady_in_past:
                            dummy_steady = np.repeat(
                                current_state['agent'][each_agent]['pose'][current_frame_idx - 1, :][np.newaxis, :], 80,
                                axis=0)
                            prior_agent_ids.append(each_agent)
                            prior_agent_traj.append(dummy_steady)
                            # current_state['agent'][each_agent]['marking'] = "Parking"
                            continue
                        else:
                            current_state['agent'][each_agent]['marking'] = None

                        # 2. search all possible route from this lane and add trajectory from the lane following model
                        # random shooting for all possible routes
                        routes = []
                        for _ in range(10):
                            if isinstance(current_lanes, list):
                                # got a lane changing option list
                                current_lane = random.choice(current_lanes)
                                idx = current_lanes.index(current_lane)
                                current_closest_pt_idx = current_closest_pt_indices[idx]
                            else:
                                current_lane = current_lanes
                                current_closest_pt_idx = current_closest_pt_indices
                            lanes_in_a_route = [current_lane]
                            current_looping = current_lane
                            route_traj_left = np.array(current_state['road'][current_looping]['xyz'][current_closest_pt_idx+10:, :2], ndmin=2)
                            next_lanes = current_state['road'][current_looping]['next_lanes']
                            while len(next_lanes) > 0 and len(lanes_in_a_route) < 5:
                                lanes_in_a_route.append(current_looping)
                                current_looping = random.choice(next_lanes)
                                if current_looping not in current_state['road']:
                                    continue
                                next_lanes = current_state['road'][current_looping]['next_lanes']
                                route_traj_left = np.concatenate((route_traj_left, current_state['road'][current_looping]['xyz'][:, :2]))
                            if lanes_in_a_route not in routes:
                                routes.append(lanes_in_a_route)
                                varies = [1, 0.5, 0.9, 1.1, 1.5, 2.0]
                                # varies = [1]
                                for mul in varies:
                                # for m in range(3):
                                #     if m == 0:
                                #         mul = 1.0
                                #     elif m == 1:
                                #         mul = 0.95
                                #     else:
                                #         mul = 1.05

                                    # get a traj
                                    other_interpolator = SudoInterpolator(route_traj_left.copy(), each_agent_current_pose)
                                    # traj_with_yaw = self.get_trajectory_from_interpolator(my_interpolator=other_interpolator,
                                    #                                                       my_current_speed=each_agent_current_v_per_step,
                                    #                                                       a_per_step=each_agent_current_a_per_step * mul,
                                    #                                                       desired_speed=my_target_speed,
                                    #                                                       check_turning_dynamics=False)
                                    traj_with_yaw = self.get_trajectory_from_interpolator(
                                        my_interpolator=other_interpolator,
                                        my_current_speed=each_agent_current_v_per_step * mul,
                                        a_per_step=each_agent_current_a_per_step,
                                        desired_speed=my_target_speed,
                                        check_turning_dynamics=False)
                                    other_agent_traj.append(traj_with_yaw)
                                    other_agent_ids.append(each_agent)

            if PRINT_TIMER:
                print(f"Time spent on ego planning other agents:  {time.perf_counter() - last_tic:04f}s")
                last_tic = time.perf_counter()

            # check collisions with ego
            prior_collisions = []  # [[collision_frame, target_id], ..] from idx small to large
            collisions = []  # [[collision_frame, target_id], ..] from idx small to large
            ego_org_traj = my_interpolated_trajectory

            def check_traffic_light():
                total_time_frame = 80
                for current_time in range(total_time_frame):
                    if current_frame_idx + current_time < 90:
                        traffic_light_ending_pts = self.get_traffic_light_collision_pts(current_state=current_state,
                                                                                        current_frame_idx=current_frame_idx + min(5, current_time))
                    else:
                        traffic_light_ending_pts = []
                    ego_pose = ego_org_traj[current_time]
                    if abs(ego_pose[0]) < 1.1 and abs(ego_pose[1]) < 1.1:
                        continue
                    ego_agent = Agent(x=ego_pose[0],
                                      y=ego_pose[1],
                                      yaw=ego_pose[3],
                                      length=current_state['agent'][ego_agent_id]['shape'][0][1],
                                      width=current_state['agent'][ego_agent_id]['shape'][0][0],
                                      agent_id=ego_agent_id)

                    # check if ego agent is running a red light
                    if abs(ego_org_traj[-1, 0] + 1) < 0.01 or abs(ego_org_traj[0, 0] + 1) < 0.01:
                        ego_dist = 0
                    else:
                        ego_dist = utils.euclidean_distance(ego_org_traj[-1, :2], ego_org_traj[0, :2])
                    if abs(ego_org_traj[60, 3] + 1) < 0.01:
                        ego_turning_right = False
                    else:
                        ego_yaw_diff = -utils.normalize_angle(ego_org_traj[60, 3] - ego_org_traj[0, 3])
                        ego_running_red_light = False
                        if math.pi / 180 * 15 < ego_yaw_diff and abs(ego_org_traj[60, 3] + 1) > 0.01:
                            ego_turning_right = True
                        else:
                            ego_turning_right = False

                    if not ego_turning_right and ego_dist > 10:
                        for tl_pt in traffic_light_ending_pts:
                            dummy_tf_agent = Agent(x=tl_pt[0], y=tl_pt[1], yaw=0,
                                                   length=TRAFFIC_LIGHT_COLLISION_SIZE,
                                                   width=TRAFFIC_LIGHT_COLLISION_SIZE,
                                                   agent_id=99999)
                            running = utils.check_collision(
                                checking_agent=ego_agent,
                                target_agent=dummy_tf_agent)
                            if running:
                                ego_running_red_light = True
                                return current_time
                return None

            def detect_conflicts_and_solve(others_trajectory, target_agent_ids, always_yield=False):
                total_time_frame = 80
                my_reactors = []
                for current_time in range(total_time_frame):
                    if current_frame_idx + current_time < 90:
                        traffic_light_ending_pts = self.get_traffic_light_collision_pts(current_state=current_state,
                                                                                        current_frame_idx=current_frame_idx + min(5, current_time))
                    else:
                        traffic_light_ending_pts = []
                    ego_running_red_light = False
                    ego_time_length = ego_org_traj.shape[0]

                    if current_time >= ego_time_length:
                        print("break ego length: ", current_time, ego_time_length)
                        break
                    ego_pose = ego_org_traj[current_time]
                    if ego_pose[0] == -1.0 and ego_pose[1] == -1.0:
                        continue
                    ego_agent = Agent(x=ego_pose[0],
                                      y=ego_pose[1],
                                      yaw=ego_pose[3],
                                      length=current_state['agent'][ego_agent_id]['shape'][0][1],
                                      width=current_state['agent'][ego_agent_id]['shape'][0][0],
                                      agent_id=ego_agent_id)
                    # check if ego agent is running a red light
                    if ego_org_traj[-1, 0] == -1.0 or ego_org_traj[0, 0] == -1.0:
                        ego_dist = 0
                    else:
                        ego_dist = utils.euclidean_distance(ego_org_traj[-1, :2], ego_org_traj[0, :2])
                    if ego_org_traj[20, 3] == -1.0:
                        ego_turning_right = False
                    else:
                        ego_yaw_diff = -utils.normalize_angle(ego_org_traj[20, 3] - ego_org_traj[0, 3])
                        ego_running_red_light = False
                        if math.pi / 180 * 15 < ego_yaw_diff and ego_org_traj[20, 3] != -1.0:
                            ego_turning_right = True
                        else:
                            ego_turning_right = False
                    if not ego_turning_right and ego_dist > 10:
                        for tl_pt in traffic_light_ending_pts:
                            dummy_tf_agent = Agent(x=tl_pt[0], y=tl_pt[1], yaw=0,
                                                   length=TRAFFIC_LIGHT_COLLISION_SIZE,
                                                   width=TRAFFIC_LIGHT_COLLISION_SIZE,
                                                   agent_id=99999)
                            running = utils.check_collision(
                                checking_agent=ego_agent,
                                target_agent=dummy_tf_agent)
                            if running:
                                ego_running_red_light = True
                                break

                    if ego_running_red_light:
                        earliest_collision_idx = current_time
                        collision_point = ego_org_traj[current_time, :2]
                        earliest_conflict_agent = 99999
                        target_speed = 0
                        each_other_traj, detected_relation = None, None
                        return [earliest_collision_idx, collision_point, earliest_conflict_agent, target_speed, None, None]

                    for j, each_other_traj in enumerate(others_trajectory):
                        target_agent_id = target_agent_ids[j]
                        if target_agent_id == 5190:
                            print("test: 443", current_time, my_reactors)

                        # Unset=0, Vehicle=1, Pedestrian=2, Cyclist=3, Other=4
                        target_type = int(current_state['agent'][target_agent_id]['type'])
                        if target_type not in self.vehicle_types:
                            target_shape = [max(2, current_state['agent'][target_agent_id]['shape'][0][0]),
                                            max(6, current_state['agent'][target_agent_id]['shape'][0][1])]
                        else:
                            target_shape = [max(1, current_state['agent'][target_agent_id]['shape'][0][0]),
                                            max(1, current_state['agent'][target_agent_id]['shape'][0][1])]

                        if target_agent_id in my_reactors:
                            if target_agent_id == 5190:
                                print("test: 456", current_time)
                            continue
                        total_frame_in_target = each_other_traj.shape[0]
                        if current_time > total_frame_in_target - 1:
                            if target_agent_id == 5190:
                                print("test: 461", current_time)
                            continue
                        target_pose = each_other_traj[current_time]
                        if target_pose[0] == -1.0 or target_pose[1] == -1.0:
                            continue

                        # check if target agent is running a red light
                        yaw_diff = utils.normalize_angle(each_other_traj[-1, 3] - each_other_traj[0, 3])
                        dist = utils.euclidean_distance(each_other_traj[-1, :2], each_other_traj[0, :2])
                        target_agent = Agent(x=target_pose[0],
                                             y=target_pose[1],
                                             yaw=target_pose[3],
                                             length=target_shape[1],
                                             width=target_shape[0],
                                             agent_id=target_agent_id)
                        # check target agent is stopping for a red light
                        running_red_light = False
                        if dist > 10:
                            for tl_pt in traffic_light_ending_pts:
                                dummy_tf_agent = Agent(x=tl_pt[0], y=tl_pt[1], yaw=0,
                                                       length=TRAFFIC_LIGHT_COLLISION_SIZE,
                                                       width=TRAFFIC_LIGHT_COLLISION_SIZE,
                                                       agent_id=99999)
                                running = utils.check_collision(
                                    checking_agent=target_agent,
                                    target_agent=dummy_tf_agent)
                                if running:
                                    # check if they are on two sides of the red light
                                    ego_tf_yaw = utils.get_angle_of_a_line(pt1=ego_pose[:2], pt2=tl_pt[:2])
                                    target_tf_yae = utils.get_angle_of_a_line(pt1=target_pose[:2], pt2=tl_pt[:2])
                                    if abs(utils.normalize_angle(ego_tf_yaw - target_tf_yae)) < math.pi / 2:
                                        running_red_light = True
                                        break

                        if running_red_light and not constant_v:
                            continue

                        if target_agent_id == 5190:
                            print("test: 498", current_time)

                        # check collision with ego vehicle
                        has_collision = utils.check_collision(checking_agent=ego_agent,
                                                              target_agent=target_agent)

                        if current_time < ego_time_length - 1:
                            ego_pose2 = ego_org_traj[current_time + 1]
                            ego_agent2 = Agent(x=(ego_pose2[0] + ego_pose[0]) / 2,
                                               y=(ego_pose2[1] + ego_pose[1]) / 2,
                                               yaw=get_angle_of_a_line(ego_pose[:2], ego_pose2[:2]),
                                               length=max(2, euclidean_distance(ego_pose2[:2], ego_pose[:2])),
                                               width=current_state['agent'][ego_agent_id]['shape'][0][0],
                                               agent_id=ego_agent_id)
                            if current_time < total_time_frame - 1:
                                target_pose2 = each_other_traj[current_time + 1]
                                target_agent2 = Agent(x=target_pose2[0],
                                                      y=target_pose2[1],
                                                      yaw=target_pose2[3],
                                                      length=target_shape[1],
                                                      width=target_shape[0],
                                                      agent_id=target_agent_id)
                                has_collision |= utils.check_collision(checking_agent=ego_agent2, target_agent=target_agent2)
                            else:
                                has_collision |= utils.check_collision(checking_agent=ego_agent2, target_agent=target_agent)

                        if target_agent_id == 5190:
                            print("test: 525", has_collision)

                        if has_collision:
                            if not always_yield:
                                # FORWARD COLLISION CHECKINGS
                                target_pose_0 = each_other_traj[0]
                                target_agent_0 = Agent(x=target_pose_0[0],
                                                       y=target_pose_0[1],
                                                       yaw=target_pose_0[3],
                                                       length=target_shape[1],
                                                       width=target_shape[0],
                                                       agent_id=target_agent_id)
                                collision_0 = False
                                for fcc_time in range(total_time_frame):
                                    ego_pose = ego_org_traj[fcc_time]
                                    if ego_pose[0] == -1.0 and ego_pose[1] == -1.0:
                                        continue
                                    ego_agent = Agent(x=ego_pose[0],
                                                      y=ego_pose[1],
                                                      yaw=ego_pose[3],
                                                      length=current_state['agent'][ego_agent_id]['shape'][0][1],
                                                      width=current_state['agent'][ego_agent_id]['shape'][0][0],
                                                      agent_id=ego_agent_id)

                                    collision_0 |= utils.check_collision(ego_agent, target_agent_0)
                                    if collision_0:
                                        break

                                ego_pose_0 = ego_org_traj[0]
                                ego_agent_0 = Agent(x=ego_pose_0[0],
                                                    y=ego_pose_0[1],
                                                    yaw=ego_pose_0[3],
                                                    length=current_state['agent'][ego_agent_id]['shape'][0][1],
                                                    width=current_state['agent'][ego_agent_id]['shape'][0][0],
                                                    agent_id=ego_agent_id)
                                collision_1 = False
                                for fcc_time in range(total_time_frame):
                                    target_pose = each_other_traj[fcc_time]
                                    if target_pose[0] == -1.0 or target_pose[1] == -1.0:
                                        continue
                                    target_agent = Agent(x=target_pose[0],
                                                         y=target_pose[1],
                                                         yaw=target_pose[3],
                                                         length=target_shape[1],
                                                         width=target_shape[0],
                                                         agent_id=target_agent_id)

                                    collision_1 |= utils.check_collision(target_agent, ego_agent_0)
                                    if collision_1:
                                        break
                                # collision_1 = utils.check_collision(target_agent, ego_agent_0)
                                # collision_1 |= utils.check_collision(target_agent2, ego_agent_0)

                                if collision_0 and self.predict_with_rules:
                                    # yield
                                    detected_relation = [[target_agent_id, ego_agent_id, 'FCC']]
                                elif collision_1 and self.predict_with_rules:
                                    # pass
                                    my_reactors.append(target_agent_id)
                                    continue
                                else:
                                    # check relation
                                    # if collision, solve conflict
                                    predict_tic = time.perf_counter()
                                    self.online_predictor.relation_pred_onetime(each_pair=[ego_agent_id, target_agent_id],
                                                                                current_frame=current_frame_idx,
                                                                                clear_history=True,
                                                                                with_rules=self.predict_with_rules,
                                                                                current_data=current_state)
                                    detected_relation = self.online_predictor.data['predicting']['relation']
                                    predict_time = time.perf_counter() - predict_tic
                                    current_state['predict_timer'].append(predict_time)

                                    if [ego_agent_id, target_agent_id] in detected_relation:
                                        if [target_agent_id, ego_agent_id] in detected_relation:
                                            # bi-directional relations, still yield
                                            pass
                                        else:
                                            # not to yield, and skip conflict
                                            my_reactors.append(target_agent_id)
                                            continue
                            else:
                                detected_relation = [[target_agent_id, ego_agent_id, 'Prior']]

                            copy = []
                            for each_r in detected_relation:
                                if len(each_r) == 2:
                                    copy.append([each_r[0], each_r[1], 'predict'])
                                else:
                                    copy.append(each_r)
                            detected_relation = copy

                            earliest_collision_idx = current_time
                            collision_point = ego_org_traj[current_time, :2]
                            earliest_conflict_agent = target_agent_id

                            if total_frame_in_target - current_time > 5:
                                target_speed = euclidean_distance(each_other_traj[current_time, :2], each_other_traj[current_time + 5, :2]) / 5
                            elif current_time > 5:
                                target_speed = euclidean_distance(each_other_traj[current_time - 5, :2], each_other_traj[current_time, :2]) / 5
                            else:
                                target_speed = 0
                            return [earliest_collision_idx, collision_point, earliest_conflict_agent, target_speed, each_other_traj, detected_relation]

                return None

            earliest_collision_idx = None
            collision_point = None
            earliest_conflict_agent = None
            target_speed = None
            to_yield = False
            detected_relation = None

            tf_light_frame_idx = check_traffic_light()

            # save trajectories
            # for p, each_traj in enumerate(other_agent_traj):
            #     this_agent_id = other_agent_ids[p]
            #     if this_agent_id not in current_state['predicting']['guilded_trajectory']:
            #         current_state['predicting']['guilded_trajectory'][this_agent_id] = []
            #     current_state['predicting']['guilded_trajectory'][this_agent_id] += [each_traj]

            # process prior collision pairs
            rst = detect_conflicts_and_solve(prior_agent_traj, prior_agent_ids, always_yield=True)
            current_state['predicting']['all_relations_last_step'] = []
            if rst is not None and rst[5] is not None:
                current_state['predicting']['all_relations_last_step'] += rst[5]

            if rst is not None and len(rst) == 6:
                earliest_collision_idx, collision_point, earliest_conflict_agent, target_speed, each_other_traj, detected_relation = rst
                # print("test yield prior: ", ego_agent_id, earliest_conflict_agent, earliest_collision_idx, target_speed)
                if each_other_traj is not None:
                    current_state['predicting']['trajectory_to_mark'].append(each_other_traj)
                to_yield = True

            if PRINT_TIMER:
                print(f"Time spent on prior collision detection:  {time.perf_counter() - last_tic:04f}s")
                last_tic = time.perf_counter()

            # if earliest_collision_idx is None:
            # check collisions with not prior collisions
            relation_pred = self.predict_relations_for_ego
            rst = detect_conflicts_and_solve(other_agent_traj, other_agent_ids, always_yield=(not relation_pred))
            if rst is not None and rst[5] is not None:
                current_state['predicting']['all_relations_last_step'] += rst[5]

            if rst is not None and len(rst) == 6:
                if not to_yield or rst[0] < earliest_collision_idx:
                    earliest_collision_idx, collision_point, earliest_conflict_agent, target_speed, each_other_traj, detected_relation = rst
                to_yield = True
                # print("test yield: ", ego_agent_id, earliest_conflict_agent, earliest_collision_idx, target_speed)
                if each_other_traj is not None:
                    current_state['predicting']['trajectory_to_mark'].append(each_other_traj)
            if PRINT_TIMER:
                print(f"Time spent on non-prior collision detection:  {time.perf_counter() - last_tic:04f}s")
                last_tic = time.perf_counter()

            if earliest_collision_idx is not None and (tf_light_frame_idx is None or earliest_collision_idx < tf_light_frame_idx):
                # data to save
                if detected_relation is not None:
                    if 'relations_per_frame_ego' not in current_state['predicting']:
                        current_state['predicting']['relations_per_frame_ego'] = {}
                    for dt in range(self.planning_interval):
                        if (current_frame_idx + dt) not in current_state['predicting']['relations_per_frame_ego']:
                            current_state['predicting']['relations_per_frame_ego'][current_frame_idx + dt] = []
                        current_state['predicting']['relations_per_frame_ego'][current_frame_idx + dt] += detected_relation
            elif tf_light_frame_idx is not None:
                earliest_collision_idx = tf_light_frame_idx
                collision_point = ego_org_traj[earliest_collision_idx, :2]
                earliest_conflict_agent = 99999
                target_speed = 0
                detected_relation = None
                to_yield = True

            if to_yield:
                distance_to_minuse = S0
                if earliest_conflict_agent == 99999:
                    distance_to_minuse = 0.1
                distance_to_travel = my_interpolator.get_distance_with_index(earliest_collision_idx) - distance_to_minuse
                stopping_point = my_interpolator.interpolate(distance_to_travel - distance_to_minuse)[:2]

                # current_state['predicting']['mark_pts'] = [stopping_point]
                if distance_to_travel < MINIMAL_DISTANCE_PER_STEP:
                # if euclidean_distance(my_traj[0, :2],
                #                       stopping_point) < MINIMAL_DISTANCE_TO_TRAVEL or distance_to_travel < MINIMAL_DISTANCE_TO_TRAVEL or my_current_v_per_step < 0.1:
                    planed_traj = self.get_trajectory_from_interpolator(my_interpolator=my_interpolator,
                                                                        my_current_speed=my_current_v_per_step,
                                                                        desired_speed=my_target_speed,
                                                                        emergency_stop=True)
                    current_state['predicting']['emergency_stopping'] = True
                    # planed_traj = self.get_rescale_trajectory(reactor_current_pose=my_current_pose,
                    #                                           reactor_traj=my_traj,
                    #                                           reactor_interpolator=my_interpolator,
                    #                                           scale=scale,
                    #                                           current_v_per_step=my_current_v_per_step,
                    #                                           target_speed=my_target_speed)
                elif my_current_v_per_step < 0.1 and euclidean_distance(my_traj[0, :2], my_traj[-1, :2]) < MINIMAL_DISTANCE_TO_TRAVEL:
                    planed_traj = self.get_trajectory_from_interpolator(my_interpolator=my_interpolator,
                                                                        my_current_speed=my_current_v_per_step,
                                                                        desired_speed=my_target_speed,
                                                                        hold_still=True)
                else:
                    planed_traj = self.adjust_speed_for_collision(interpolator=my_interpolator,
                                                                  distance_to_end=distance_to_travel,
                                                                  current_v=my_current_v_per_step,
                                                                  end_point_v=min(my_current_v_per_step * 0.8,
                                                                                  target_speed))
                    assert len(planed_traj.shape) > 1, planed_traj.shape
                    # my_interpolator = SudoInterpolator(my_traj, my_current_pose)
                    # planed_traj = self.get_rescale_trajectory(reactor_current_pose=my_current_pose,
                    #                                           reactor_traj=my_traj,
                    #                                           reactor_interpolator=my_interpolator,
                    #                                           scale=1,
                    #                                           current_v_per_step=my_current_v_per_step,
                    #                                           target_speed=my_target_speed)
            else:
                if euclidean_distance(my_traj[0, :2], my_traj[-1, :2]) < MINIMAL_DISTANCE_TO_TRAVEL:
                    planed_traj = self.get_trajectory_from_interpolator(my_interpolator=my_interpolator,
                                                                        my_current_speed=my_current_v_per_step,
                                                                        desired_speed=my_target_speed,
                                                                        hold_still=True)
                else:
                    # print("test: no yield for ", ego_agent_id)
                    planed_traj = my_interpolated_trajectory

            if PRINT_TIMER:
                print(f"Time spent on adjust speed:  {time.perf_counter() - last_tic:04f}s")
                last_tic = time.perf_counter()

            current_state['predicting']['trajectory_to_mark'].append(planed_traj)

            if planed_traj.shape[0] < 10:
                planed_traj = self.get_trajectory_from_interpolator(my_interpolator=my_interpolator,
                                                                    my_current_speed=my_current_v_per_step,
                                                                    desired_speed=my_target_speed,
                                                                    hold_still=True)
            assert planed_traj.shape[0] > 5,  planed_traj.shape

            planning_horizon, _ = planed_traj.shape
            jumping_idx = self.assert_traj(planed_traj[:total_time_frame - current_frame_idx, :2])
            if jumping_idx != -1:
                if jumping_idx >= 6:
                    planed_traj[jumping_idx:, :] = -1
                else:
                    print(f'Early jumping {jumping_idx} {ego_agent_id}')
                    # assert False, f'Jumping early: {jumping_idx} {ego_agent_id}'
            planned_shape = planed_traj[:total_time_frame - current_frame_idx, :].shape
            if total_time_frame - current_frame_idx > 1 and len(planned_shape) > 1 and planned_shape[1] == 4:
                current_state['agent'][ego_agent_id]['pose'][current_frame_idx:planning_horizon + current_frame_idx, :] = planed_traj[:total_time_frame - current_frame_idx, :]
                current_state['agent'][ego_agent_id]['pose'][planning_horizon + current_frame_idx:, :] = -1
            else:
                print("WARNING: No planning trajectory replacing!! ", total_time_frame, current_frame_idx, planned_shape, planed_traj)
            # print("testing traj: ", ego_agent_id, current_state['agent'][ego_agent_id]['pose'][current_frame_idx-5:current_frame_idx+20, :2])

            planner_time = time.perf_counter() - planner_tic
            current_state['planner_timer'].append(planner_time)

        return current_state
