from prediction.M2I.predictor import M2IPredictor
import numpy as np
import math
import logging
import copy
import random
import time

import interactive_sim.envs.util as utils
import plan.helper as plan_helper
import agents.car as car

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

def get_current_pose_and_v(current_state, agent_id, current_frame_idx):
    agent_dic = current_state['predicting']['original_trajectory']
    my_current_pose = agent_dic[agent_id]['pose'][current_frame_idx - 1]
    if agent_dic[agent_id]['pose'][current_frame_idx - 1, 0] == -1 or agent_dic[agent_id]['pose'][current_frame_idx - 6, 0] == -1:
        my_current_v_per_step = 0
        print("Past invalid for ", agent_id, " and setting v to 0")
    else:
        my_current_v_per_step = euclidean_distance(agent_dic[agent_id]['pose'][current_frame_idx - 1, :2],
                                                   agent_dic[agent_id]['pose'][current_frame_idx - 6, :2]) / 5
    return my_current_pose, my_current_v_per_step


class EnvPlanner:
    """
    EnvPlanner is capable of using as much information as it can to satisfy its loss like avoiding collisions.
    EnvPlanner can assume it's controlling all agents around if it does not exacerbate the sim-2-real gap.
    While the baseline planner or any planner controlling the ego vehicle can only use the prediction or past data
    """

    def __init__(self, env_config, predictor, dataset='Waymo', map_api=None):
        self.planning_from = env_config.env.planning_from
        self.planning_interval = env_config.env.planning_interval
        self.planning_horizon = env_config.env.planning_horizon
        self.planning_to = env_config.env.planning_to
        self.scenario_frame_number = 0
        self.online_predictor = predictor
        self.method_testing = env_config.env.testing_method  # 0=densetnt with dropout, 1=0+post-processing, 2=1+relation
        self.test_task = env_config.env.test_task
        self.all_relevant = env_config.env.all_relevant
        self.follow_loaded_relation = env_config.env.follow_loaded_relation
        self.follow_prediction_traj = env_config.env.follow_prediction
        self.target_lanes = [0, 0]  # lane_index, point_index
        self.routed_traj = {}
        self.follow_gt_first = env_config.env.follow_gt_first

        self.predict_env_for_ego_collisions = env_config.env.predict_env_for_ego_collisions
        self.predict_relations_for_ego = env_config.env.predict_relations_for_ego
        self.predict_with_rules = env_config.env.predict_with_rules
        self.frame_rate = env_config.env.frame_rate

        self.current_on_road = True
        self.dataset = dataset
        self.online_predictor.dataset = dataset

        self.valid_lane_types = [1, 2] if self.dataset == 'Waymo' else [0, 11]
        self.vehicle_types = [1] if self.dataset == 'Waymo' else [0, 7]  # Waymo: Unset=0, Vehicle=1, Pedestrian=2, Cyclist=3, Other=4
        self.map_api = map_api  # NuPlan only
        self.past_lanes = {}

    def reset(self, *args, **kwargs):
        time1 = time.perf_counter()
        self.online_predictor(new_data=kwargs['new_data'], model_path=kwargs['model_path'],
                              time_horizon=kwargs['time_horizon'], predict_device=kwargs['predict_device'],
                              use_prediction=(self.follow_prediction_traj or self.predict_env_for_ego_collisions) and kwargs['ego_planner'],
                              predictor_list=kwargs['predictor_list'])
        time2 = time.perf_counter()
        self.online_predictor.setting_goal_points(current_data=kwargs['new_data'])
        self.current_on_road = True
        print(f"predictor reset with {time2-time1:04f}s")
        # self.data = self.online_predictor.data

    def is_planning(self, current_frame_idx):
        self.scenario_frame_number = current_frame_idx
        frame_diff = self.scenario_frame_number - self.planning_from
        if frame_diff >= 0 and frame_diff % self.planning_interval == 0:
            return True
        return False

    def is_first_planning(self, current_frame_idx):
        self.scenario_frame_number = current_frame_idx
        frame_diff = self.scenario_frame_number - self.planning_from
        if frame_diff >= 0 and frame_diff == 0: # frame_diff % self.planning_interval == 0:
            return True
        return False

    def collision_based_relevant_detection(self, current_frame_idx, current_state, predict_ego=True):
        ego_agent = current_state['predicting']['ego_id'][1]
        # print("before: ", current_state['predicting']['relevant_agents'], bool(current_state['predicting']['relevant_agents']))
        if not current_state['predicting']['relevant_agents']:
            relevant_agents = [ego_agent]
            undetected_piles = [ego_agent]
        else:
            relevant_agents = current_state['predicting']['relevant_agents'].copy()
            if ego_agent not in relevant_agents:
                relevant_agents += [ego_agent]
            undetected_piles = relevant_agents.copy()
        colliding_pairs = []
        while len(undetected_piles) > 0:
            if self.all_relevant:
                # hard force all agents as relevant
                current_agent = undetected_piles.pop()
                for each_agent_id in current_state['agent']:
                    if each_agent_id != current_agent:
                        relevant_agents.append(each_agent_id)
                break

            current_agent = undetected_piles.pop()
            ego_poses = current_state['agent'][current_agent]['pose']
            ego_shape = current_state['agent'][current_agent]['shape'][0]
            detected_pairs = []
            ego_agent_0 = None
            for idx, each_pose in enumerate(ego_poses):
                if idx <= current_frame_idx:
                    continue
                ego_agent_packed =Agent(x=each_pose[0],
                                             y=each_pose[1],
                                             yaw=each_pose[3],
                                             length=max(1, ego_shape[1]),
                                             width=max(1, ego_shape[0]),
                                             agent_id=current_agent)
                if ego_agent_0 is None:
                    ego_agent_0 = ego_agent_packed
                for each_agent_id in current_state['agent']:
                    if [current_agent, each_agent_id] in detected_pairs:
                        continue
                    if each_agent_id == current_agent or each_agent_id in relevant_agents:
                        continue
                    each_agent_frame_num = current_state['agent'][each_agent_id]['pose'].shape[0]
                    if idx >= each_agent_frame_num:
                        continue
                    target_agent_packed =Agent(x=current_state['agent'][each_agent_id]['pose'][idx, 0],
                                                    y=current_state['agent'][each_agent_id]['pose'][idx, 1],
                                                    yaw=current_state['agent'][each_agent_id]['pose'][idx, 3],
                                                    length=current_state['agent'][each_agent_id]['shape'][0][1],
                                                    width=current_state['agent'][each_agent_id]['shape'][0][0],
                                                    agent_id=each_agent_id)
                    if each_pose[0] == -1 or each_pose[1] == -1 or current_state['agent'][each_agent_id]['pose'][idx, 0] == -1 or current_state['agent'][each_agent_id]['pose'][idx, 1] == -1:
                        continue
                    collision = utils.check_collision(ego_agent_packed, target_agent_packed)
                    if collision:
                        detected_pairs.append([current_agent, each_agent_id])
                        yield_ego = True

                        # FORWARD COLLISION CHECKINGS
                        collision_0 = utils.check_collision(ego_agent_0, target_agent_packed)
                        if collision_0:
                            detected_relation = [[ego_agent_0, target_agent_packed]]
                        else:
                            # check relation
                            # print(f"In: {current_agent} {each_agent_id} {undetected_piles} {current_state['predicting']['relation']}")
                            self.online_predictor.relation_pred_onetime(each_pair=[current_agent, each_agent_id],
                                                                        current_frame=current_frame_idx,
                                                                        clear_history=True,
                                                                        current_data=current_state)
                            # print(f"Out: {current_agent} {each_agent_id} {undetected_piles} {current_state['predicting']['relation']}")
                            detected_relation = current_state['predicting']['relation']
                            if [each_agent_id, current_agent] in detected_relation:
                                if [current_agent, each_agent_id] in detected_relation:
                                    # bi-directional relations, still yield
                                    pass
                                else:
                                    yield_ego = False

                        if yield_ego or self.method_testing < 2:
                            relevant_agents.append(each_agent_id)
                            undetected_piles.append(each_agent_id)
                            if [current_agent, each_agent_id] not in colliding_pairs and [each_agent_id, current_agent] not in colliding_pairs:
                                colliding_pairs.append([current_agent, each_agent_id])

            # print(f"Detected for {current_agent} with {undetected_piles}")
        if self.test_task != 1:
            # don't predict ego
            relevant_agents.remove(ego_agent)
        current_state['predicting']['relevant_agents'] = relevant_agents
        current_state['predicting']['colliding_pairs'] = colliding_pairs
        # print(f"Collision based relevant agent detected finished: \n{relevant_agents} \n{colliding_pairs}")

    def clear_markers_per_step(self, current_state, current_frame_idx):
        if self.is_planning(current_frame_idx):
            current_state['predicting']['relation'] = []
            current_state['predicting']['points_to_mark'] = []
            current_state['predicting']['trajectory_to_mark'] = []

    def get_prediction_trajectories(self, current_frame_idx, current_state=None, time_horizon=80):
        if self.is_planning(current_frame_idx):
            frame_diff = self.scenario_frame_number - self.planning_from
            self.collision_based_relevant_detection(current_frame_idx, current_state)
            current_state['predicting']['relation'] = []
            for each_pair in current_state['predicting']['colliding_pairs']:
                self.online_predictor.relation_pred_onetime(each_pair=each_pair, current_data=current_state,
                                                            current_frame=current_frame_idx)
            if self.follow_prediction_traj and len(current_state['predicting']['relevant_agents']) > 0:
                if self.method_testing < 0:
                    self.online_predictor.variety_predict(frame_diff)
                else:
                    self.online_predictor.marginal_predict(frame_diff)
            self.online_predictor.last_predict_frame = frame_diff + 5
            return True
        else:
            return False

    # def update_env_trajectory_speed_only(self, current_frame_idx, relevant_only=True, current_state=None):
    def update_env_trajectory_for_sudo_base_planner(self, current_frame_idx, current_state=None):
        """
        the sudo base planner for the ego vehicle
        """
        if self.test_task in [1, 2]:
            # predict ego
            return current_state

        # self.scenario_frame_number = current_frame_idx
        ego_id = current_state['predicting']['ego_id'][1]

        # for each_agent in current_state['agent']:
        #     if each_agent in [748, 781, 735]:
        #         current_state['predicting']['trajectory_to_mark'].append(
        #             current_state['predicting']['original_trajectory'][each_agent]['pose'][:, :])
        # frame_diff = self.scenario_frame_number - self.planning_from
        # if frame_diff >= 0 and frame_diff == 0: # frame_diff % self.planning_interval == 0:
        if self.is_first_planning(current_frame_idx):
            # print("updating ego trajectory: ", self.planning_interval, self.scenario_frame_number)
            # current_state['predicting']['trajectory_to_mark'].append(
            #     current_state['predicting']['original_trajectory'][ego_id]['pose'][current_frame_idx:, :])
            my_current_pose = current_state['agent'][ego_id]['pose'][current_frame_idx - 1]
            my_current_v_per_step = euclidean_distance(
                current_state['agent'][ego_id]['pose'][current_frame_idx - 1, :2],
                current_state['agent'][ego_id]['pose'][current_frame_idx - 2, :2])
            org_pose = current_state['predicting']['original_trajectory'][ego_id]['pose'].copy()

            projected_pose_on_original = my_current_pose
            closest_distance = 999999
            closest_index = 0
            for idx, each_pose in enumerate(org_pose):
                dist = euclidean_distance(each_pose[:2], my_current_pose[:2])
                if dist < closest_distance:
                    closest_distance = dist
                    projected_pose_on_original = each_pose
                    closest_index = idx
            my_interpolator = SudoInterpolator(org_pose[closest_index:, :2], projected_pose_on_original)
            # my_current_pose = projected_pose_on_original
            total_frames = current_state['agent'][ego_id]['pose'].shape[0]
            total_distance_traveled = 0
            for i in range(total_frames - current_frame_idx):
                my_current_v_per_step -= A_SLOWDOWN_DESIRE/self.frame_rate/self.frame_rate
                step_speed = euclidean_distance(
                    current_state['agent'][ego_id]['pose'][current_frame_idx+i - 1, :2],
                    current_state['agent'][ego_id]['pose'][current_frame_idx+i - 2, :2])
                my_current_v_per_step = max(0, min(my_current_v_per_step, step_speed))
                current_state['agent'][ego_id]['pose'][current_frame_idx+i, :] = my_interpolator.interpolate(total_distance_traveled + my_current_v_per_step)
                total_distance_traveled += my_current_v_per_step

        if self.is_planning(self.scenario_frame_number):
            # current_state['predicting']['trajectory_to_mark'].append(
            #     current_state['predicting']['original_trajectory'][ego_id]['pose'][current_frame_idx:, :])
            current_state['predicting']['trajectory_to_mark'].append(current_state['agent'][ego_id]['pose'][current_frame_idx:, :])

        return current_state

    def find_closes_lane(self, current_state, agent_id, my_current_v_per_step, my_current_pose, no_unparallel=False,
                         return_list=False, current_route=[]):
        # find a closest lane to trace
        closest_dist = 999999
        closest_dist_no_yaw = 999999
        closest_dist_threshold = 5
        closest_lane = None
        closest_lane_no_yaw = None
        closest_lane_pt_no_yaw_idx = None
        closest_lane_pt_idx = None

        current_lane = None
        current_closest_pt_idx = None
        dist_to_lane = None
        distance_threshold = None

        closest_lanes_same_dir = []
        closest_lanes_idx_same_dir = []

        for each_lane in current_state['road']:
            if len(current_route) > 0 and each_lane not in current_route:
                continue

            if isinstance(current_state['road'][each_lane]['type'], int):
                if current_state['road'][each_lane]['type'] not in self.valid_lane_types:
                    continue
            else:
                if current_state['road'][each_lane]['type'][0] not in self.valid_lane_types:
                    continue

            road_xy = current_state['road'][each_lane]['xyz'][:, :2]
            if road_xy.shape[0] < 3:
                continue
            current_lane_closest_dist = 999999
            current_lane_closest_idx = None

            for j, each_xy in enumerate(road_xy):
                road_yaw = current_state['road'][each_lane]['dir'][j]
                dist = euclidean_distance(each_xy, my_current_pose[:2])
                yaw_diff = abs(utils.normalize_angle(my_current_pose[3] - road_yaw))
                if dist < closest_dist_no_yaw:
                    closest_lane_no_yaw = each_lane
                    closest_dist_no_yaw = dist
                    closest_lane_pt_no_yaw_idx = j
                if yaw_diff < math.pi / 180 * 20 and dist < closest_dist_threshold:
                    if dist < closest_dist:
                        closest_lane = each_lane
                        closest_dist = dist
                        closest_lane_pt_idx = j
                    if dist < current_lane_closest_dist:
                        current_lane_closest_dist = dist
                        current_lane_closest_idx = j

            # classify current agent as a lane changer or not:
            if my_current_v_per_step > 0.1 and 0.5 < current_lane_closest_dist < 3.2 and each_lane not in closest_lanes_same_dir and current_state['road'][each_lane]['turning'] == 0:
                closest_lanes_same_dir.append(each_lane)
                closest_lanes_idx_same_dir.append(current_lane_closest_idx)

        if closest_lane is not None and not 0.5 < closest_dist < 3.2:
            closest_lanes_same_dir = []
            closest_lanes_idx_same_dir = []


        if closest_lane is not None:
            current_lane = closest_lane
            current_closest_pt_idx = closest_lane_pt_idx
            dist_to_lane = closest_dist
            distance_threshold = max(7, max(7 * my_current_v_per_step, dist_to_lane))
        elif closest_lane_no_yaw is not None and not no_unparallel:
            current_lane = closest_lane_no_yaw
            current_closest_pt_idx = closest_lane_pt_no_yaw_idx
            dist_to_lane = closest_dist_no_yaw
            distance_threshold = max(10, dist_to_lane)
        else:
            logging.warning(f'No current lane founded: {agent_id}')
            # return
        if return_list:
            if len(closest_lanes_same_dir) > 0:
                return closest_lanes_same_dir, closest_lanes_idx_same_dir, dist_to_lane, distance_threshold
            else:
                return [current_lane], [current_closest_pt_idx], dist_to_lane, distance_threshold
        else:
            return current_lane, current_closest_pt_idx, dist_to_lane, distance_threshold

    def set_route(self, goal_pt, road_dic, current_pose=None, previous_routes=None, max_number_of_routes=50, route_roadblock_check=None, agent_id=None):
        from nuplan.common.actor_state.state_representation import Point2D
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer

        closest_lane_id, dist_to_lane = self.map_api.get_distance_to_nearest_map_object(point=Point2D(current_pose[0], current_pose[1]),
                                                                                        layer=SemanticMapLayer.LANE)
        target_lane_id, dist_to_lane = self.map_api.get_distance_to_nearest_map_object(point=Point2D(goal_pt[0], goal_pt[1]),
                                                                                       layer=SemanticMapLayer.LANE)

        if route_roadblock_check is not None and agent_id == 'ego':
            route_lanes = []
            for each_roadbloack in route_roadblock_check:
                if each_roadbloack not in road_dic:
                    continue
                route_lanes += road_dic[each_roadbloack]['lower_level']
            if closest_lane_id not in route_lanes:
                closest_lane_id, dist_to_lane = self.map_api.get_distance_to_nearest_map_object(
                    point=Point2D(current_pose[0], current_pose[1]),
                    layer=SemanticMapLayer.LANE_CONNECTOR)
                if closest_lane_id not in route_lanes:
                    for each_lane in route_lanes:
                        if each_lane not in self.past_lanes:
                            print("[env planner] WARNING: closest lane/connector in original route not found with closest lanes for ego")
                            closest_lane_id = each_lane
                            dist_to_lane = 1
                            break

        if not isinstance(dist_to_lane, int) or dist_to_lane > 30:
            target_lane_id, dist_to_lane = self.map_api.get_distance_to_nearest_map_object(
                point=Point2D(goal_pt[0], goal_pt[1]),
                layer=SemanticMapLayer.LANE_CONNECTOR)

        closest_lane_id = int(closest_lane_id)
        target_lane_id = int(target_lane_id)

        available_routes = []
        checking_pile = [[closest_lane_id]]
        lanes_visited = []

        if previous_routes is not None:
            for each_route in previous_routes:
                if closest_lane_id in each_route:
                    closest_lane_idx = each_route.index(closest_lane_id)
                    available_routes.append(each_route[closest_lane_idx:])

        while len(checking_pile) > 0 and len(available_routes) < max_number_of_routes:
            # BFS
            next_pile = []
            for each_route in checking_pile:
                latest_lane = each_route[-1]
                if latest_lane not in road_dic:
                    continue
                if latest_lane == target_lane_id:
                    available_routes.append(each_route+[target_lane_id])
                    next_pile = [[closest_lane_id]]
                    lanes_visited = []
                else:
                    all_next_lanes = road_dic[latest_lane]['next_lanes']
                    uppder_roadblock = road_dic[latest_lane]['upper_level'][0]
                    ENVCHANGE_LANE = False
                    if uppder_roadblock in road_dic and ENVCHANGE_LANE:
                        parallel_lanes = road_dic[uppder_roadblock]['lower_level']
                    else:
                        parallel_lanes = []

                    all_next_lanes += parallel_lanes
                    # all_next_lanes += self.road_dic[latest_lane]['upper_level']
                    # if len(all_next_lanes) == 0 and len(each_route) == 1:
                    #     # starting from a dead end, turn around
                    #     all_next_lanes = road_dic[latest_lane]['previous_lanes']
                    for each_next_lane in all_next_lanes:
                        if each_next_lane in each_route:
                            # avoid circles
                            continue
                        if each_next_lane not in lanes_visited:
                            next_pile.append(each_route+[each_next_lane])
                            lanes_visited.append(each_next_lane)
                        else:
                            for each_available_route in available_routes:
                                if each_next_lane in each_available_route:
                                    idx = each_available_route.index(each_next_lane)
                                    if idx != 0:
                                        route_to_add = each_route + [each_next_lane] + each_available_route[idx:]
                                        if route_to_add not in available_routes:
                                            available_routes.append(route_to_add)
                                            break

            checking_pile = next_pile
        return available_routes

    def get_reroute_traj(self, current_state, agent_id, current_frame_idx,
                         follow_org_route=False, dynamic_turnings=True, current_route=[], is_ego=False):
        """
        return a marginal planned trajectory with a simple lane follower
        for NuPlan, use route_roadbloacks. a list of road bloacks
        for Waymo, use route, a list of lane_ids, and prior, a list of lane_ids detected from the original gt trajectories
        """
        assert self.routed_traj is not None, self.routed_traj
        # generate a trajectory based on the route
        # 1. get the route for relevant agents
        # find the closest lane to trace
        my_current_pose, my_current_v_per_step = plan_helper.get_current_pose_and_v(current_state=current_state,
                                                                                    agent_id=agent_id,
                                                                                    current_frame_idx=current_frame_idx)
        my_current_v_per_step = np.clip(my_current_v_per_step, a_min=0, a_max=7)
        goal_pt, goal_yaw = self.online_predictor.goal_setter.get_goal(current_data=current_state,
                                                                       agent_id=agent_id,
                                                                       dataset=self.dataset)
        if PRINT_TIMER:
            last_tic = time.perf_counter()
        if agent_id not in self.past_lanes:
            self.past_lanes[agent_id] = []
        if self.dataset == 'NuPlan' and is_ego:
            goal_lane, _, _ = plan_helper.find_closest_lane(
                current_state=current_state,
                my_current_pose=[goal_pt[0], goal_pt[1], -1, goal_yaw],
                valid_lane_types=self.valid_lane_types,
            )
            # current_route is a list of multiple routes to choose
            if len(current_route) == 0:
                lanes_in_route = []
                route_roadblocks = current_state['route'] if 'route' in current_state else None
                for each_block in route_roadblocks:
                    if each_block not in current_state['road']:
                        continue
                    lanes_in_route += current_state['road'][each_block]['lower_level']
                current_lanes, current_closest_pt_indices, dist_to_lane = plan_helper.find_closest_lane(
                    current_state=current_state,
                    my_current_pose=my_current_pose,
                    selected_lanes=lanes_in_route,
                    valid_lane_types=self.valid_lane_types,
                    excluded_lanes=self.past_lanes[agent_id]
                )
            else:
                selected_lanes = []
                for each_route in current_route:
                    selected_lanes += each_route
                current_lanes, current_closest_pt_indices, dist_to_lane = plan_helper.find_closest_lane(
                    current_state=current_state,
                    my_current_pose=my_current_pose,
                    selected_lanes=selected_lanes,
                    valid_lane_types=self.valid_lane_types,
                    excluded_lanes=self.past_lanes[agent_id]
                )
        else:
            if len(current_route) > 0:
                current_route = current_route[0]
            current_lanes, current_closest_pt_indices, dist_to_lane = plan_helper.find_closest_lane(
                current_state=current_state,
                my_current_pose=my_current_pose,
                selected_lanes=current_route,
                valid_lane_types=self.valid_lane_types,
                excluded_lanes=self.past_lanes[agent_id]
            )
        if dist_to_lane is not None:
            distance_threshold = max(self.frame_rate, max(self.frame_rate * my_current_v_per_step, dist_to_lane))
        else:
            dist_to_lane = 999
        self.current_on_road = not (dist_to_lane > OFF_ROAD_DIST)
        if self.dataset == 'NuPlan' and len(current_route) == 0 and is_ego:
            pass
            # route_roadblocks = current_state['route'] if 'route' in current_state else None
            # current_routes = self.set_route(road_dic=current_state['road'],
            #                                 goal_pt=[goal_pt[0], goal_pt[1], 0, goal_yaw], current_pose=my_current_pose,
            #                                 previous_routes=[current_route], max_number_of_routes=1,
            #                                 route_roadblock_check=route_roadblocks,
            #                                 agent_id=agent_id)
            # print(f"Got {len(current_routes)} for {agent_id} with {goal_pt} and {my_current_pose} given route {route_roadblocks}")
            # current_route = current_routes[0] if len(current_routes) > 0 else []
        else:
            if current_lanes in current_route and not isinstance(current_lanes, list):
                for each_past_lane in current_route[:current_route.index(current_lanes)]:
                    if each_past_lane not in self.past_lanes[agent_id]:
                        self.past_lanes[agent_id].append(each_past_lane)

        if isinstance(current_lanes, list):
            # deprecated
            lane_found_in_route = False
            for each_lane in current_lanes:
                if each_lane in current_route:
                    current_lane = each_lane
                    lane_found_in_route = True
                    break
            if not lane_found_in_route:
                current_lane = random.choice(current_lanes)
            idx = current_lanes.index(current_lane)
            current_closest_pt_idx = current_closest_pt_indices[idx]
        else:
            current_lane = current_lanes
            current_closest_pt_idx = current_closest_pt_indices

        if PRINT_TIMER:
            print(f"Time spent on first lane search:  {time.perf_counter() - last_tic:04f}s")
            last_tic = time.perf_counter()

        if self.dataset == 'NuPlan' and is_ego:
            # use route_roadblocks
            prior_lanes = []
            if current_lane is None:
                print("WARNING: Ego Current Lane not found")
        elif len(current_route) == 0:
            # get route from the original trajectory, this route does not have to be neither accurate nor connected
            prior_lanes = []
            org_closest_pt_idx = []
            for i in range(50):
                if i + current_frame_idx > 90:
                    break
                if i == 0:
                    continue
                if i % 10 != 0:
                    continue
                looping_pose, looping_v = get_current_pose_and_v(current_state=current_state,
                                                                 agent_id=agent_id,
                                                                 current_frame_idx=current_frame_idx + i)

                # looping_lane, looping_closest_idx, _, _ = self.find_closes_lane(current_state=current_state,
                #                                                                 agent_id=agent_id,
                #                                                                 my_current_v_per_step=looping_v,
                #                                                                 my_current_pose=looping_pose,
                #                                                                 no_unparallel=follow_org_route,
                #                                                                 return_list=False)

                looping_lane, looping_closest_idx, dist_to_lane = plan_helper.find_closest_lane(
                    current_state=current_state,
                    my_current_pose=looping_pose,
                    # include_unparallel=not follow_org_route
                    include_unparallel=False,
                    valid_lane_types=self.valid_lane_types,
                    excluded_lanes=self.past_lanes[agent_id]
                )

                if looping_lane is not None and looping_lane not in prior_lanes and dist_to_lane < 5:
                    prior_lanes.append(looping_lane)
                    org_closest_pt_idx.append(looping_closest_idx)

            if PRINT_TIMER:
                print(f"Time spent on loop lane search:  {time.perf_counter() - last_tic:04f}s")
                last_tic = time.perf_counter()
        else:
            prior_lanes = current_route

        # 2. find a spot to enter
        # Make connection with BC
        accum_dist = -0.0001
        p4 = None
        cuttin_lane_id = None
        cuttin_lane_idx = None
        first_lane = True

        def search_lanes(current_lane, route_roadblocks):
            result_lanes = []

            if goal_lane not in self.past_lanes['ego']:
                goal_roadblock = current_state['road'][goal_lane]['upper_level'][0]
                current_roadblock = current_state['road'][current_lane]['upper_level'][0]
                if goal_roadblock == current_roadblock:
                    current_lane = goal_lane

            lanes_to_loop = [[current_lane]]
            visited_lanes = [current_lane]

            while len(lanes_to_loop) > 0:
                looping_lanes = lanes_to_loop.pop()
                if len(looping_lanes) >= 3:
                    result_lanes.append(looping_lanes)
                    continue
                looping_lane = looping_lanes[-1]
                looping_roadblock = current_state['road'][looping_lane]['upper_level'][0]
                if looping_roadblock not in route_roadblocks:
                    continue
                # no lane changing
                # all_lanes_in_block = current_state['road'][looping_roadblock]['lower_level']
                # for each_lane in all_lanes_in_block:
                #     if each_lane not in visited_lanes:
                #         visited_lanes.append(each_lane)
                #         lanes_to_loop.append(looping_lanes[:-1]+[each_lane])
                next_lanes = current_state['road'][looping_lane]['next_lanes']
                for each_lane in next_lanes:
                    if each_lane not in visited_lanes:
                        visited_lanes.append(each_lane)
                        if each_lane not in current_state['road']:
                            result_lanes.append(looping_lanes+[each_lane])
                            continue
                        each_block = current_state['road'][each_lane]['upper_level'][0]
                        if each_block not in route_roadblocks:
                            continue
                        lanes_to_loop.append(looping_lanes+[each_lane])
                if len(lanes_to_loop) == 0 and len(looping_lanes) > 0:
                    result_lanes.append(looping_lanes)
            return result_lanes

        if self.dataset == 'NuPlan' and is_ego and current_lane is not None:
            route_roadblocks = current_state['route'] if 'route' in current_state else None
            current_upper_roadblock = current_state['road'][current_lane]['upper_level'][0]
            if current_upper_roadblock not in route_roadblocks:
                route_roadblocks += [current_upper_roadblock]
            while len(route_roadblocks) < 3:
                route_roadblocks.append(current_state['road'][route_roadblocks[-1]]['next_lanes'][0])
            # assumption: not far from current lane
            result_lanes = search_lanes(current_lane, route_roadblocks)

            if len(result_lanes) == 0:
                # choose a random lane from the first roadblock
                print("WARNING: No available route found")
                assert False, 'No Available Route Found for ego'

            result_traj = []
            for each_route in result_lanes:
                current_trajectory = None
                reference_trajectory = None
                reference_yaw = None
                for each_lane in each_route:
                    if reference_trajectory is None:
                        reference_trajectory = current_state['road'][each_lane]['xyz'][current_closest_pt_idx:, :2].copy()
                        reference_yaw = current_state['road'][each_lane]['dir'][current_closest_pt_idx:].copy()
                    else:
                        reference_trajectory = np.concatenate((reference_trajectory,
                                                               current_state['road'][each_lane]['xyz'][:, :2].copy()))
                        reference_yaw = np.concatenate((reference_yaw,
                                                        current_state['road'][each_lane]['dir'].copy()))
                # get CBC
                starting_index = int(my_current_v_per_step * self.frame_rate * 2)
                starting_index = min(starting_index, reference_trajectory.shape[0] - 1)
                p4 = reference_trajectory[starting_index, :2]
                starting_yaw = -utils.normalize_angle(reference_yaw[starting_index] + math.pi / 2)
                delta = euclidean_distance(p4, my_current_pose[:2]) / 4
                x, y = math.sin(starting_yaw) * delta + p4[0], math.cos(starting_yaw) * delta + p4[1]
                p3 = [x, y]

                p1 = my_current_pose[:2]
                yaw = - utils.normalize_angle(my_current_pose[3] + math.pi / 2)
                # delta = euclidean_distance(p4, my_current_pose[:2]) / 4
                delta = min(70/self.frame_rate, euclidean_distance(p4, my_current_pose[:2]) / 2)
                x, y = -math.sin(yaw) * delta + my_current_pose[0], -math.cos(yaw) * delta + my_current_pose[1]
                p2 = [x, y]
                if euclidean_distance(p4, p1) > 2:
                    if my_current_v_per_step < 1:
                        proper_v_for_cbc = (my_current_v_per_step + 1) / 2
                    else:
                        proper_v_for_cbc = my_current_v_per_step

                    connection_traj = self.trajectory_from_cubic_BC(p1=p1, p2=p2, p3=p3, p4=p4, v=proper_v_for_cbc)
                    current_trajectory = np.concatenate((connection_traj, reference_trajectory[starting_index:, :2]))
                else:
                    current_trajectory = reference_trajectory[starting_index:, :2]
                result_traj.append(current_trajectory)
                current_state['predicting']['trajectory_to_mark'].append(current_trajectory)

            self.routed_traj[agent_id] = result_traj
            return self.routed_traj[agent_id], result_lanes

        if current_lane is not None:
            current_looping_lane = current_lane
            while_counter = 0
            if distance_threshold > 100:
                print("Closest lane detection failded: ", agent_id, current_looping_lane, distance_threshold, my_current_v_per_step, dist_to_lane, current_route)
            else:
                distance_threshold = max(distance_threshold, self.frame_rate * my_current_v_per_step)

                while accum_dist < distance_threshold and distance_threshold <= 100:
                    if while_counter > 100:
                        print("ERROR: Infinite looping lanes")
                        break


                    while_counter += 1
                    # turning: 1=left turn, 2=right turn, 3=UTurn
                    # UTurn -> Skip
                    # Left/Right check distance, if < 15 then skip, else not skip
                    current_looping_lane_turning = current_state['road'][current_looping_lane]['turning']
                    if dynamic_turnings and current_looping_lane_turning == 3 or (current_looping_lane_turning in [1, 2] and euclidean_distance(current_state['road'][current_looping_lane]['xyz'][-1, :2], my_current_pose[:2]) < 15):
                        # skip turning lanes
                        # accum_dist = distance_threshold - 0.1
                        pass
                    elif while_counter > 50:
                        print("Inifinite looping lanes (agent_id/current_lane): ", agent_id, current_looping_lane)
                        accum_dist = distance_threshold - 0.1
                    else:
                        if first_lane:
                            road_xy = current_state['road'][current_looping_lane]['xyz'][current_closest_pt_idx:, :2].copy()
                        else:
                            road_xy = current_state['road'][current_looping_lane]['xyz'][:, :2].copy()
                        for j, each_xy in enumerate(road_xy):
                            if j == 0:
                                continue
                            accum_dist += euclidean_distance(each_xy, road_xy[j - 1])
                            if accum_dist >= distance_threshold:
                                p4 = each_xy
                                if first_lane:
                                    yaw = - utils.normalize_angle(
                                        current_state['road'][current_looping_lane]['dir'][j + current_closest_pt_idx] + math.pi / 2)
                                else:
                                    yaw = - utils.normalize_angle(
                                        current_state['road'][current_looping_lane]['dir'][j] + math.pi / 2)
                                delta = euclidean_distance(p4, my_current_pose[:2]) / 4
                                x, y = math.sin(yaw) * delta + p4[0], math.cos(yaw) * delta + p4[1]
                                p3 = [x, y]
                                cuttin_lane_id = current_looping_lane
                                if first_lane:
                                    cuttin_lane_idx = j + current_closest_pt_idx
                                else:
                                    cuttin_lane_idx = j
                                break

                    if p4 is None:
                        if current_looping_lane in prior_lanes and current_looping_lane != prior_lanes[-1]:
                            # if already has route, then use previous route
                            current_lane_route_idx = prior_lanes.index(current_looping_lane)
                            current_looping_lane = prior_lanes[current_lane_route_idx+1]
                        else:
                            # if not, try to loop a new route
                            next_lanes = current_state['road'][current_looping_lane]['next_lanes']
                            next_lane_found = False
                            if follow_org_route:
                                if current_looping_lane in prior_lanes:  # True:
                                    # follow original lanes
                                    current_idx = prior_lanes.index(current_looping_lane)
                                    if current_idx < len(prior_lanes) - 1:
                                        next_lane = prior_lanes[current_idx + 1]
                                        next_lane_found = True
                                        if next_lane in next_lanes:
                                            # next lane connected, loop this next lane and continue next loop
                                            current_looping_lane = next_lane
                                        else:
                                            # next lane not connected
                                            # 1. find closest point
                                            road_xy = current_state['road'][current_looping_lane]['xyz'][:, :2].copy()
                                            closest_dist = 999999
                                            closest_lane_idx = None
                                            turning_yaw = None
                                            for j, each_xy in enumerate(road_xy):
                                                dist = euclidean_distance(each_xy[:2], my_current_pose[:2])
                                                if dist < closest_dist:
                                                    closest_lane_idx = j
                                                    closest_dist = dist
                                                    turning_yaw = utils.normalize_angle(my_current_pose[3] - current_state['road'][current_looping_lane]['dir'][j])
                                            if closest_lane_idx is None:
                                                # follow no next lane logic below
                                                next_lane_found = False
                                            else:
                                                max_turning_dist = 120 / math.pi
                                                if closest_dist >= max_turning_dist:
                                                    # too far for max turning speed 15m/s
                                                    if turning_yaw > math.pi / 2:
                                                        # turn towards target lane first on the right
                                                        yaw = - utils.normalize_angle(my_current_pose[3] + math.pi / 2) + math / 2
                                                        delta = 180 / math.pi
                                                        x, y = math.sin(yaw) * delta + my_current_pose[0], math.cos(yaw) * delta + my_current_pose[1]
                                                        p4 = [x, y]
                                                        yaw = yaw - math / 2
                                                        delta = delta / 2
                                                        x, y = math.sin(yaw) * delta + my_current_pose[0], math.cos(yaw) * delta + my_current_pose[1]
                                                        p3 = [x, y]
                                                        break
                                                    if turning_yaw <= math.pi / 2:
                                                        # turn towards target lane first on the right
                                                        yaw = - utils.normalize_angle(my_current_pose[3] + math.pi / 2) - math / 2
                                                        delta = 180 / math.pi
                                                        x, y = math.sin(yaw) * delta + my_current_pose[0], math.cos(yaw) * delta + my_current_pose[1]
                                                        p4 = [x, y]
                                                        yaw = yaw + math / 2
                                                        delta = delta / 2
                                                        x, y = math.sin(yaw) * delta + my_current_pose[0], math.cos(yaw) * delta + my_current_pose[1]
                                                        p3 = [x, y]
                                                        break
                                                else:
                                                    accum_dist = distance_threshold - 0.1

                            if not next_lane_found:
                                # follow prior or choose a random one as the next
                                if len(next_lanes) > 0:
                                    current_looping_lane_changes = False
                                    for each_lane in next_lanes:
                                        if each_lane in prior_lanes:
                                            current_looping_lane = each_lane
                                            current_looping_lane_changes = True
                                    if not current_looping_lane_changes:
                                        # random choose one lane as route
                                        current_looping_lane = random.choice(next_lanes)
                                else:
                                    print("warning: no next lane found with breaking the lane finding loop")
                                    break
                                    # return
                    else:
                        break
                    first_lane = False

        if PRINT_TIMER:
            print(f"Time spent on while loop:  {time.perf_counter() - last_tic:04f}s")
            last_tic = time.perf_counter()

        if p4 is None:
            # not found any lane at all, generate a linear line forward
            # 3. gennerate p1 and p2
            p1 = my_current_pose[:2]
            yaw = - utils.normalize_angle(my_current_pose[3] + math.pi / 2)
            delta = self.planning_horizon
            x, y = -math.sin(yaw) * delta + my_current_pose[0], -math.cos(yaw) * delta + \
                   my_current_pose[1]
            p2 = [x, y]
            p3 = p2
            x, y = -math.sin(yaw) * delta + p2[0], -math.cos(yaw) * delta + p2[1]
            p4 = [x, y]
            # 4. generate a curve with cubic BC
            if my_current_v_per_step < 1:
                proper_v_for_cbc = (my_current_v_per_step + 1) / 2
            else:
                proper_v_for_cbc = my_current_v_per_step
            if euclidean_distance(p4, p1) > 1:
                print(f"No lanes found for route of {agent_id} {proper_v_for_cbc} {my_current_pose}")
                connection_traj = self.trajectory_from_cubic_BC(p1=p1, p2=p2, p3=p3, p4=p4, v=proper_v_for_cbc)
            else:
                assert False, f"Error: P4, P1 overlapping {p4, p1}"
            assert connection_traj.shape[0] > 0, connection_traj.shape
            self.routed_traj[agent_id] = connection_traj
        else:
            assert cuttin_lane_id is not None
            # 3. gennerate p1 and p2
            p1 = my_current_pose[:2]
            yaw = - utils.normalize_angle(my_current_pose[3] + math.pi / 2)
            # delta = euclidean_distance(p4, my_current_pose[:2]) / 4
            delta = min(7, euclidean_distance(p4, my_current_pose[:2]) / 2)
            x, y = -math.sin(yaw) * delta + my_current_pose[0], -math.cos(yaw) * delta + \
                   my_current_pose[1]
            p2 = [x, y]

            if my_current_v_per_step < 1:
                proper_v_for_cbc = (my_current_v_per_step + 1) / 2
            else:
                proper_v_for_cbc = my_current_v_per_step

            connection_traj = self.trajectory_from_cubic_BC(p1=p1, p2=p2, p3=p3, p4=p4, v=proper_v_for_cbc)
            # loop out a route
            current_looping_lane = cuttin_lane_id
            lanes_in_a_route = [current_looping_lane]
            route_traj_left = np.array(current_state['road'][current_looping_lane]['xyz'][cuttin_lane_idx:, :2], ndmin=2)
            next_lanes = current_state['road'][current_looping_lane]['next_lanes']
            while len(next_lanes) > 0 and len(lanes_in_a_route) < 10:
                any_lane_in_route = False
                if len(prior_lanes) > 0:
                    for each_next_lane in next_lanes:
                        if each_next_lane in prior_lanes:
                            any_lane_in_route = True
                            current_looping_lane = each_next_lane
                            break
                if not any_lane_in_route:
                    # try to follow original route
                    current_lane_changed = False
                    lanes_to_choose = []
                    for each_next_lane in next_lanes:
                        if each_next_lane in prior_lanes:
                            current_looping_lane = each_next_lane
                            current_lane_changed = True
                            break
                        if each_next_lane in current_state['road']:
                            lanes_to_choose.append(each_next_lane)
                    if current_lane_changed:
                        pass
                    elif len(lanes_to_choose) == 0:
                        print("NO VALID NEXT LANE TO CHOOSE from env_planner for ", agent_id)
                        break
                    else:
                        # random choose one lane as route
                        current_looping_lane = random.choice(lanes_to_choose)

                # amend route manually for scenario 54 file 00000
                # if current_looping_lane == 109:
                #     current_looping_lane = 112
                # if current_looping_lane == 131:
                #     current_looping_lane = 132
                if current_looping_lane not in current_state['road']:
                    print("selected lane not found in road dic")
                    break
                lanes_in_a_route.append(current_looping_lane)
                next_lanes = current_state['road'][current_looping_lane]['next_lanes']
                # route_traj_left = np.concatenate(
                #     (route_traj_left, current_state['road'][current_looping_lane]['xyz'][:, :2]))
                route_traj_left = np.concatenate(
                    (route_traj_left, current_state['road'][current_looping_lane]['xyz'][10:, :2]))  # start with a margin to avoid overlapping ends and starts
            if len(current_route) == 0:
                # initiation the route and return
                current_route = lanes_in_a_route
                if is_ego:
                    goal_pt, goal_yaw = self.online_predictor.goal_setter.get_goal(current_data=current_state,
                                                                                   agent_id=agent_id,
                                                                                   dataset=self.dataset)
                    assert goal_pt is not None and goal_yaw is not None, goal_pt
                    ending_lane, ending_lane_idx, dist_to_ending_lane = plan_helper.find_closest_lane(
                        current_state=current_state,
                        my_current_pose=[goal_pt[0], goal_pt[1], 0, goal_yaw],
                        valid_lane_types=self.valid_lane_types
                    )

                    if ending_lane is not None:
                        if dist_to_ending_lane > 30:
                            logging.warning('Goal Point Off Road')
                        self.target_lanes = [ending_lane, ending_lane_idx]

                        if ending_lane not in lanes_in_a_route:
                            back_looping_counter = 0
                            back_to_loop_lanes = [ending_lane]
                            target_lane = ending_lane
                            while back_looping_counter < 10:
                                back_looping_counter += 1
                                current_back_looping_lane = back_to_loop_lanes.pop()
                                _, _, distance_to_ending_lane = plan_helper.find_closest_lane(
                                    current_state=current_state,
                                    my_current_pose=my_current_pose,
                                    selected_lanes=[current_back_looping_lane],
                                    valid_lane_types=self.valid_lane_types
                                )
                                if distance_to_ending_lane < OFF_ROAD_DIST:
                                    target_lane = current_back_looping_lane
                                    break
                                else:
                                    prev_lanes = current_state['road'][current_back_looping_lane]['previous_lanes']
                                    if not isinstance(prev_lanes, list):
                                        prev_lanes = prev_lanes.tolist()
                                    if len(prev_lanes) == 0:
                                        break
                                    back_to_loop_lanes += prev_lanes

                            current_route = [target_lane]
                    else:
                        logging.warning('No Lane Found for Goal Point at all')

            route_traj_left = np.array(route_traj_left, ndmin=2)
            # 4. generate a curve with cubic BC
            if euclidean_distance(p4, p1) > 2:
                if len(route_traj_left.shape) < 2:
                    print(route_traj_left.shape, route_traj_left)
                    self.routed_traj[agent_id] = connection_traj
                else:
                    if euclidean_distance(p4, p1) > 1 and len(connection_traj.shape) > 0 and connection_traj.shape[0] > 1:
                        # concatenate org_traj, connection_traj, route_traj_left
                        self.routed_traj[agent_id] = np.concatenate(
                            (connection_traj, route_traj_left))
                    else:
                        self.routed_traj[agent_id] = route_traj_left
            else:
                self.routed_traj[agent_id] = route_traj_left

        if PRINT_TIMER:
            print(f"Time spent on CBC:  {time.perf_counter() - last_tic:04f}s")
            last_tic = time.perf_counter()
        if DRAW_CBC_PTS:
            current_state['predicting']['mark_pts'] = [p4, p3, p2, p1]
        if is_ego:
            if self.dataset == 'NuPlan':
                return [self.routed_traj[agent_id]], current_route
            else:
                return [self.routed_traj[agent_id]], [current_route]
        else:
            return self.routed_traj[agent_id], current_route

    def adjust_speed_for_collision(self, interpolator, distance_to_end, current_v, end_point_v, reschedule_speed_profile=False):
        # constant deceleration
        time_to_collision = min(self.planning_horizon, distance_to_end / (current_v + end_point_v + 0.0001) * 2)
        time_to_decelerate = abs(current_v - end_point_v) / (0.1/self.frame_rate)
        traj_to_return = []
        desired_deceleration = 0.2 /self.frame_rate
        if time_to_collision < time_to_decelerate:
            # decelerate more than 3m/ss
            deceleration = (end_point_v - current_v) / time_to_collision
            dist_travelled = 0
            for i in range(int(time_to_collision)):
                current_v += deceleration * 1.2
                current_v = max(0, current_v)
                dist_travelled += current_v
                traj_to_return.append(interpolator.interpolate(dist_travelled))
            current_len = len(traj_to_return)
            while current_len < 100:
                dist_travelled += current_v
                traj_to_return.append(interpolator.interpolate(dist_travelled))
                current_len = len(traj_to_return)
        else:
            # decelerate with 2.5m/ss
            time_for_current_speed = np.clip(((distance_to_end - 3 - (current_v+end_point_v)/2*time_to_decelerate) / (current_v + 0.0001)), 0, self.frame_rate*self.frame_rate)
            dist_travelled = 0
            if time_for_current_speed > 1:
                for i in range(int(time_for_current_speed)):
                    if reschedule_speed_profile:
                        dist_travelled += current_v
                    else:
                        if i == 0:
                            dist_travelled += current_v
                        elif i >= interpolator.trajectory.shape[0]:
                            dist_travelled += current_v
                        else:
                            current_v_hat = interpolator.get_speed_with_index(i)
                            if abs(current_v_hat - current_v) > 2 / self.frame_rate:
                                print("WARNING: sharp speed changing", current_v, current_v_hat)
                            current_v = current_v_hat
                            dist_travelled += current_v
                    traj_to_return.append(interpolator.interpolate(dist_travelled))
            for i in range(int(time_to_decelerate)):
                current_v -= desired_deceleration
                current_v = max(0, current_v)
                dist_travelled += current_v
                traj_to_return.append(interpolator.interpolate(dist_travelled))
            current_len = len(traj_to_return)
            while current_len < 100:
                dist_travelled += current_v
                traj_to_return.append(interpolator.interpolate(dist_travelled))
                current_len = len(traj_to_return)
        if len(traj_to_return) > 0:
            short = self.planning_horizon - len(traj_to_return)
            for _ in range(short):
                traj_to_return.append(traj_to_return[-1])
        else:
            for _ in range(self.planning_horizon):
                traj_to_return.append(interpolator.interpolate(0))
        return np.array(traj_to_return, ndmin=2)

    def get_traffic_light_collision_pts(self, current_state, current_frame_idx,
                                        continue_time_threshold=5):
        tl_dics = current_state['traffic_light']
        road_dics = current_state['road']
        traffic_light_ending_pts = []
        for lane_id in tl_dics.keys():
            if lane_id == -1:
                continue
            tl = tl_dics[lane_id]
            # get the position of the end of this lane
            # Unknown = 0, Arrow_Stop = 1, Arrow_Caution = 2, Arrow_Go = 3, Stop = 4, Caution = 5, Go = 6, Flashing_Stop = 7, Flashing_Caution = 8
            try:
                tl_state = tl["state"][current_frame_idx]
            except:
                tl_state = tl["state"][0]

            if tl_state in [1, 4, 7]:
                end_of_tf_checking = min(len(tl["state"]), current_frame_idx + continue_time_threshold)
                all_red = True
                for k in range(current_frame_idx, end_of_tf_checking):
                    if tl["state"][k] not in [1, 4, 7]:
                        all_red = False
                        break
                if all_red:
                    for seg_id in road_dics.keys():
                        if lane_id == seg_id:
                            road_seg = road_dics[seg_id]
                            if self.dataset == 'Waymo':
                                if road_seg["type"] in [1, 2, 3]:
                                    if len(road_seg["dir"].shape) < 1:
                                        continue
                                    if road_seg['turning'] == 1 and tl_state in [4, 7]:
                                        # can do right turn with red light
                                        continue
                                    end_point = road_seg["xyz"][0][:2]
                                    traffic_light_ending_pts.append(end_point)
                                break
                            elif self.dataset == 'NuPlan':
                                end_point = road_seg["xyz"][0][:2]
                                traffic_light_ending_pts.append(end_point)
                                break
                            else:
                                assert False, f'Unknown dataset in env planner - {self.dataset}'
        return traffic_light_ending_pts

    def check_past_goal(self, traj, current_idx, current_state, agent_id):
        # if 'follow_goal' in current_state['predicting'] and agent_id in current_state['predicting']['follow_goal'] and not current_state['predicting']['follow_goal'][agent_id]:
        #     return True
        # detect by angle
        index = 1
        valid = abs(current_state['predicting']['original_trajectory'][agent_id]['pose'][-1, :2][0] + 1) > 0.01
        while not valid:
            index += 1
            valid = abs(current_state['predicting']['original_trajectory'][agent_id]['pose'][-index, :2][0] + 1) > 0.01
        original_goal = current_state['predicting']['original_trajectory'][agent_id]['pose'][-index, :2]

        total_frame = traj.shape[0]
        if current_idx + self.planning_interval * 2 > total_frame - 1 or current_idx + self.planning_interval + self.frame_rate > total_frame - 1:
            return False

        next_checking_pt = traj[current_idx+self.planning_interval*2, :2]
        angle_to_goal = get_angle_of_a_line(next_checking_pt, original_goal)
        goal_yaw = current_state['predicting']['original_trajectory'][agent_id]['pose'][-1, 3]
        past_goal = False
        normalized_angle = utils.normalize_angle(angle_to_goal - goal_yaw)
        if normalized_angle > math.pi / 2 or normalized_angle < -math.pi / 2:
            past_goal = True
        # detect by distance for low speed trajectories
        two_point_dist = euclidean_distance(original_goal, next_checking_pt)
        if two_point_dist < MINIMAL_DISTANCE_TO_GOAL:
            past_goal = True
        # goal_distance2 = euclidean_distance(marginal_traj[self.planning_interval + 20, :2], origial_goal)
        two_point_dist = euclidean_distance(traj[current_idx+self.planning_interval, :2],
                                            traj[current_idx+self.planning_interval+self.frame_rate, :2])
        if two_point_dist < MINIMAL_SPEED_TO_TRACK_ORG_GOAL:
            past_goal = True
        if past_goal:
            current_state['predicting']['follow_goal'][agent_id] = False
        else:
            current_state['predicting']['follow_goal'][agent_id] = True
        return past_goal

    def get_trajectory_from_interpolator(self, my_interpolator, my_current_speed, a_per_step=None,
                                         check_turning_dynamics=True, desired_speed=7,
                                         emergency_stop=False, hold_still=False,
                                         agent_id=None, a_scale_turning=0.7, a_scale_not_turning=0.9):
        total_frames = self.planning_horizon
        total_pts_in_interpolator = my_interpolator.trajectory.shape[0]
        trajectory = np.ones((total_frames, 4)) * -1
        # get proper speed for turning
        largest_yaw_change = -1
        largest_yaw_change_idx = None
        if check_turning_dynamics and not emergency_stop:
            for i in range(min(200, total_pts_in_interpolator - 2)):
                if my_interpolator.trajectory[i, 0] == -1.0 or my_interpolator.trajectory[i+1, 0] == -1.0 or my_interpolator.trajectory[i+2, 0] == -1.0:
                    continue
                current_yaw = utils.normalize_angle(get_angle_of_a_line(pt1=my_interpolator.trajectory[i, :2], pt2=my_interpolator.trajectory[i+1, :2]))
                next_yaw = utils.normalize_angle(get_angle_of_a_line(pt1=my_interpolator.trajectory[i+1, :2], pt2=my_interpolator.trajectory[i+2, :2]))
                dist = utils.euclidean_distance(pt1=my_interpolator.trajectory[i, :2], pt2=my_interpolator.trajectory[i+1, :2])
                yaw_diff = abs(utils.normalize_angle(next_yaw - current_yaw))
                if yaw_diff > largest_yaw_change and 0.04 < yaw_diff < math.pi / 2 * 0.9 and 100 > dist > 0.3:
                    largest_yaw_change = yaw_diff
                    largest_yaw_change_idx = i
            proper_speed_minimal = max(5, math.pi / 3 / largest_yaw_change)  # calculate based on 20m/s turning for 12s a whole round with a 10hz data in m/s
            proper_speed_minimal_per_frame = proper_speed_minimal / self.frame_rate
            if largest_yaw_change_idx is not None:
                deceleration_frames = max(0, largest_yaw_change_idx - abs(my_current_speed - proper_speed_minimal_per_frame) / (A_SLOWDOWN_DESIRE / self.frame_rate / self.frame_rate / 2))
            else:
                deceleration_frames = 99999
        if agent_id is not None:
            pass
        dist_past = 0
        current_speed = my_current_speed
        for i in range(total_frames):
            if current_speed < 0.1:
                low_speed_a_scale = 1 * self.frame_rate
            else:
                low_speed_a_scale = 0.1 * self.frame_rate
            if hold_still:
                trajectory[i] = my_interpolator.interpolate(0)
                continue
            elif emergency_stop:
                current_speed -= A_SLOWDOWN_DESIRE / self.frame_rate
            elif largest_yaw_change_idx is not None:
                proper_speed_minimal_per_frame = max(0.5, min(proper_speed_minimal_per_frame, 5))
                if largest_yaw_change_idx >= i >= deceleration_frames:
                    if current_speed > proper_speed_minimal_per_frame:
                        current_speed -= A_SLOWDOWN_DESIRE / self.frame_rate / 2
                    else:
                        current_speed += A_SPEEDUP_DESIRE / self.frame_rate * a_scale_not_turning * low_speed_a_scale
                elif i < deceleration_frames:
                    if current_speed < desired_speed / 4.7:
                        # if far away from the turnings and current speed is smaller than 15m/s, then speed up
                        # else keep current speed
                        if a_per_step is not None:
                            current_speed += max(-A_SLOWDOWN_DESIRE / self.frame_rate, min(A_SPEEDUP_DESIRE / self.frame_rate * low_speed_a_scale, a_per_step))
                        else:
                            current_speed += A_SPEEDUP_DESIRE / self.frame_rate * a_scale_turning * low_speed_a_scale
                elif i > largest_yaw_change_idx:
                    if current_speed > proper_speed_minimal_per_frame:
                        current_speed -= A_SLOWDOWN_DESIRE / self.frame_rate
                    else:
                        if a_per_step is not None:
                            current_speed += max(-A_SLOWDOWN_DESIRE / self.frame_rate, min(A_SPEEDUP_DESIRE / self.frame_rate * low_speed_a_scale, a_per_step))
                        else:
                            current_speed += A_SPEEDUP_DESIRE / self.frame_rate * a_scale_turning * low_speed_a_scale
            else:
                if current_speed < desired_speed:
                    if a_per_step is not None:
                        current_speed += max(-A_SLOWDOWN_DESIRE / self.frame_rate, min(A_SPEEDUP_DESIRE / self.frame_rate * low_speed_a_scale, a_per_step))
                    else:
                        current_speed += A_SPEEDUP_DESIRE / self.frame_rate * a_scale_not_turning * low_speed_a_scale  # accelerate with 0.2 of desired acceleration
            current_speed = max(0, current_speed)
            dist_past += current_speed
            trajectory[i] = my_interpolator.interpolate(dist_past)
        return trajectory

    def update_env_trajectory_reguild(self, current_frame_idx, relevant_only=True,
                                      current_state=None, plan_for_ego=False, dynamic_env=True):
        """
        plan and update trajectory to commit for relevant environment agents
        current_frame_idx: 1,2,3,...,11(first frame to plan)
        """
        # if self.online_predictor.prediction_data is None:
        #     logging.warning('Skip planning: Planning before making a prediction')
        #     return

        if not dynamic_env:
            return current_state

        # self.scenario_frame_number = current_frame_idx
        # frame_diff = self.scenario_frame_number - self.planning_from

        if self.is_planning(current_frame_idx):
        # if frame_diff >= 0 and frame_diff % self.planning_interval == 0:
            # load scenario data
            if current_state is None:
                return
            agents = current_state['agent']
            relevant_agents = current_state['predicting']['relevant_agents']
            edges = current_state['predicting']['relation']
            # XPts = current_state['predicting']['XPt']
            # select marginal prediction traj
            # prediction_traj_dic_m = current_state['predicting']['marginal_trajectory']

            # prediction_traj_dic_c = current_state['predicting']['conditional_trajectory']
            # prediction_traj_dic_m = prediction_traj_dic_c
            ego_id = current_state['predicting']['ego_id'][1]
            agents_dic_copy = copy.deepcopy(current_state['agent'])

            for agent_id in agents:
                # loop each relevant agent
                if relevant_only and agent_id not in relevant_agents:
                    continue

                current_state['agent'][agent_id]['action'] = None
                total_time_frame = current_state['agent'][agent_id]['pose'].shape[0]
                goal_point = current_state['predicting']['goal_pts'][agent_id]
                my_current_pose = current_state['agent'][agent_id]['pose'][current_frame_idx - 1]
                my_current_v_per_step = euclidean_distance(current_state['agent'][agent_id]['pose'][current_frame_idx - 1, :2],
                                                           current_state['agent'][agent_id]['pose'][current_frame_idx - 6, :2])/5
                my_target_speed = 70 / self.frame_rate

                if my_current_v_per_step > 100 / self.frame_rate:
                    my_current_v_per_step = 10 / self.frame_rate
                org_pose = current_state['predicting']['original_trajectory'][agent_id]['pose'].copy()

                # for non-vehicle types agent, skip
                if int(current_state['agent'][agent_id]['type']) not in self.vehicle_types:
                    continue

                # rst = prediction_traj_dic_m[agent_id]['rst']
                # score = np.exp(prediction_traj_dic_m[agent_id]['score'])
                # score /= np.sum(score)
                # best_idx = np.argmax(score)
                # prediction_traj_m = rst[best_idx]

                # use_rules = 0  # 0=hybird, 1=use rules only
                # info: always use rules for env agents
                use_rules = not self.follow_prediction_traj
                if use_rules:
                    # past_goal = self.check_past_goal(traj=current_state['agent'][agent_id]['pose'],
                    #                                  current_idx=current_frame_idx,
                    #                                  current_state=current_state,
                    #                                  agent_id=agent_id)
                    my_traj, _ = self.get_reroute_traj(current_state=current_state,
                                                       agent_id=agent_id,
                                                       current_frame_idx=current_frame_idx)
                else:
                    routed_traj, _ = self.get_reroute_traj(current_state=current_state,
                                                           agent_id=agent_id,
                                                           current_frame_idx=current_frame_idx)
                    marginal_trajs = current_state['predicting']['marginal_trajectory'][agent_id]['rst'][0]
                    x_dist = []
                    for r_p in routed_traj[:50, :2]:
                        line_dist = []
                        for m_p in marginal_trajs[:50, :2]:
                            dist = euclidean_distance(r_p, m_p)
                            line_dist.append(dist)
                        x_dist.append(min(line_dist))
                    minimal_distance = max(x_dist)
                    if True:
                    # if minimal_distance < 3:
                        my_traj = marginal_trajs
                    else:
                        my_traj = routed_traj

                # current_state['predicting']['routed_trajectory'][agent_id]
                # if False:
                #     # use prediction trajectory
                #     target_lanes = org_pose
                #     if agent_id in current_state['lanes_traveled']:
                #         lane_traveled_list = current_state['lanes_traveled'][agent_id]
                #         if len(lane_traveled_list) > 0:
                #             for i, each_lane_id in enumerate(lane_traveled_list):
                #                 if i == 0:
                #                     target_lanes = current_state['road'][each_lane_id]['xyz'][:, :2].copy()
                #                 else:
                #                     target_lanes = np.concatenate(
                #                         (target_lanes, current_state['road'][each_lane_id]['xyz'][:, :2])).copy()
                #     prediction_traj_m, follow_org = self.select_trajectory_from_prediction(prediction_traj_dic_m, agent_id,
                #                                                                            goal_point,
                #                                                                            original_trajectory=target_lanes, #org_pose,
                #                                                                            remaining_frames=min(10, total_time_frame - current_frame_idx),
                #                                                                            follow_goal=
                #                                                                            current_state['predicting'][
                #                                                                                'follow_goal'][
                #                                                                                agent_id],
                #                                                                            follow_original_as_default=follow_org_as_default)
                #     assert prediction_traj_m is not None, f'{agent_id} / {relevant_agents}'
                #     action = 0  # 0=No Action, 1=Follow, 2=Yield
                #     my_traj = prediction_traj_m.copy()

                # detect trajectory collisions
                # after collision detection, we have earliest_collision_idx, earliest_target_id, latest_collision_idx(for that earliest collision detected
                my_interpolator = SudoInterpolator(my_traj.copy(), my_current_pose)
                interpolated_trajectory = self.get_trajectory_from_interpolator(my_interpolator=my_interpolator,
                                                                                my_current_speed=my_current_v_per_step,
                                                                                agent_id=agent_id)
                my_interpolator = SudoInterpolator(interpolated_trajectory.copy(), my_current_pose)

                earliest_collision_idx = None
                earliest_target_agent = None
                collision_point = None

                traffic_light_ending_pts = self.get_traffic_light_collision_pts(current_state=current_state,
                                                                                current_frame_idx=current_frame_idx)
                tl_checked = False
                running_red_light = False

                if self.method_testing < 1:
                    continue

                # check collisions for ego from frame 1 of the prediction trajectory
                ego_index_checking = 1  # current_frame_idx+1
                collision_detected_now = False
                latest_collision_id = None
                end_checking_frame = np.clip(current_frame_idx + REACTION_AFTER, 0, total_time_frame)
                end_checking_frame = min(end_checking_frame, current_frame_idx+self.planning_horizon)
                # pack an Agent object for collision detection
                my_reactors = []
                for i in range(current_frame_idx, end_checking_frame):
                    ego_index_checking = i - current_frame_idx
                    ego_pose2_valid = False
                    if i - current_frame_idx > 0:
                        ego_pose2 = interpolated_trajectory[ego_index_checking - 1]
                        if abs(ego_pose2[0]) < 1.1 and abs(ego_pose2[1]) < 1.1:
                            pass
                        else:
                            ego_agent2 =Agent(x=(ego_pose2[0] + ego_pose[0]) / 2,
                                               y=(ego_pose2[1] + ego_pose[1]) / 2,
                                               yaw=get_angle_of_a_line(ego_pose2[:2], ego_pose[:2]),
                                               length=euclidean_distance(ego_pose2[:2], ego_pose[:2]),
                                               width=max(1, current_state['agent'][agent_id]['shape'][0][0]),
                                               agent_id=agent_id)
                            ego_pose2_valid = True

                    for each_other_agent in agents:
                        if each_other_agent == agent_id:
                            continue
                        if each_other_agent in my_reactors:
                            continue
                        if current_state['agent'][each_other_agent]['shape'][0][1] == -1:
                            continue
                        if ego_index_checking >= interpolated_trajectory.shape[0]:
                            continue
                        ego_pose = interpolated_trajectory[ego_index_checking, :]  # ego start checking from frame 0
                        if abs(ego_pose[0]) < 1.1 and abs(ego_pose[1]) < 1.1:
                            # print("WARNING invalid pose for collision detection: ", pose_in_pred)
                            continue
                        ego_agent =Agent(x=ego_pose[0],
                                          y=ego_pose[1],
                                          yaw=ego_pose[3],
                                          length=max(1, current_state['agent'][agent_id]['shape'][0][1]),
                                          width=max(1, current_state['agent'][agent_id]['shape'][0][0]),
                                          agent_id=agent_id)

                        # check traffic light violation
                        for tl_pt in traffic_light_ending_pts:
                            dummy_tf_agent = Agent(x=tl_pt[0], y=tl_pt[1], yaw=0,
                                                   length=TRAFFIC_LIGHT_COLLISION_SIZE,
                                                   width=TRAFFIC_LIGHT_COLLISION_SIZE, agent_id=99999)
                            running = utils.check_collision(
                                checking_agent=ego_agent,
                                target_agent=dummy_tf_agent)
                            if ego_pose2_valid:
                                running |= utils.check_collision(
                                    checking_agent=ego_agent2,
                                    target_agent=dummy_tf_agent)
                            if running:
                                running_red_light = True
                                earliest_collision_idx = ego_index_checking
                                collision_point = [ego_pose[0], ego_pose[1]]
                                earliest_target_agent = 99999
                                target_speed = 0
                                # break collision detection
                                break

                        if running_red_light:
                            to_yield = True
                            break

                        each_other_agent_pose_array = current_state['agent'][each_other_agent]['pose']
                        target_current_pose = each_other_agent_pose_array[i]
                        target_agent =Agent(x=target_current_pose[0],
                                             y=target_current_pose[1],
                                             yaw=target_current_pose[3],
                                             length=max(1, current_state['agent'][each_other_agent]['shape'][0][1]),
                                             width=max(1, current_state['agent'][each_other_agent]['shape'][0][0]),
                                             agent_id=each_other_agent)
                        has_collision = utils.check_collision(checking_agent=ego_agent,
                                                              target_agent=target_agent)
                        if ego_pose2_valid:
                            has_collision |= utils.check_collision(checking_agent=ego_agent2,
                                                                   target_agent=target_agent)
                        to_yield = False
                        if has_collision:
                            to_yield = True
                            # solve this conflict
                            found_in_loaded = False
                            if self.follow_loaded_relation:
                                detected_relation = []
                                for edge in current_state['edges']:
                                    if agent_id == edge[0] and each_other_agent == edge[1]:
                                        to_yield = False
                                        found_in_loaded = True
                                        break
                                current_state['predicting']['relation'] += [agent_id, each_other_agent]
                            if not found_in_loaded:
                                # FORWARD COLLISION CHECKINGS
                                target_pose_0 = each_other_agent_pose_array[current_frame_idx]
                                target_agent_0 =Agent(x=target_pose_0[0],
                                                       y=target_pose_0[1],
                                                       yaw=target_pose_0[3],
                                                       length=max(1, current_state['agent'][each_other_agent]['shape'][0][1]),
                                                       width=max(1, current_state['agent'][each_other_agent]['shape'][0][0]),
                                                       agent_id=each_other_agent)
                                collision_0 = utils.check_collision(ego_agent, target_agent_0)
                                if ego_pose2_valid:
                                    collision_0 |= utils.check_collision(ego_agent2, target_agent_0)
                                if collision_0:
                                    # yield
                                    detected_relation = [[each_other_agent, agent_id]]
                                else:
                                    # FCC backwards
                                    ego_agent_0 =Agent(
                                        x=interpolated_trajectory[0, 0],
                                        y=interpolated_trajectory[0, 1],
                                        yaw=interpolated_trajectory[0, 3],
                                        length=max(1, current_state['agent'][agent_id]['shape'][0][1]),
                                        width=max(1, current_state['agent'][agent_id]['shape'][0][0]),
                                        agent_id=agent_id)
                                    collision_back = utils.check_collision(ego_agent_0, target_agent)
                                    if collision_back:
                                        # not yield
                                        detected_relation = [[agent_id, each_other_agent]]
                                    else:
                                        # check relation
                                        self.online_predictor.relation_pred_onetime(each_pair=[agent_id, each_other_agent],
                                                                                    current_frame=current_frame_idx,
                                                                                    clear_history=True,
                                                                                    current_data=current_state)
                                        detected_relation = current_state['predicting']['relation']

                                # data to save
                                if 'relations_per_frame_env' not in current_state['predicting']:
                                    current_state['predicting']['relations_per_frame_env'] = {}
                                for dt in range(self.planning_interval):
                                    if (current_frame_idx + dt) not in current_state['predicting']['relations_per_frame_env']:
                                        current_state['predicting']['relations_per_frame_env'][current_frame_idx + dt] = []
                                    current_state['predicting']['relations_per_frame_env'][current_frame_idx + dt] += detected_relation

                                if [agent_id, each_other_agent] in detected_relation:
                                    if [each_other_agent, agent_id] in detected_relation:
                                        # bi-directional relations, still yield
                                        pass
                                    else:
                                        my_reactors.append(each_other_agent)
                                        to_yield = False

                        if to_yield:
                            earliest_collision_idx = ego_index_checking
                            collision_point = [ego_pose[0], ego_pose[1]]
                            earliest_target_agent = each_other_agent
                            if abs(each_other_agent_pose_array[i, 0] + 1) < 0.1 or abs(each_other_agent_pose_array[i-5, 0] + 1) < 0.1:
                                target_speed = 0
                            else:
                                target_speed = euclidean_distance(each_other_agent_pose_array[i, :2], each_other_agent_pose_array[i-5, :2]) / 5
                            break
                    if earliest_collision_idx is not None:
                        break

                if earliest_collision_idx is not None or self.method_testing < 2:
                    distance_to_travel = my_interpolator.get_distance_with_index(earliest_collision_idx) - S0
                    stopping_point = my_interpolator.interpolate(max(0, distance_to_travel - S0))[:2]
                    if euclidean_distance(interpolated_trajectory[0, :2],
                                          stopping_point) < MINIMAL_DISTANCE_TO_TRAVEL or distance_to_travel < MINIMAL_DISTANCE_TO_TRAVEL or my_current_v_per_step < 0.1:
                        planed_traj = self.get_trajectory_from_interpolator(my_interpolator=my_interpolator,
                                                                            my_current_speed=my_current_v_per_step,
                                                                            desired_speed=my_target_speed,
                                                                            emergency_stop=True)
                        agents_dic_copy[agent_id]['action'] = 'stop'
                    else:
                        planed_traj = self.adjust_speed_for_collision(interpolator=my_interpolator,
                                                                      distance_to_end=distance_to_travel,
                                                                      current_v=my_current_v_per_step,
                                                                      end_point_v=min(my_current_v_per_step * 0.8,
                                                                                      target_speed))
                        assert len(planed_traj.shape) > 1, planed_traj.shape
                        agents_dic_copy[agent_id]['action'] = 'yield'

                    # print("Yielding log: ", agent_id, each_other_agent, earliest_target_agent, earliest_collision_idx, distance_to_travel)
                else:
                    # no conflicts to yield
                    if euclidean_distance(interpolated_trajectory[0, :2], interpolated_trajectory[-1, :2]) < MINIMAL_DISTANCE_TO_TRAVEL:
                        planed_traj = self.get_trajectory_from_interpolator(my_interpolator=my_interpolator,
                                                                            my_current_speed=my_current_v_per_step,
                                                                            desired_speed=my_target_speed,
                                                                            hold_still=True)
                    else:
                        planed_traj = interpolated_trajectory
                    agents_dic_copy[agent_id]['action'] = 'controlled'

                if self.test_task == 1:
                    plan_for_ego = True
                if not plan_for_ego and ego_id == agent_id:
                    agents_dic_copy[agent_id]['action'] = None
                else:
                    if self.test_task != 2:
                        if collision_point is not None:
                            current_state['predicting']['points_to_mark'].append(collision_point)
                        current_state['predicting']['trajectory_to_mark'].append(planed_traj)

                    # if agent_id == 181:
                    #     for each_traj in prediction_traj_dic_m[agent_id]['rst']:
                    #         current_state['predicting']['trajectory_to_mark'].append(each_traj)

                    # replace the trajectory
                    planning_horizon, _ = planed_traj.shape
                    agents_dic_copy[agent_id]['pose'][current_frame_idx:planning_horizon+current_frame_idx, :] = planed_traj[:total_time_frame - current_frame_idx, :]
            current_state['agent'] = agents_dic_copy
        return current_state

    def trajectory_from_cubic_BC(self, p1, p2, p3, p4, v):
        # form a Bezier Curve
        total_dist = utils.euclidean_distance(p4, p1)
        total_t = min(93, int(total_dist/max(1, v)))
        traj_to_return = []
        for i in range(total_t):
            if i >= 92:
                break
            t = (i+1)/total_t
            p0_x = pow((1 - t), 3) * p1[0]
            p0_y = pow((1 - t), 3) * p1[1]
            p1_x = 3 * pow((1 - t), 2) * t * p2[0]
            p1_y = 3 * pow((1 - t), 2) * t * p2[1]
            p2_x = 3 * (1 - t) * pow(t, 2) * p3[0]
            p2_y = 3 * (1 - t) * pow(t, 2) * p3[1]
            p3_x = pow(t, 3) * p4[0]
            p3_y = pow(t, 3) * p4[1]
            traj_to_return.append((p0_x+p1_x+p2_x+p3_x, p0_y+p1_y+p2_y+p3_y))
        return np.array(traj_to_return, ndmin=2)

    def select_trajectory_from_prediction(self, prediction_dic, agent_id, goal_point, original_trajectory,
                                          remaining_frames, follow_goal=False, follow_original_as_default=True):
        if agent_id not in prediction_dic:
            return None

        # if always follow original as default
        if follow_original_as_default:
            follow_original = True
        else:
            follow_original = False
        rst = prediction_dic[agent_id]['rst']
        score = np.exp(prediction_dic[agent_id]['score'])
        score /= np.sum(score)
        if isinstance(rst, type([])):
            total_rst = len(rst)
        else:
            total_rst = rst.shape[0]

        if self.method_testing < 0:
            # SimNet variety does not follow original path
            return rst[0], False

        if follow_original:
            # select the closest prediction and return
            distance = np.zeros_like(score)
            for i in range(total_rst):
                distance[i] = self.get_l2_regulate_distance_for_two_trajectories(original_trajectory, rst[i], remaining_frames)
            best_idx = np.argmax(score/distance)
        else:
            best_idx = np.argmax(score)

        follow_goal = False

        return rst[best_idx], follow_goal



        # if follow_goal:
        #     distance = np.zeros_like(score)
        #     for i in range(total_rst):
        #         distance[i] = self.get_l2_regulate_distance_for_two_trajectories(original_trajectory, rst[i], remaining_frames)
        #     if min(distance) > MAX_DEVIATION_FOR_PREDICTION and remaining_frames > 5:
        #         follow_original = True
        #     best_idx = np.argmax(score/distance)
        # else:
        #     best_idx = np.argmax(score)
        #
        # distance_from_current_pose = self.get_l2_regulate_distance_for_two_trajectories(original_trajectory, [rst[best_idx, 0, :]], remaining_frames)
        # current_v = euclidean_distance(rst[best_idx, 0, :2], rst[best_idx, 1, :2])
        # if distance_from_current_pose > current_v:
        #     # too far to project back
        #     follow_original = False
        # yaw_diff = utils.normalize_angle(original_trajectory[0, 3] - original_trajectory[-1, 3])
        # if abs(yaw_diff) < math.pi/180*45:
        #     if current_v < MINIMAL_SPEED_TO_TRACK_ORG_GOAL:
        #         follow_original = False
        # elif follow_goal:
        #     follow_original = True
        #
        # return rst[best_idx], follow_original

    def get_l2_regulate_distance_for_two_trajectories(self, original_trajectory, compared_trajectory, comparing_frames):
        distance = []
        for idx1, each_pose in enumerate(compared_trajectory):
            if idx1 > comparing_frames:
                break
            distances_across = []
            for idx2, each_in_org in enumerate(original_trajectory):
                l2 = euclidean_distance(each_pose[:2], each_in_org[:2])
                distances_across.append(l2)
            distance.append(min(distances_across))
        # return distance
        return max(distance)

    def get_rescale_trajectory(self, reactor_current_pose, reactor_traj, reactor_interpolator, scale, debug=False,
                               current_v_per_step=None, constant_speed=False, current_a_per_step=None, target_speed=7,
                               follow_lanes=False):
        total_time = min(150, reactor_traj.shape[0])
        traj_to_return = np.zeros([total_time, 4])
        total_distance_traveled = []
        if current_v_per_step is not None:
            current_v = current_v_per_step
        else:
            current_v = euclidean_distance(reactor_current_pose[:2], reactor_traj[0, :2])
        for i in range(total_time):
            if constant_speed:
                if current_a_per_step is None:
                    dist = current_v
                else:
                    current_v += max(-A_SLOWDOWN_DESIRE/self.frame_rate, min(A_SPEEDUP_DESIRE/self.frame_rate, current_a_per_step))
                    current_v = max(0, current_v)
                    dist = current_v
            else:
                if i == 0:
                    dist = utils.euclidean_distance(reactor_current_pose[:2], reactor_traj[i, :2])*scale
                else:
                    dist = utils.euclidean_distance(reactor_traj[i-1, :2], reactor_traj[i, :2])*scale
                if dist > current_v + A_SPEEDUP_DESIRE/self.frame_rate:
                    current_v += A_SPEEDUP_DESIRE/self.frame_rate
                    current_v = min(target_speed, current_v)
                    dist = current_v
                elif dist < current_v - A_SLOWDOWN_DESIRE/self.frame_rate:
                    current_v -= A_SLOWDOWN_DESIRE/self.frame_rate
                    current_v = max(0, current_v)
                    dist = current_v
            total_distance_traveled.append(dist)
        total_distance_traveled = np.cumsum(total_distance_traveled)
        for i in range(len(total_distance_traveled)):
            traj_to_return[i, :] = reactor_interpolator.interpolate(total_distance_traveled[i], debug=debug)
        return traj_to_return

    def filter_trajectory_after_goal_point(self, traj, goal_point):
        last_pose = None
        last_distance = 999999
        traj_to_returen = traj.copy()
        for idx, each_pose in enumerate(traj):
            if last_pose is not None:
                traj_to_returen[idx, :] = last_pose
                continue
            next_distance = euclidean_distance(each_pose[:2], goal_point)
            if next_distance < last_distance + 0.001:
                last_distance = next_distance
            else:
                last_pose = each_pose
        return traj_to_returen

    def get_action(self):
        return 0

    def assert_traj(self, traj):
        total_time, _ = traj.shape
        if total_time < 30:
            return -1
        for i in range(total_time):
            if i == 0:
                continue
            if i >= total_time - 3 or i >= 20:
                break
            dist_1 = euclidean_distance(traj[6+i, :2], traj[1+i, :2]) / 5
            dist_2 = euclidean_distance(traj[5+i, :2], traj[i, :2]) / 5
            if abs(dist_1 - dist_2) > 5.0/self.frame_rate:
                print("Warning: frame jumping at: ", i, abs(dist_1 - dist_2))
                return i
        return -1


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
                print(f"{i} {next_step} {distance} {total_frame} {self.current_pose}")
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
        if distance_input - 2 > distance:
            # hide it outshoot
            # logging.warning(f'Over shooting while planning!!!!!!!!!')
            return self.get_state_from_poses(pose1, pose2, distance, next_step)
        else:
            # return current pose if trajectory not moved at all
            return self.current_pose
            # pose1 = self.current_pose[:2]
            # pose2 = pose[0, :2]
            # return self.get_state_from_poses(pose1, pose2, 0, 0.001)

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
                    step = euclidean_distance(self.current_pose[:2], pose[i, :2])
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


class Agent(car.Agent):
    def yaw_changer(self, yaw):
        return change_axis(-yaw)
