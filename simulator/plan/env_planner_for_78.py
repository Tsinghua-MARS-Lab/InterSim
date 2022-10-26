from prediction.M2I.predictor import M2IPredictor
import numpy as np
import math
import logging
import interactive_sim.envs.util as utils
import copy

S0 = 2
T = 0.25 #1.5  # reaction time when following
DELTA = 4  # the power term in IDM
PLANNING_HORIZON = 5  # in frames
PREDICTION_HTZ = 10  # prediction_htz
T_HEADWAY = 0.2
A_SPEEDUP_DESIRE = 1.5  # A
A_SLOWDOWN_DESIRE = 1.5  # B
XPT_SHRESHOLD = 0.7
MINIMAL_DISTANCE_PER_STEP = 0.05
MINIMAL_DISTANCE_TO_TRAVEL = 4
MINIMAL_DISTANCE_TO_RESCALE = -999 #0.1
REACTION_AFTER = 60  # in frames
MINIMAL_SCALE = 0.3
MAX_DEVIATION_FOR_PREDICTION = 4
TRAFFIC_LIGHT_COLLISION_SIZE = 2

MINIMAL_SPEED_TO_TRACK_ORG_GOAL = 5
MINIMAL_DISTANCE_TO_GOAL = 15

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

class EnvPlanner:
    """
    EnvPlanner is capable of using as much information as it can to satisfy its loss like avoiding collisions.
    EnvPlanner can assume it's controlling all agents around if it does not exacerbate the sim-2-real gap.
    While the baseline planner or any planner controlling the ego vehicle can only use the prediction or past data
    """

    def __init__(self, env_config, predictor):
        self.planning_from = env_config.env.planning_from
        self.planning_interval = env_config.env.planning_interval
        self.scenario_frame_number = 0
        self.online_predictor = predictor
        self.method_testing = env_config.env.testing_method  # 0=densetnt with dropout, 1=0+post-processing, 2=1+relation
        self.test_task = env_config.env.test_task
        # self.data = None

    def reset(self, *args, **kwargs):
        self.online_predictor(new_data=kwargs['new_data'], model_path=kwargs['model_path'], time_horizon=kwargs['time_horizon'], predict_device=kwargs['predict_device'])
        self.online_predictor.setting_goal_points()
        # self.data = self.online_predictor.data

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
            current_agent = undetected_piles.pop()
            ego_poses = current_state['agent'][current_agent]['pose']
            ego_shape = current_state['agent'][current_agent]['shape'][0]
            detected_pairs = []
            for idx, each_pose in enumerate(ego_poses):
                if idx <= current_frame_idx:
                    continue
                ego_agent_packed = Agent(x=each_pose[0],
                                         y=each_pose[1],
                                         yaw=each_pose[3],
                                         length=ego_shape[1],
                                         width=ego_shape[0],
                                         agent_id=current_agent)
                for each_agent_id in current_state['agent']:
                    if [current_agent, each_agent_id] in detected_pairs:
                        continue
                    if each_agent_id == current_agent or each_agent_id in relevant_agents:
                        continue
                    target_agent_packed = Agent(x=current_state['agent'][each_agent_id]['pose'][idx, 0],
                                                y=current_state['agent'][each_agent_id]['pose'][idx, 1],
                                                yaw=current_state['agent'][each_agent_id]['pose'][idx, 3],
                                                length=current_state['agent'][each_agent_id]['shape'][0][1],
                                                width=current_state['agent'][each_agent_id]['shape'][0][0],
                                                agent_id=each_agent_id)
                    # collision = utils.check_collision_for_two_agents_rotate_and_dist_check(ego_agent_packed, target_agent_packed)
                    collision = utils.check_collision_two_methods(ego_agent_packed, target_agent_packed)
                    if collision:
                        detected_pairs.append([current_agent, each_agent_id])
                        yield_ego = True
                        # print(f"In: {current_agent} {each_agent_id} {undetected_piles} {current_state['predicting']['relation']}")
                        self.online_predictor.relation_pred_onetime(each_pair=[current_agent, each_agent_id],
                                                                    current_frame=current_frame_idx)
                        # print(
                        #     f"Out: {current_agent} {each_agent_id} {undetected_piles} {current_state['predicting']['relation']}")
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
        if 857 not in relevant_agents and 996 in relevant_agents:
            relevant_agents.append(857)
        print("test test: ", relevant_agents)
        current_state['predicting']['relevant_agents'] = relevant_agents
        current_state['predicting']['colliding_pairs'] = colliding_pairs
        # print(f"Collision based relevant agent detected finished: \n{relevant_agents} \n{colliding_pairs}")

    def clear_markers_per_step(self, current_state, current_frame_idx):
        self.scenario_frame_number = current_frame_idx
        frame_diff = self.scenario_frame_number - self.planning_from
        if frame_diff >= 0 and frame_diff % self.planning_interval == 0:
            current_state['predicting']['relation'] = []
            current_state['predicting']['points_to_mark'] = []
            current_state['predicting']['trajectory_to_mark'] = []

    def get_prediction_trajectories(self, current_frame_idx, current_state=None, time_horizon=80):
        self.scenario_frame_number = current_frame_idx
        frame_diff = self.scenario_frame_number - self.planning_from

        if frame_diff >= 0 and frame_diff % self.planning_interval == 0:
            # print("predicting relation: ", self.planning_interval, self.scenario_frame_number, frame_diff)
            # self.online_predictor.relationship_pred(frame_diff)
            self.collision_based_relevant_detection(current_frame_idx, current_state)
            current_state['predicting']['relation'] = []
            self.online_predictor.update_state(current_state)
            for each_pair in current_state['predicting']['colliding_pairs']:
                self.online_predictor.relation_pred_onetime(each_pair, current_frame_idx)
            # self.online_predictor.relation_pred_onetime(current_frame_idx)
            # use dynamic goal point for marginal prediction
            # use static goal point if goal point still far away
            goal_points = {}
            if len(current_state['predicting']['relevant_agents']) > 0:
                if self.method_testing < 0:
                    self.online_predictor.variety_predict(frame_diff)
                else:
                    self.online_predictor.marginal_predict(frame_diff)

                    self.online_predictor.variety_predict(frame_diff)
                    # for agent in current_state['agent']:
                    #     state_pose = current_state['agent'][agent]['pose']
                        # v_per_step = euclidean_distance(state_pose[current_frame_idx, :2], state_pose[current_frame_idx+1, :2])
                        # if v_per_step > 1:
                        #     goal_points[agent] = current_state['predicting']['original_trajectory'][agent]['pose'][-1, :2]
                        # check if is going to pass goal point, then change to dynamic goal point
                    # follow goal
                    for agent in current_state['predicting']['marginal_trajectory']:
                        marginal_traj = current_state['predicting']['marginal_trajectory'][agent]['rst'][0]
                        origial_goal = current_state['predicting']['original_trajectory'][agent]['pose'][-1, :2]
                        # checking if is turning
                        # yaw_diff = current_state['predicting']['original_trajectory'][agent]['pose'][-1, 3] - current_state['predicting']['original_trajectory'][agent]['pose'][0, 3]
                        # yaw_diff = utils.normalize_angle(yaw_diff)
                        # check if past the goal point
                        angle_to_goal = get_angle_of_a_line(marginal_traj[self.planning_interval + 10, :2], origial_goal)
                        goal_yaw = current_state['predicting']['original_trajectory'][agent]['pose'][-1, 3]
                        past_goal = False
                        normalized_angle = utils.normalize_angle(angle_to_goal - goal_yaw)
                        if normalized_angle > math.pi/2 or normalized_angle < -math.pi/2:
                            past_goal = True
                        # turning = abs(yaw_diff) > math.pi / 180 * 45
                        # goal_distance1 = euclidean_distance(marginal_traj[self.planning_interval + 10, :2], origial_goal)
                        # goal_distance2 = euclidean_distance(marginal_traj[self.planning_interval + 20, :2], origial_goal)
                        two_point_dist = euclidean_distance(marginal_traj[self.planning_interval + 10, :2], marginal_traj[self.planning_interval + 20, :2])
                        # print(f"BBBBBBBBBBBBBBB {agent} {origial_goal} {goal_distance1} {goal_distance2} {two_point_dist} {goal_yaw} {angle_to_goal} {normalized_angle}")
                        current_state['predicting']['follow_goal'][agent] = False
                        if two_point_dist < MINIMAL_SPEED_TO_TRACK_ORG_GOAL:
                            continue
                        if euclidean_distance(origial_goal, marginal_traj[0, :2]) < MINIMAL_DISTANCE_TO_GOAL:
                            continue
                        # if goal_distance2 > (goal_distance1 + 2):
                        if past_goal:
                            continue
                        else:
                            goal_points[agent] = origial_goal
                            current_state['predicting']['follow_goal'][agent] = True
                            # print(f"{agent} is chasing goal point")
                            # print(f"TTTTTTTTTTTTTTTTTT {agent} {origial_goal} {goal_distance1} {goal_distance2} {two_point_dist}")
                        # print(f"TTT {agent} {v_per_step}")
                    if len(list(goal_points.keys())) > 0:
                        self.online_predictor.guilded_marginal_predict(ending_points=goal_points, current_frame=frame_diff)
            # self.online_predictor.reactor_predict(frame_diff)
            # self.online_predictor.xpt_predict(frame_diff)
            # update prediction data for visualize !!! Temporal for debugging
            marginal_prediction_results = self.online_predictor.data['predicting']['marginal_trajectory']
            conditional_prediction_results = marginal_prediction_results
            # conditional_prediction_results = self.online_predictor.data['predicting']['conditional_trajectory']
            agent_to_pred = {}

            for each_agent_id in conditional_prediction_results:
                if isinstance(each_agent_id, int):
                    if each_agent_id not in agent_to_pred:
                        agent_to_pred[each_agent_id] = {}
                    agent_to_pred[each_agent_id][frame_diff + 5] = {}
                    pred_trajectory = np.zeros([6, time_horizon, 2])
                    pred_yaw = np.zeros([6, time_horizon])
                    pred_scores = np.zeros([6, ])
                    # prediction_trajs = marginal_prediction_results[each_agent_id]['rst']
                    # prediction_scores = np.exp(marginal_prediction_results[each_agent_id]['score'])

                    prediction_trajs = conditional_prediction_results[each_agent_id]['rst']
                    prediction_scores = np.exp(conditional_prediction_results[each_agent_id]['score'])
                    default_yaw = current_state['agent'][each_agent_id]['pose'][self.scenario_frame_number, 3]

                    for each_prediction in range(6):
                        if each_agent_id in goal_points:
                            pred_trajectory[each_prediction, :, :] = self.filter_trajectory_after_goal_point(traj=prediction_trajs[each_prediction, :, :],
                                                                                                             goal_point=goal_points[each_agent_id])
                        else:
                            pred_trajectory[each_prediction, :, :] = prediction_trajs[each_prediction, :, :]
                        pred_scores[each_prediction] = prediction_scores[each_prediction]
                        pred_yaw[each_prediction] = calculate_yaw_from_states(pred_trajectory[each_prediction],
                                                                              default_yaw)
                    agent_to_pred[each_agent_id][frame_diff + 5]['pred_trajectory'] = pred_trajectory
                    agent_to_pred[each_agent_id][frame_diff + 5]['pred_yaw'] = pred_yaw
                    if np.sum(pred_scores) > 0.01:
                        agent_to_pred[each_agent_id][frame_diff + 5]['pred_scores'] = pred_scores / np.sum(
                            pred_scores)
                    else:
                        agent_to_pred[each_agent_id][frame_diff + 5]['pred_scores'] = pred_scores

            self.online_predictor.prediction_data = agent_to_pred
            self.online_predictor.last_predict_frame = frame_diff + 5
            return True
        else:
            return False

    # def update_env_trajectory_speed_only(self, current_frame_idx, relevant_only=True, current_state=None):
    def update_env_trajectory_for_sudo_base_planner(self, current_frame_idx, current_state=None):
        """
        the sudo base planner for the ego vehicle
        """
        if self.test_task == 1:
            # predict ego
            return current_state

        self.scenario_frame_number = current_frame_idx
        frame_diff = self.scenario_frame_number - self.planning_from
        ego_id = current_state['predicting']['ego_id'][1]

        # for each_agent in current_state['agent']:
        #     if each_agent in [748, 781, 735]:
        #         current_state['predicting']['trajectory_to_mark'].append(
        #             current_state['predicting']['original_trajectory'][each_agent]['pose'][:, :])

        if frame_diff >= 0 and frame_diff == 0: # frame_diff % self.planning_interval == 0:
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
                my_current_v_per_step -= A_SLOWDOWN_DESIRE/10/10
                step_speed = euclidean_distance(
                    current_state['agent'][ego_id]['pose'][current_frame_idx+i - 1, :2],
                    current_state['agent'][ego_id]['pose'][current_frame_idx+i - 2, :2])
                my_current_v_per_step = max(0, min(my_current_v_per_step, step_speed))
                current_state['agent'][ego_id]['pose'][current_frame_idx+i, :] = my_interpolator.interpolate(total_distance_traveled + my_current_v_per_step)
                total_distance_traveled += my_current_v_per_step

        if frame_diff >= 0 and frame_diff % self.planning_interval == 0:
            # current_state['predicting']['trajectory_to_mark'].append(
            #     current_state['predicting']['original_trajectory'][ego_id]['pose'][current_frame_idx:, :])
            current_state['predicting']['trajectory_to_mark'].append(current_state['agent'][ego_id]['pose'][current_frame_idx:, :])

        return current_state

    def update_env_trajectory_reguild(self, current_frame_idx, relevant_only=True, current_state=None, plan_for_ego=False):
        """
        plan and update trajectory to commit for relevant environment agents
        current_frame_idx: 1,2,3,...,11(first frame to plan)
        """
        if self.online_predictor.prediction_data is None:
            logging.warning('Skip planning: Planning before making a prediction')
            return
        self.scenario_frame_number = current_frame_idx
        frame_diff = self.scenario_frame_number - self.planning_from

        if frame_diff >= 0 and frame_diff % self.planning_interval == 0:
            # print("updating trajectory: ", self.planning_interval, self.scenario_frame_number)
            if current_state is None:
                logging.warning('Skip planning: Current state is None')
                return
            agents = current_state['agent']
            relevant_agents = current_state['predicting']['relevant_agents']
            edges = current_state['predicting']['relation']
            # XPts = current_state['predicting']['XPt']
            # select marginal prediction traj
            prediction_traj_dic_m = current_state['predicting']['marginal_trajectory']

            # prediction_traj_dic_c = current_state['predicting']['conditional_trajectory']
            # prediction_traj_dic_m = prediction_traj_dic_c
            ego_id = current_state['predicting']['ego_id'][1]

            agents_dic_copy = copy.deepcopy(current_state['agent'])

            for agent_id in agents:
                # if relevant
                if relevant_only and agent_id not in relevant_agents:
                    continue
                current_state['agent'][agent_id]['action'] = None
                total_time_frame = current_state['agent'][agent_id]['pose'].shape[0]
                goal_point = current_state['predicting']['goal_pts'][agent_id]
                my_current_pose = current_state['agent'][agent_id]['pose'][current_frame_idx - 1]
                my_current_v_per_step = euclidean_distance(current_state['agent'][agent_id]['pose'][current_frame_idx - 1, :2],
                                                           current_state['agent'][agent_id]['pose'][current_frame_idx - 6, :2])/5
                org_pose = current_state['predicting']['original_trajectory'][agent_id]['pose'].copy()
                if self.method_testing<0:
                    follow_org_as_default = False
                else:
                    follow_org_as_default = True

                prediction_traj_m, follow_org = self.select_trajectory_from_prediction(prediction_traj_dic_m, agent_id,
                                                                                       goal_point,
                                                                                       original_trajectory=org_pose,
                                                                                       remaining_frames=total_time_frame - current_frame_idx,
                                                                                       follow_goal=
                                                                                       current_state['predicting'][
                                                                                           'follow_goal'][
                                                                                           agent_id],
                                                                                       follow_original_as_default=follow_org_as_default)
                assert prediction_traj_m is not None, f'{agent_id} / {relevant_agents}'
                if False:
                # if follow_org:
                    # print(f"{agent_id} is following original trajectory due to large deviation in predictions")
                    # possibly deviate too much, follow original trajectory
                    # Note currently agent is at least 20 frames to travel till the end of current scenario and decided to follow the original goal before reaching it
                    # 1. project current position onto the original trajectory
                    projected_pose_on_original = my_current_pose
                    # predicted_v_per_step = euclidean_distance(prediction_traj_m[0, :2], prediction_traj_m[1, :2])
                    closest_distance = 999999
                    closest_index = 0
                    for idx, each_pose in enumerate(org_pose):
                        if idx == org_pose.shape[0] - 1:
                            break
                        dist = euclidean_distance(each_pose[:2], my_current_pose[:2])
                        if dist < closest_distance:
                            closest_distance = dist
                            projected_pose_on_original = each_pose
                            closest_index = idx
                    my_interpolator = SudoInterpolator(org_pose[closest_index+1:, :2], projected_pose_on_original)
                    my_current_pose = projected_pose_on_original
                    # my_current_v_per_step = min(my_current_v_per_step, predicted_v_per_step)
                    distance_past = euclidean_distance(projected_pose_on_original[:2], org_pose[closest_index+1, :2])
                    for i in range(prediction_traj_m.shape[0]-1):
                        if current_frame_idx+i >= 91:
                            break
                        last_step_dist = euclidean_distance(
                            current_state['agent'][agent_id]['pose'][current_frame_idx + i, :2],
                            current_state['agent'][agent_id]['pose'][current_frame_idx + i - 1, :2])
                        predicted_v_per_step = euclidean_distance(prediction_traj_m[i, :2], prediction_traj_m[i+1, :2])
                        if predicted_v_per_step < 1:
                            looping_v = last_step_dist
                        else:
                            looping_v = min(last_step_dist, predicted_v_per_step)
                        distance_past += looping_v
                        prediction_traj_m[i, :] = my_interpolator.interpolate(distance_past)[:2]

                action = 0  # 0=No Action, 1=Follow, 2=Yield
                my_traj = prediction_traj_m.copy()

                # detect trajectory collisions
                # after collision detection, we have earliest_collision_idx, earliest_target_id, latest_collision_idx(for that earliest collision detected
                my_interpolator = SudoInterpolator(my_traj.copy(), my_current_pose)

                tl_dics = current_state['traffic_light']
                road_dics = current_state['road']

                earliest_collision_idx = None
                earliest_target_agent = None
                collision_point = None
                # length_before_collision = None
                reactor_ids_from_relation = [edge[1] for edge in edges if agent_id == edge[0]]
                influencer_ids_from_relation = [edge[0] for edge in edges if agent_id == edge[1]]

                traffic_light_ending_pts = []
                for lane_id in tl_dics.keys():
                    if lane_id == -1:
                        continue
                    tl = tl_dics[lane_id]
                    # get the position of the end of this lane
                    tl_state = tl["state"][current_frame_idx]  # Unknown = 0, Arrow_Stop = 1, Arrow_Caution = 2, Arrow_Go = 3, Stop = 4, Caution = 5, Go = 6, Flashing_Stop = 7, Flashing_Caution = 8
                    if tl_state in [1, 4, 7]:
                        for seg_id in road_dics.keys():
                            if lane_id == seg_id:
                                road_seg = road_dics[seg_id]
                                if road_seg["type"] in [1, 2, 3]:
                                    if len(road_seg["dir"].shape) < 1:
                                        continue
                                    end_point = road_seg["xyz"][0][:2]
                                    traffic_light_ending_pts.append(end_point)
                                break

                for each_other_agent in agents:
                    if self.method_testing < 1:
                        break
                    if each_other_agent == agent_id:
                        continue
                    if self.method_testing < 2:
                        i_am_the_influencer = False
                    else:
                        i_am_the_influencer = each_other_agent in reactor_ids_from_relation and each_other_agent not in influencer_ids_from_relation
                    ego_index_checking = 1  # current_frame_idx+1
                    collision_detected_now = False
                    latest_collision_id = None
                    end_checking_frame = np.clip(current_frame_idx+REACTION_AFTER, 0, 91)

                    for i in range(current_frame_idx, end_checking_frame):
                        # pack an Agent object for collision detection
                        current_frame_distance = my_interpolator.get_distance_with_index(ego_index_checking)
                        pose_in_pred = my_interpolator.interpolate(current_frame_distance)
                        if abs(pose_in_pred[0]) < 1.1 and abs(pose_in_pred[1]) < 1.1:
                            print("WARNING invalid pose: ", pose_in_pred)
                            continue

                        # if agent_id == 932 and each_other_agent == 716:
                        #     pose_in_pred = my_interpolator.interpolate(current_frame_distance, debug=True)
                        #     print(
                        #         f"test test: {ego_index_checking} {i} {current_frame_distance} {pose_in_pred} {current_state['agent'][each_other_agent]['pose'][i]}")
                        # else:
                        #     pose_in_pred = my_interpolator.interpolate(current_frame_distance)
                        ego_agent = Agent(x=pose_in_pred[0],
                                          y=pose_in_pred[1],
                                          yaw=pose_in_pred[3],
                                          length=current_state['agent'][agent_id]['shape'][0][1],
                                          width=current_state['agent'][agent_id]['shape'][0][0],
                                          agent_id=agent_id)
                        target_current_pose = current_state['agent'][each_other_agent]['pose'][i]
                        target_agent = Agent(x=target_current_pose[0],
                                             y=target_current_pose[1],
                                             yaw=target_current_pose[3],
                                             length=current_state['agent'][each_other_agent]['shape'][0][1],
                                             width=current_state['agent'][each_other_agent]['shape'][0][0],
                                             agent_id=each_other_agent)
                        # has_collision = utils.check_collision_for_two_agents_rotate_and_dist_check(checking_agent=ego_agent,
                        #                                                                            target_agent=target_agent,)
                        has_collision = utils.check_collision_two_methods(checking_agent=ego_agent,
                                                                          target_agent=target_agent,
                                                                          vertical_margin=2.0,
                                                                          vertical_margin2=2.0,
                                                                          horizon_margin=2.0)


                                                                                                   # vertical_margin=1,
                                                                                                   # vertical_margin2=1,
                                                                                                   # horizon_margin=1)
                        # has_collision = utils.check_collision_for_two_agents_rotate_and_dist_check(checking_agent=ego_agent,
                        #                                                                            target_agent=target_agent)

                        if i > 0:
                            # additionally detect collision between two frames for high speed moving misses
                            current_frame_distance2 = my_interpolator.get_distance_with_index(ego_index_checking - 1)
                            pose_in_pred2 = my_interpolator.interpolate(current_frame_distance2)
                            if abs(pose_in_pred2[0]) < 1.1 and abs(pose_in_pred2[1]) < 1.1:
                                continue
                            target_current_pose2 = current_state['agent'][each_other_agent]['pose'][i-1]
                            ego_agent2 = Agent(x=(pose_in_pred2[0] + pose_in_pred[0]) / 2,
                                               y=(pose_in_pred2[1] + pose_in_pred[1]) / 2,
                                               yaw=get_angle_of_a_line(pose_in_pred2[:2], pose_in_pred[:2]),
                                               length=euclidean_distance(pose_in_pred2[:2], pose_in_pred[:2]),
                                               width=current_state['agent'][agent_id]['shape'][0][0],
                                               agent_id=agent_id)
                            target_agent2 = Agent(x=(target_current_pose[0] + target_current_pose2[0]) / 2,
                                                  y=(target_current_pose[1] + target_current_pose2[1]) / 2,
                                                  yaw=get_angle_of_a_line(target_current_pose2[:2],
                                                                          target_current_pose[:2]),
                                                  length=euclidean_distance(target_current_pose2[:2],
                                                                            target_current_pose[:2]),
                                                  width=current_state['agent'][each_other_agent]['shape'][0][0],
                                                  agent_id=each_other_agent)
                            has_collision |= utils.check_collision_for_two_agents_rotate_and_dist_check(checking_agent=ego_agent2,
                                                                                                        target_agent=target_agent2,)

                        # check traffic light violation
                        running_red_light = False
                        for tl_pt in traffic_light_ending_pts:
                            dummy_tf_agent = Agent(x=tl_pt[0], y=tl_pt[1], yaw=0,
                                                   length=TRAFFIC_LIGHT_COLLISION_SIZE, width=TRAFFIC_LIGHT_COLLISION_SIZE, agent_id=99999)
                            running = utils.check_collision_for_two_agents_rotate_and_dist_check(
                                checking_agent=ego_agent,
                                target_agent=dummy_tf_agent)
                            if i > 0:
                                running |= utils.check_collision_for_two_agents_rotate_and_dist_check(
                                    checking_agent=ego_agent2,
                                    target_agent=dummy_tf_agent)
                            if running:
                                running_red_light = True
                                break

                        if has_collision and not i_am_the_influencer:
                            collision_detected_now = True
                            # print(f'test c: {agent_id} {each_other_agent} {ego_index_checking} {i}')
                            # pose = current_state['agent'][each_other_agent]['pose'][i]
                            # shape1 = current_state['agent'][agent_id]['shape'][0]
                            # shape2 = current_state['agent'][each_other_agent]['shape'][0]
                            # print(f'test 1: {pose_in_pred} {pose} {shape1} {shape2}')
                            # stick ego index to check latest collision index for current target agent
                            if earliest_collision_idx is None or ego_index_checking < earliest_collision_idx:
                                collision_point = [pose_in_pred[0], pose_in_pred[1]]
                                earliest_collision_idx = ego_index_checking
                                earliest_target_agent = each_other_agent
                                # length_before_collision = current_state['agent'][agent_id]['shape'][0][1] + current_state['agent'][each_other_agent]['shape'][0][1]
                        elif running_red_light:
                            collision_detected_now = True
                            if earliest_collision_idx is None or ego_index_checking < earliest_collision_idx:
                                collision_point = [pose_in_pred[0], pose_in_pred[1]]
                                earliest_collision_idx = ego_index_checking
                                earliest_target_agent = 99999
                        else:
                            if collision_detected_now and (earliest_target_agent == each_other_agent) and (latest_collision_id is None):
                                latest_collision_id = ego_index_checking
                            collision_detected_now = False
                            ego_index_checking += 1

                # if agent_id == 857:
                #     scale = 0
                #     planed_traj = self.get_rescale_trajectory(reactor_current_pose=my_current_pose,
                #                                               reactor_traj=my_traj,
                #                                               reactor_interpolator=my_interpolator,
                #                                               scale=scale,
                #                                               current_v_per_step=my_current_v_per_step)

                if earliest_collision_idx is not None:
                # if earliest_collision_idx is not None:
                    # solving the collision by predicted relations
                    # 1. skip solving the ego agent's relationship
                    to_yield = True
                    # if agent_id != current_state['predicting']['ego_id'] and earliest_target_agent != 99999:
                    if earliest_target_agent != 99999:
                        # 2. solving the pair: [agent_id, earliest_target_agent]
                        self.online_predictor.relation_pred_onetime(each_pair=[agent_id, earliest_target_agent],
                                                                    current_frame=frame_diff)
                        detected_relation = current_state['predicting']['relation']
                        if [agent_id, earliest_target_agent] in detected_relation:
                            if [earliest_target_agent, agent_id] in detected_relation:
                                # bi-directional relations, still yield
                                pass
                            else:
                                to_yield = False

                        # fast one yield
                        my_speed = euclidean_distance(my_traj[earliest_collision_idx, :2], my_traj[earliest_collision_idx+1, :2])
                        target_speed = euclidean_distance(current_state['agent'][earliest_target_agent]['pose'][earliest_collision_idx][:2],
                                                          current_state['agent'][earliest_target_agent]['pose'][earliest_collision_idx+1][:2])
                        if my_speed < target_speed * 0.7:
                            to_yield = False
                    if to_yield or self.method_testing < 2 or agent_id == 857:
                        distance_to_travel = my_interpolator.get_distance_with_index(earliest_collision_idx) - S0 * 2
                        l2_to_collision = euclidean_distance(collision_point, my_current_pose[:2]) - S0 * 3

                        print("Testttest: ", agent_id, distance_to_travel, l2_to_collision, MINIMAL_DISTANCE_TO_RESCALE)
                        if distance_to_travel < MINIMAL_DISTANCE_TO_RESCALE or l2_to_collision < MINIMAL_DISTANCE_TO_RESCALE:
                            # very close to collision point, sharp speed down
                            scale = 0.05
                            # if latest_collision_id is None:
                            #     total_distance_in_prediction = my_interpolator.get_distance_with_index(-1) + 0.0001
                            # else:
                            #     total_distance_in_prediction = my_interpolator.get_distance_with_index(
                            #         latest_collision_id) + 0.0001
                            # if distance_to_travel < MINIMAL_DISTANCE_TO_TRAVEL: # or l2_to_collision < MINIMAL_DISTANCE_TO_TRAVEL:
                            #     scale = 0
                            # else:
                            #     scale = np.clip(distance_to_travel / total_distance_in_prediction, 0, 1)
                            # if scale < MINIMAL_SCALE:
                            #     scale = 0
                            planed_traj = self.get_rescale_trajectory(reactor_current_pose=my_current_pose,
                                                                      reactor_traj=my_traj,
                                                                      reactor_interpolator=my_interpolator,
                                                                      scale=scale,
                                                                      current_v_per_step=my_current_v_per_step)
                            agents_dic_copy[agent_id]['action'] = 'yield'  # purple
                            # print(
                            #     f"test scale: {agent_id} {earliest_target_agent} {earliest_collision_idx} {scale} {distance_to_travel} {l2_to_collision} {total_distance_in_prediction}")
                            # print(
                            #     f"test scale: {agent_id} {earliest_target_agent} {earliest_collision_idx} {scale} {distance_to_travel} {l2_to_collision}")
                        else:
                            # if not follow_org:
                            if True:
                                # collision_point = current_state['agent'][agent_id]['pose'][earliest_collision_idx, :2]
                                stopping_point = my_interpolator.interpolate(distance_to_travel - S0 * 2)[:2]
                                self.online_predictor.guilded_predict(ending_points={agent_id: stopping_point.copy()},
                                                                      current_frame=frame_diff)
                                # self.online_predictor.guilded_predict(ending_points={agent_id: collision_point.copy()},
                                #                                       current_frame=frame_diff)
                                # follow_org_after_collision = current_state['predicting']['follow_goal'][agent_id] & (euclidean_distance(stopping_point, my_current_pose[:2]) > MINIMAL_DISTANCE_TO_TRAVEL * 3)
                                follow_org_after_collision = False
                                my_traj, _ = self.select_trajectory_from_prediction(
                                    current_state['predicting']['guilded_trajectory'], agent_id, goal_point,
                                    original_trajectory=current_state['predicting']['original_trajectory'][agent_id]['pose'][self.planning_from:, :].copy(),
                                    remaining_frames=total_time_frame-current_frame_idx, follow_original_as_default=follow_org_after_collision)
                                # my_traj = self.filter_trajectory_after_goal_point(my_traj, goal_point[0])
                                my_interpolator = SudoInterpolator(my_traj, my_current_pose)
                            agents_dic_copy[agent_id]['action'] = 'follow'  # yellow
                            # print(f"test reguild: {agent_id} {earliest_target_agent} {earliest_collision_idx} {collision_point} {distance_to_travel}")
                            scale = 1
                            if euclidean_distance(my_traj[0, :2], my_traj[-1, :2]) < MINIMAL_DISTANCE_TO_TRAVEL:
                                if euclidean_distance(my_current_pose[:2], my_traj[0, :2]) > 1.0:
                                    # never followed into this logic
                                    scale = 0.1
                                else:
                                    scale = 0.1
                            planed_traj = self.get_rescale_trajectory(reactor_current_pose=my_current_pose,
                                                                      reactor_traj=my_traj,
                                                                      reactor_interpolator=my_interpolator,
                                                                      scale=scale,
                                                                      current_v_per_step=my_current_v_per_step)
                    else:
                        planed_traj = self.get_rescale_trajectory(reactor_current_pose=my_current_pose,
                                                                  reactor_traj=my_traj,
                                                                  reactor_interpolator=my_interpolator,
                                                                  scale=1,
                                                                  current_v_per_step=my_current_v_per_step)
                else:

                    if euclidean_distance(my_traj[0, :2], my_traj[-1, :2]) < MINIMAL_DISTANCE_TO_TRAVEL:
                        scale = 0
                    else:
                        scale = 1
                    planed_traj = self.get_rescale_trajectory(reactor_current_pose=my_current_pose,
                                                              reactor_traj=my_traj,
                                                              reactor_interpolator=my_interpolator,
                                                              scale=scale,
                                                              current_v_per_step=my_current_v_per_step)

                agents_dic_copy[agent_id]['action'] = 'follow'  # yellow
                    # print(f"test trajectory: {agent_id} {my_traj[-1]}")
                # prev = current_state['agent'][agent_id]['pose'][50, :]
                # print(f'test poses replace: {prev} {planed_traj[50-11]} {my_traj[50-11]}')
                if self.test_task == 1:
                    plan_for_ego = True
                if not plan_for_ego and ego_id == agent_id:
                    agents_dic_copy[agent_id]['action'] = None
                else:
                    if collision_point is not None:
                        current_state['predicting']['points_to_mark'].append(collision_point)
                    current_state['predicting']['trajectory_to_mark'].append(planed_traj)

                    # if agent_id == 181:
                    #     for each_traj in prediction_traj_dic_m[agent_id]['rst']:
                    #         current_state['predicting']['trajectory_to_mark'].append(each_traj)

                    planning_horizon, _ = planed_traj.shape
                    agents_dic_copy[agent_id]['pose'][current_frame_idx:planning_horizon+current_frame_idx, :] = planed_traj[
                                                                                      :total_time_frame - current_frame_idx,
                                                                                      :]
                    # current_state['agent'][agent_id]['pose'][current_frame_idx:planning_horizon+current_frame_idx, :] = planed_traj[
                    #                                                                   :total_time_frame - current_frame_idx,
                    #                                                                   :]
            current_state['agent'] = agents_dic_copy
        return current_state

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

        if self.method_testing < 0:
            # SimNet variety does not follow original path
            return rst[0], False

        if follow_goal:
            distance = np.zeros_like(score)
            for i in range(6):
                distance[i] = self.get_l2_regulate_distance_for_two_trajectories(original_trajectory, rst[i], remaining_frames)
            # print(f"test: {agent_id} {score} {distance} {score/distance}")
            if min(distance) > MAX_DEVIATION_FOR_PREDICTION and remaining_frames > 5:
                follow_original = True
            best_idx = np.argmax(score/distance)
        else:
            best_idx = np.argmax(score)

        distance_from_current_pose = self.get_l2_regulate_distance_for_two_trajectories(original_trajectory, [rst[best_idx, 0, :]], remaining_frames)
        current_v = euclidean_distance(rst[best_idx, 0, :2], rst[best_idx, 1, :2])
        if distance_from_current_pose > current_v:
            # too far to project back
            follow_original = False
        yaw_diff = utils.normalize_angle(original_trajectory[0, 3] - original_trajectory[-1, 3])
        if abs(yaw_diff) < math.pi/180*45:
            if current_v < MINIMAL_SPEED_TO_TRACK_ORG_GOAL:
                follow_original = False
        elif follow_goal:
            follow_original = True

        return rst[best_idx], follow_original

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

    def get_rescale_trajectory(self, reactor_current_pose, reactor_traj, reactor_interpolator, scale, debug=False, current_v_per_step=None):
        total_time = reactor_traj.shape[0]
        traj_to_return = np.zeros([total_time, 4])
        # if scale == 0.1:
        #     # stop to v0 at step 5
        #     current_vx = reactor_traj[0, 0] - reactor_current_pose[0]
        #     current_vy = reactor_traj[0, 1] - reactor_current_pose[1]
        #     for i in range(4):
        #         x = reactor_current_pose[0] + current_vx / 4 * (3-i)
        #         y = reactor_current_pose[1] + current_vy / 4 * (3-i)
        #         reactor_current_pose = traj_to_return[i] = [x, y, 0, reactor_current_pose[3].copy()]
        #     traj_to_return[4:, :] = traj_to_return[3, :]
        #     return traj_to_return

        total_distance_traveled = []
        if current_v_per_step is not None:
            current_v = current_v_per_step
        else:
            current_v = euclidean_distance(reactor_current_pose[:2], reactor_traj[0, :2])
        for i in range(total_time):
            if i == 0:
                dist = utils.euclidean_distance(reactor_current_pose[:2], reactor_traj[i, :2])
            else:
                dist = utils.euclidean_distance(reactor_traj[i-1, :2], reactor_traj[i, :2])
            if dist > current_v + A_SPEEDUP_DESIRE/10/10:
                current_v += A_SPEEDUP_DESIRE/10/10
                dist = current_v
            elif dist < current_v - A_SLOWDOWN_DESIRE/10/10:
                current_v -= A_SLOWDOWN_DESIRE/10/10
                dist = current_v
            total_distance_traveled.append(dist)
        total_distance_traveled = np.cumsum(total_distance_traveled)
        prev_yaw = reactor_current_pose[3]
        for i in range(len(total_distance_traveled)):
            traj_to_return[i, :] = reactor_interpolator.interpolate(total_distance_traveled[i]*scale, debug=debug)
            if total_distance_traveled[i] < 1:
                traj_to_return[i, 3] = prev_yaw
            else:
                prev_yaw = traj_to_return[i, 3]
            # traj_to_return[i, :] = reactor_interpolator.interpolate(0)
        average_yaw_diff = abs(utils.normalize_angle(np.average(traj_to_return[-5:-1, 3]) - reactor_current_pose[3]))
        if average_yaw_diff < math.pi / 180 * 10:
            # a reluctant to turn
            traj_to_return[:, 3] = reactor_current_pose[3]
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
                    step = euclidean_distance(self.current_pose[:2], pose[i, :2])
                else:
                    step = euclidean_distance(pose[i, :2], pose[i-1, :2])
                if step > MINIMAL_DISTANCE_PER_STEP:
                    distance += step
        return distance


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
