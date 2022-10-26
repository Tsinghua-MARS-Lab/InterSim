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


class TrajPredPlanner(EnvPlanner):
    """
    TrajPredPlanner uses trajectory prediction to avoid other road users
    TrajPredPlanner should not use the gt future trajectory information for planning.
    The TrajPredPlanner has its own predictor which does not share information with the predictor of the EnvPlanner
    The TrajPredPlanner is used to control the ego agent only.
    The TrajPredPlanner is derived from the EnvPlanner, change predefined functions to build/test your own planner.
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
                    current_state['predicting']['mark_pts'] = [goal_pt]

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

            # predict on all agents for detection checking
            self.online_predictor.marginal_predict(current_frame=current_frame_idx,
                                                   selected_agent_ids='all',
                                                   current_data=current_state)
            for each_agent_id in self.online_predictor.data['predicting']['marginal_trajectory'].keys():
                if each_agent_id == ego_agent_id:
                    continue
                # if k = 1
                k = 1
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
                    current_state['predicting']['trajectory_to_mark'].append(pred_traj_with_yaw)

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
                    if abs(ego_org_traj[20, 3] + 1) < 0.01:
                        ego_turning_right = False
                    else:
                        ego_yaw_diff = -utils.normalize_angle(ego_org_traj[20, 3] - ego_org_traj[0, 3])
                        ego_running_red_light = False
                        if math.pi / 180 * 15 < ego_yaw_diff and abs(ego_org_traj[20, 3] + 1) > 0.01:
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
                        return [earliest_collision_idx, collision_point, earliest_conflict_agent, target_speed, None]

                    for j, each_other_traj in enumerate(others_trajectory):
                        target_agent_id = target_agent_ids[j]

                        target_shape = [max(1, current_state['agent'][target_agent_id]['shape'][0][0]),
                                        max(1, current_state['agent'][target_agent_id]['shape'][0][1])]

                        if target_agent_id in my_reactors:
                            continue
                        total_frame_in_target = each_other_traj.shape[0]
                        if current_time > total_frame_in_target - 1:
                            continue
                        target_pose = each_other_traj[current_time]
                        if abs(target_pose[0]) < 1.1 and abs(target_pose[1]) < 1.1:
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

                        if running_red_light:
                            continue
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
                                has_collision |= utils.check_collision(checking_agent=ego_agent2,
                                                                                            target_agent=target_agent2)
                            else:
                                has_collision |= utils.check_collision(checking_agent=ego_agent2,
                                                                                            target_agent=target_agent)

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
                                collision_0 = utils.check_collision(ego_agent, target_agent_0)
                                collision_0 |= utils.check_collision(ego_agent2, target_agent_0)
                                if collision_0:
                                    # yield
                                    detected_relation = [[target_agent_id, ego_agent_id]]
                                else:
                                    # check relation
                                    # if collision, solve conflict
                                    self.online_predictor.relation_pred_onetime(each_pair=[ego_agent_id, target_agent_id],
                                                                                current_frame=current_frame_idx,
                                                                                clear_history=True,
                                                                                current_data=current_state)
                                    detected_relation = self.online_predictor.data['predicting']['relation']

                                    if [ego_agent_id, target_agent_id] in detected_relation:
                                        if [target_agent_id, ego_agent_id] in detected_relation:
                                            # bi-directional relations, still yield
                                            pass
                                        else:
                                            # not to yield, and skip conflict
                                            my_reactors.append(target_agent_id)
                                            continue
                            else:
                                detected_relation = [[target_agent_id, ego_agent_id]]

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

            # process prior collision pairs
            rst = detect_conflicts_and_solve(prior_agent_traj, prior_agent_ids, always_yield=True)
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
            rst = detect_conflicts_and_solve(other_agent_traj, other_agent_ids, always_yield=False)
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

            if earliest_collision_idx is not None and tf_light_frame_idx is not None and earliest_collision_idx < tf_light_frame_idx:
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
                if euclidean_distance(my_traj[0, :2],
                                      stopping_point) < MINIMAL_DISTANCE_TO_TRAVEL or distance_to_travel < MINIMAL_DISTANCE_TO_TRAVEL or my_current_v_per_step < 0.1:
                    planed_traj = self.get_trajectory_from_interpolator(my_interpolator=my_interpolator,
                                                                        my_current_speed=my_current_v_per_step,
                                                                        desired_speed=my_target_speed,
                                                                        emergency_stop=True)
                    current_state['predicting']['emergency_stopping'] = True
                else:
                    planed_traj = self.adjust_speed_for_collision(interpolator=my_interpolator,
                                                                  distance_to_end=distance_to_travel,
                                                                  current_v=my_current_v_per_step,
                                                                  end_point_v=min(my_current_v_per_step * 0.8,
                                                                                  target_speed))
                    assert len(planed_traj.shape) > 1, planed_traj.shape
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

            planning_horizon, _ = planed_traj.shape
            jumping_idx = self.assert_traj(planed_traj[:total_time_frame - current_frame_idx, :2])
            if jumping_idx != -1:
                if jumping_idx >= 6:
                    planed_traj[jumping_idx:, :] = -1
                else:
                    print(f'Early jumping {jumping_idx} {ego_agent_id}')
                    # assert False, f'Jumping early: {jumping_idx} {ego_agent_id}'
            current_state['agent'][ego_agent_id]['pose'][current_frame_idx:planning_horizon + current_frame_idx, :] = planed_traj[:total_time_frame - current_frame_idx, :]
        return current_state

