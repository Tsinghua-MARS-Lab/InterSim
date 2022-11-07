from typing import List
import interactive_sim.envs.util as utils
import math
from enum import IntEnum


class Actions(IntEnum):
    follow = 1
    laneChange = 2  # for same direction lanes (including merge)
    skip = 3  # for intersection turning lanes
    cutIn = 4  # for not connected lanes / other default situations

    @staticmethod
    def to_string(a: int):
        return str(Actions(a)).split('.')[1]

class GoalSetter:
    def __init__(self):
        self.data = None

    def __call__(self, *args, **kwargs):
        self.data = kwargs['new_data']

    def get_goal(self, current_data, agent_id, dataset='Waymo') -> List:
        # get last valid point as the goal point
        # agent_dic = current_data['agent'][agent_id]
        agent_dic = current_data['predicting']['original_trajectory'][agent_id]
        yaw = None
        point = None
        if dataset == 'Waymo':
            # Waymo
            for frame_idx in range(1, 80):
                if yaw is not None:
                    break
                if agent_dic['pose'][-frame_idx][0] != -1 and agent_dic['pose'][-frame_idx][1] != -1:
                    point = [agent_dic['pose'][-frame_idx][0], agent_dic['pose'][-frame_idx][1]]
                    yaw = agent_dic['pose'][-frame_idx][3]
                    break
        elif dataset == 'NuPlan':
            # NuPlan
            assert 'ego_goal' in current_data, 'Goal Setter: Not found goal in data dic'
            goal = current_data['ego_goal']
            if agent_id == 'ego' and goal is not None:
                point = [goal[0], goal[1]]
                yaw = goal[3]
            else:
                for frame_idx in range(1, 180):
                    if yaw is not None:
                        break
                    if agent_dic['pose'][-frame_idx][0] != -1 and agent_dic['pose'][-frame_idx][1] != -1:
                        point = [agent_dic['pose'][-frame_idx][0], agent_dic['pose'][-frame_idx][1]]
                        yaw = agent_dic['pose'][-frame_idx][3]
                        break
                if point is None:
                    if agent_id == 'ego':
                        # print('ERROR: goal point is none ', agent_dic['pose'], agent_id)
                        print('[Static goal] ERROR: goal point is none ', agent_id)
                    point = [0, 0]
                    yaw = 0
        return [point, yaw]

    # def get_route_and_actions(self, agent_id: int):
    #     agent_dic = self.data['agent'][agent_id]
    #     road_dic = self.data['road']
    #     gt_trajectory = agent_dic['pose']
    #     total_time = gt_trajectory.shape[0]
    #     lanes_following = []  # [lane_a, lane_b, lane_c, ...]
    #     enter_idx = []  # [idx_lane_a, idx_lane_b, idx_lane_c, ...]
    #     actions = []
    #     for frame_idx in range(total_time):
    #         current_pose = gt_trajectory[frame_idx]
    #         # variables for looping the closest lane
    #         closest_dist = 999999
    #         closest_dist_threshold = 2
    #         closest_lane = None
    #         closest_lane_pt_idx = None
    #         for each_lane in road_dic:
    #             if road_dic[each_lane]['type'][0] not in [1, 2]:
    #                 continue
    #             road_xy = road_dic[each_lane]['xyz'][:, :2]
    #             if road_xy.shape[0] < 3:
    #                 # skip lane with 1 or 2 points
    #                 continue
    #             for j, each_xy in enumerate(road_xy):
    #                 road_yaw = road_dic[each_lane]['dir'][j]
    #                 yaw_diff = abs(utils.normalize_angle(current_pose[3] - road_yaw))
    #                 dist = utils.euclidean_distance(each_xy, current_pose[:2])
    #                 if dist < closest_dist_threshold and dist < closest_dist and yaw_diff < math.pi / 180 * 30:
    #                     closest_lane = each_lane
    #                     closest_dist = dist
    #                     closest_lane_pt_idx = j
    #         # end of looping lanes
    #         if closest_lane is not None and closest_lane not in lanes_following:
    #             lanes_following.append(closest_lane)
    #             enter_idx.append(closest_lane_pt_idx)
    #             if len(lanes_following) > 1:
    #                 # classify actions from lane_a to lane_b
    #                 prev_lane = lanes_following[-2]
    #                 last_enter_idx = enter_idx[-2]
    #                 next_lanes = road_dic[prev_lane]['next_lanes']
    #                 if closest_lane in next_lanes:
    #                     # connecting
    #                     # check if current_lane is a turning intersection lane
    #                     current_lane_next_lanes = road_dic[closest_lane]['next_lanes']
    #                     if len(current_lane_next_lanes) == 1 and road_dic[prev_lane]['outbound'] and road_dic[closest_lane]['turning'] in [1, 2, 3]:
    #                         # this is a turning intersection lane, action skip
    #                         actions.append(Actions.skip)
    #                     else:
    #                         # this is not a turning intersection lane, action follow
    #                         actions.append(Actions.follow)
    #                 else:
    #                     # not connecting
    #                     # check yaw difference for lane changing
    #                     closest_lane_yaw = road_dic[closest_lane]['dir'][closest_lane_pt_idx]
    #                     prev_lane_yaw = road_dic[prev_lane]['dir'][last_enter_idx]
    #                     if abs(utils.normalize_angle(closest_lane_yaw - prev_lane_yaw)) < math.pi / 180 * 5:
    #                         actions.append(Actions.laneChange)
    #                     else:
    #                         actions.append(Actions.cutIn)
    #     print("test route and actions: ", agent_id, lanes_following, enter_idx, actions)
