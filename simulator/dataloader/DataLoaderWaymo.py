import tensorflow as tf
from interactive_sim.envs.util import *
import math
import os
import numpy as np
import random
from interactions.detect_relations import get_relation_on_crossing, form_tree_from_edges
import pickle

# [Debug Only]: Manually assign starting file or scene
FILE_TO_START = 0
# interesting scenes:
# 185 see 4 ways stop line, 201 roundabout, 223 & 230 for a huge intersection
# interesting scenes in the visual file:
# 54 challenging heterogeneous turnings, 53 highway cut-in against dense traffic
# 52
# 59 failure case while turning with an ending point not finishing the turning
# 78 for a good two vehicle interaction demo
# for relationship flip: 134, 138, 139, 226
# for simulation
# failure cases: 13
SCENE_TO_START = 242     # 107 for nudging  # 38 for turning large intersection failure
SAME_WAY_LANES_SEARCHING_DIST_THRESHOLD = 20
SAME_WAY_LANES_SEARCHING_DIRECTION_THRESHOLD = 0.1

def vector2radical(vector):
    x = float(vector[0])
    y = float(vector[1])
    return math.atan2(y, x + 0.01)


def velvector2value(vector_np_x, vector_np_y):
    rst = []
    for i in range(len(vector_np_x)):
        x = float(vector_np_x[i])
        y = float(vector_np_y[i])
        rst.append(math.sqrt(x * x + y * y))
    return np.array(rst)


def miniPiPi_to_zeroTwoPie(direction):
    if direction < 0:
        return direction + math.pi * 2
    else:
        return direction


def load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


roadgraph_features = {
    'roadgraph_samples/dir':
        tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
    'roadgraph_samples/id':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/type':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/valid':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/xyz':
        tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
}

# Features of other agents.
state_features = {
    'state/id':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/type':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/is_sdc':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/tracks_to_predict':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/current/bbox_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/height':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/length':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/timestamp_micros':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/valid':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/vel_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    #     'state/current/speed':
    #         tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/current/velocity_x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/width':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/z':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/future/bbox_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/height':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/length':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/timestamp_micros':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/valid':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/vel_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/width':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/z':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    #     'state/future/speed':
    #         tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/past/bbox_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/height':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/length':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/timestamp_micros':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/valid':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    #     'state/past/speed':
    #         tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/past/vel_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/width':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/z':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
}

traffic_light_features = {
    'traffic_light_state/current/state':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/valid':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/id':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/past/state':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/valid':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/id':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/future/state':
        tf.io.FixedLenFeature([80, 16], tf.int64, default_value=None),
    'traffic_light_state/future/valid':
        tf.io.FixedLenFeature([80, 16], tf.int64, default_value=None),
    'traffic_light_state/future/id':
        tf.io.FixedLenFeature([80, 16], tf.int64, default_value=None),
    'traffic_light_state/current/x':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/y':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/z':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
}

features_description = {}
features_description.update(roadgraph_features)
features_description.update(state_features)
features_description.update(traffic_light_features)


def classify_scenario(scenario_dic):
    agents = scenario_dic["agent"]
    has_turning_agent = False
    has_traffic_light = False
    traffic_lights = scenario_dic["traffic_light"]
    for tl_id in traffic_lights.keys():
        traffic_light_dic = traffic_lights[tl_id]
        for frame_idx, valid in enumerate(traffic_light_dic["valid"]):
            state = traffic_light_dic["state"][frame_idx]
            if valid == 1 and state in [1, 2, 3, 4, 5, 6, 7, 8]:
                has_traffic_light = True
                break
        if has_traffic_light:
            break

    for agent_id in agents.keys():
        agent_dic = agents[agent_id]
        # counter turning agents
        if not agent_dic["type"] == 1:
            continue
        if not agent_dic["to_predict"]:
            continue
        dirs = agent_dic["pose"][:, 3]
        if is_turning_agent(dirs):
            has_turning_agent = True
            break

    if has_traffic_light:
        return 3
    if has_turning_agent:
        return 2
    else:
        return 1


def get_derivative(v_ma):
    v_prime_np = np.delete(v_ma, 0, 0)
    v_prime_np = np.append(v_prime_np, 0)
    a = np.delete(v_prime_np - v_ma, -1, 0)
    return a


def is_turning_lane(dirs):
    # turning right=2 turning left=1 uturn=3
    # turning lane requires none invalid value for all points
    if len(dirs.shape) < 1:
        return 0
    dirs = dirs.flatten()
    for lane_direction in dirs:
        if lane_direction == -1:
            return 0

    dirivative_dirs = get_derivative(dirs)
    if dirivative_dirs.shape[0] < 1:
        return 0
    dirivative_dirs = [0 if abs(dirivative_dirs_) > 0.1 else dirivative_dirs_ for dirivative_dirs_ in dirivative_dirs]
    avg_dirivative = np.mean(dirivative_dirs)
    threshold = 0.005

    if avg_dirivative > threshold:
        # turning left
        if abs(abs(float(dirs[0]) - float(dirs[-2])) - math.pi) < 0.05:
            # U turn
            return 3
        return 1
    elif avg_dirivative < -threshold:
        # turning right
        return 2
    else:
        return 0


def is_turning_agent(dirs):
    # turning right=1 turning left=2
    if len(dirs.shape) < 1:
        return False
    dir_n = dirs.shape[0]
    start_dir = 0
    end_dir = 0
    for j in range(int(dir_n / 3)):
        angel = normalize_angle(dirs[j])
        if not (angel == -1 or angel == 0):
            start_dir = angel
            break

    for j in range(int(dir_n / 3)):
        angel = normalize_angle(dirs[-j - 1])
        if not (angel == -1 or angel == 0):
            end_dir = angel
            break

    if start_dir == 0 or end_dir == 0:
        return False
    new_start_dir = miniPiPi_to_zeroTwoPie(start_dir)
    new_end_dir = miniPiPi_to_zeroTwoPie(end_dir)

    if abs(new_start_dir - new_end_dir) > (math.pi / 180 * 30):
        return True
    return False


def get_next_laneid(current_lane_id, road_dic):
    lane_id_list = []
    current_lane_pts = road_dic[current_lane_id]["xyz"][:, :2]
    lane_type = road_dic[current_lane_id]["type"]
    if len(current_lane_pts.shape) > 1 and current_lane_pts.shape[0] > 2 and lane_type in [1, 2]:
        ending_pt = current_lane_pts[-1]
        for road_seg_id in road_dic.keys():
            if road_seg_id == current_lane_id:
                continue
            road_seg = road_dic[road_seg_id]
            road_pts = road_seg["xyz"][:, :2]
            road_type = road_seg["type"]
            if len(road_pts.shape) > 1 and road_pts.shape[0] > 2 and road_type in [1, 2]:
                road_starting_pt = road_pts[0]
                if euclidean_distance(pt1=ending_pt, pt2=road_starting_pt) < 0.1:
                    lane_id_list.append(road_seg_id)
    return np.array(lane_id_list)


def get_intersection(data_dic):
    intersection_dic = {"way": [{}, {}, {}, {}]}
    category = data_dic["category"]
    if category not in [3]:
        return None
    traffic_light_dic = data_dic["traffic_light"]
    tl_dic_to_return = traffic_light_dic.copy()
    road_dic = data_dic["road"]
    ego_way_inited = False
    for tl_key in traffic_light_dic.keys():
        if tl_key in road_dic.keys():
            if road_dic[tl_key]["type"] not in [1, 2]:
                # it's a bicycle lane
                continue
            # if road_dic[tl_key]["turning"] not in [1, 2]:
            #     continue
            index_offset = 0
            prev_lanes = road_dic[tl_key]["previous_lanes"]
            for prev_lane in prev_lanes:
                for way_idx, way_dic in enumerate(intersection_dic["way"]):
                    if way_idx == 0:
                        continue
                    if "lane_id" in way_dic.keys():
                        for outbound_lane_for_way in way_dic["lane_id"][0]:
                            if prev_lane == outbound_lane_for_way:
                                index_offset = way_idx
                                break
                        if index_offset != 0:
                            break

            if not ego_way_inited:
                ego_way = {}
                if len(road_dic[tl_key]["previous_lanes"]) < 1:
                    continue
                ego_lane = road_dic[tl_key]["previous_lanes"][0]
                outbound_lanes, inbound_lanes = search_same_way_lanes(ego_lane,
                                                                      road_dic,
                                                                      in_or_out=1, marking=1)
                ego_way["lane_id"] = [outbound_lanes, inbound_lanes]
                ego_way["direction"] = road_dic[outbound_lanes[0]]["dir"][-2][0]
                intersection_dic["way"][0] = ego_way
                ego_way_inited = True

            next_lanes = road_dic[tl_key]["next_lanes"]
            if next_lanes.shape[0] < 1:
                print("Warning: no next lane for a traffic light controlled lane ", tl_key, next_lanes)
                continue
            for next_lane in next_lanes:
                tl_lane_turning = road_dic[tl_key]["turning"]
                if tl_lane_turning == 0:
                    index = (2 + index_offset) % 4
                elif tl_lane_turning == 1:
                    index = (1 + index_offset) % 4
                elif tl_lane_turning == 2:
                    index = (3 + index_offset) % 4
                else:
                    continue

                if "lane_id" not in intersection_dic["way"][index].keys():
                    # DEBUG: uncomment to check lanes on each way
                    # if index == 2:
                    #     outbound_lanes, inbound_lanes = search_same_way_lanes(next_lane, road_dic, marking=1)
                    # else:
                    #     outbound_lanes, inbound_lanes = search_same_way_lanes(next_lane, road_dic)
                    outbound_lanes, inbound_lanes = search_same_way_lanes(next_lane, road_dic, marking=1)
                    intersection_dic["way"][index]["lane_id"] = [outbound_lanes, inbound_lanes]
                    if len(outbound_lanes) < 1:
                        continue
                    intersection_dic["way"][index]["direction"] = road_dic[outbound_lanes[0]]["dir"][-2][0]

                    # populate traffic light signals from given ones
                    populate_tl = True
                    if populate_tl:
                        if index == 2:
                            for outbound_lane in outbound_lanes:
                                # use ego way traffic lights
                                if len(intersection_dic["way"][0]["lane_id"]) < 1 or len(
                                        road_dic[outbound_lane]["next_lanes"]) < 1:
                                    continue
                                ego_outbound_lanes = intersection_dic["way"][0]["lane_id"][0]
                                target_lane = road_dic[outbound_lane]["next_lanes"][0]
                                turning = road_dic[target_lane]["turning"]
                                # find same direction lane
                                for ego_outbound_lane in ego_outbound_lanes:
                                    if len(road_dic[ego_outbound_lane]["next_lanes"]) < 1:
                                        continue
                                    ego_next_lane = road_dic[ego_outbound_lane]["next_lanes"][0]
                                    if road_dic[ego_next_lane]["turning"] == turning:
                                        if ego_next_lane in traffic_light_dic.keys():
                                            if target_lane not in traffic_light_dic.keys():
                                                tf_dic = {"valid": np.ones((91, 1)),
                                                          "state": traffic_light_dic[ego_next_lane]["state"]}
                                                tl_dic_to_return[target_lane] = tf_dic
                                                break
                        elif index in [1, 3]:
                            # process frame by frame
                            tl_states_unknown = []
                            ego_outbound_lanes = intersection_dic["way"][0]["lane_id"][0]
                            total_frames = traffic_light_dic[tl_key]["state"].shape[0]
                            for frame_idx in range(total_frames):
                                ego_green_light = False
                                for ego_outbound_lane in ego_outbound_lanes:
                                    if len(road_dic[ego_outbound_lane]["next_lanes"]) < 1:
                                        continue
                                    ego_next_lane = road_dic[ego_outbound_lane]["next_lanes"][0]
                                    if ego_next_lane in traffic_light_dic.keys():
                                        if traffic_light_dic[ego_next_lane]["state"][frame_idx] in [2, 3, 5, 6, 8]:
                                            ego_green_light = True
                                            break
                                if ego_green_light:
                                    tl_states_unknown.append(4)
                                else:
                                    tl_states_unknown.append(6)
                            tl_states_unknown_np = np.array(tl_states_unknown).reshape((total_frames, 1))
                            assert tl_states_unknown_np.shape == traffic_light_dic[tl_key]["state"].shape
                            for outbound_lane in outbound_lanes:
                                if len(road_dic[outbound_lane]["next_lanes"]) < 1:
                                    continue
                                target_lane = road_dic[outbound_lane]["next_lanes"][0]
                                if target_lane not in traffic_light_dic.keys():
                                    tf_dic = {"valid": np.ones((91, 1)), "state": tl_states_unknown_np}
                                    tl_dic_to_return[target_lane] = tf_dic

    data_dic["traffic_light"] = tl_dic_to_return
    # print(intersection_dic)
    return intersection_dic


def search_same_way_lanes(one_inbound_lane_id, road_dic, in_or_out=0, marking=0):
    # in_or_out: 0=inbound provided, 1=outbound provided
    outbound_lanes = []
    inbound_lanes = []
    # search from these inbound lanes
    xy_np = road_dic[one_inbound_lane_id]["xyz"][:, :2]
    dir_np = road_dic[one_inbound_lane_id]["dir"]
    if len(xy_np.shape) < 1 or len(dir_np.shape) < 1:
        return None
    if in_or_out:
        # outbound given
        entry_pt = xy_np[-1]
        entry_dir = dir_np[-2]
    else:
        # inbound given
        entry_pt = xy_np[0]
        entry_dir = dir_np[0]

    entry_pts_list = [entry_pt]
    out_pts_list = [entry_pt]
    pt_dist_threshold = SAME_WAY_LANES_SEARCHING_DIST_THRESHOLD
    dir_threshold = SAME_WAY_LANES_SEARCHING_DIRECTION_THRESHOLD
    if in_or_out:
        # outbound given
        entry_dir = normalize_angle(entry_dir + math.pi)

    for road_seg_key in road_dic.keys():
        # if road_seg_key == tl_key:
        #     continue
        if road_dic[road_seg_key]["type"] not in [1, 2]:
            continue
        target_xy_np = road_dic[road_seg_key]["xyz"][:, :2]
        target_dir_np = road_dic[road_seg_key]["dir"]
        if len(target_xy_np.shape) < 1 or len(target_dir_np.shape) < 1:
            continue
        if target_xy_np.shape[0] < 2:
            print("ERROR: lane target_xy_np size too short. ", road_seg_key, target_xy_np)
            continue
        if target_dir_np.shape[0] < 3:
            # print("ERROR: lane target_dir_np size too short. ", road_seg_key, target_dir_np)
            # [[-1.71647068], [0.]]
            continue
        target_seg_entry_pt = target_xy_np[0]
        target_starting_dir = target_dir_np[0]
        if abs(normalize_angle(float(target_starting_dir) - float(entry_dir))) < dir_threshold:
            for one_entry_pt in entry_pts_list:
                # disth, distv = handvdistance(one_entry_pt, target_seg_entry_pt, entry_dir)
                dist = euclidean_distance(one_entry_pt, target_seg_entry_pt)
                if dist < pt_dist_threshold:
                    # if abs(disth) < pt_dist_threshold and abs(distv) < 5:
                    inbound_lanes.append(road_seg_key)
                    entry_pts_list.append(target_seg_entry_pt)
                    if marking:
                        road_dic[road_seg_key]["marking"] = 4
                    break
        target_seg_ending_pt = target_xy_np[-1]
        target_ending_dir = target_dir_np[-2]
        if abs(normalize_angle(float(target_ending_dir) - float(normalize_angle(entry_dir + math.pi)))) < dir_threshold:
            for one_entry_pt in out_pts_list:
                dist = euclidean_distance(one_entry_pt, target_seg_ending_pt)
                if dist < pt_dist_threshold:
                    outbound_lanes.append(road_seg_key)
                    out_pts_list.append(target_seg_ending_pt)
                    if marking:
                        road_dic[road_seg_key]["marking"] = 5
                    break

    return [outbound_lanes, inbound_lanes]


def search_lanes_for_predict_agents(data_dic):
    lanes_dic = {}
    for agent_id in data_dic['agent']:
        if data_dic['agent'][agent_id]['to_predict']:
            detected_starting_lane = []
            detected_ending_lane = []
            for i in range(91):
                # if 80 > i > 10:
                #     continue
                closest_dist = 999999
                closest_lane = None
                pose = data_dic['agent'][agent_id]['pose'][i]
                if pose[0] == -1:
                    continue
                width = data_dic['agent'][agent_id]['shape'][0, 0]
                for each_lane in data_dic['road']:
                    road_xy = data_dic['road'][each_lane]['xyz'][:, :2]
                    if road_xy.shape[0] < 3:
                        continue
                    if data_dic['road'][each_lane]['type'][0] not in [1, 2]:
                        continue
                    # if data_dic['road'][each_lane]['type'] != 1:
                    #     continue
                    for j, each_xy in enumerate(road_xy):
                        road_yaw = data_dic['road'][each_lane]['dir'][j]
                        dist = euclidean_distance(each_xy, pose[:2])
                        yaw_diff = abs(normalize_angle(pose[3] - road_yaw))
                        if dist < width * 2 and dist < closest_dist and yaw_diff < math.pi/180*30:
                            closest_lane = each_lane
                            closest_dist = dist
                if closest_lane is not None:
                    if i <= 10 and closest_lane not in detected_starting_lane:
                        detected_starting_lane.append(closest_lane)
                    elif i >= 11 and closest_lane not in detected_ending_lane:
                        if closest_lane not in detected_starting_lane:
                            detected_ending_lane.append(closest_lane)

            # loop for three random following lanes
            lanes_as_target = []
            if len(detected_ending_lane) > 0:
                # randomly choose a lane
                i = random.randint(0, len(detected_ending_lane)-1)
                each_lane = detected_ending_lane[i]
                counter = 0
                while counter < 10:
                    lanes_as_target.append(each_lane)
                    data_dic['road'][each_lane]['marking'] = 1
                    next_lanes = data_dic['road'][each_lane]['next_lanes']
                    if next_lanes.shape[0] < 1:
                        break
                    i = random.randint(0, next_lanes.shape[0]-1)
                    each_lane = next_lanes[i]
                    counter += 1

            lanes_dic[agent_id] = lanes_as_target
            # if len(lanes_as_target) > 1:
            #     break

    return lanes_dic


def handvdistance(pt1, pt2, direction):
    new_pt2_x, new_pt2_y = rotate(pt1, pt2, -direction)
    return pt1[0] - new_pt2_x, pt1[1] - new_pt2_y


class WaymoDL:
    def __init__(self, filepath, file_to_start=None, max_file_number=None, gt_relation_path=None):
        tf_example_dir = filepath
        self.file_names = [os.path.join(tf_example_dir, f) for f in os.listdir(tf_example_dir) if
                           os.path.isfile(os.path.join(tf_example_dir, f)) and 'tfrecord' in f]
        self.file_names.sort()
        self.current_file_index = FILE_TO_START
        self.current_file_total_scenario = 0
        if file_to_start is not None and file_to_start >= 0:
            self.current_file_index = file_to_start
        self.max_file_number = max_file_number
        self.start_file_number = self.current_file_index
        self.end = False
        self.current_dataset = None

        self.current_scenario_index = SCENE_TO_START

        self.loaded_playback = None
        self.gt_relation_path = gt_relation_path

        # self.load_new_file()
        # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        print("Data Loader Initialized Waymo: ", self.start_file_number, FILE_TO_START, self.current_file_index, file_to_start, self.current_scenario_index, self.max_file_number, len(self.file_names))

    def load_new_file(self, first_file=False):
        if self.max_file_number is not None and self.current_file_index >= (self.start_file_number + self.max_file_number):
            print("Reached max number:", self.current_file_index, self.max_file_number, self.start_file_number, len(self.file_names))
            self.end = True
            return
        if self.current_file_index < len(self.file_names):
            if "." not in self.file_names[self.current_file_index] or "tf" not in self.file_names[self.current_file_index]:
                print("skipping invalid file: ", self.file_names[self.current_file_index])
                self.current_file_index += 1
                self.load_new_file(first_file=first_file)
                return
            print("Loading file from: ", self.file_names[self.current_file_index], " with index of ", self.current_file_index)
            # self.current_file_index += 1
            self.current_file_total_scenario = 0
            self.current_dataset = tf.data.TFRecordDataset(self.file_names[self.current_file_index], compression_type='')
            for _ in self.current_dataset.as_numpy_iterator():
                self.current_file_total_scenario += 1
            if not first_file:
                self.current_scenario_index = 0
            print(" with ", self.current_file_total_scenario, " scenarios and current is ", self.current_scenario_index)
        else:
            self.end = True

    def get_datadic_fromTFRecord(self, data, scenario_id, process_intersection=True, include_relation=True,
                                 loading_prediction_relation=False, detect_gt_relation=False,
                                 agent_to_interact_np=None, agent_only=False,
                                 only_predict_interest_agents=False, filter_config={}):
        parsed = tf.io.parse_single_example(data, features_description)
        text = ['past', 'current', 'future']

        agent_id_np = parsed['state/id'].numpy()
        agent_type_np = parsed['state/type'].numpy()
        agent_to_predict_np = parsed['state/tracks_to_predict'].numpy()
        agent_is_sdc_np = parsed['state/is_sdc'].numpy()

        agent_past_x_np = parsed['state/past/x'].numpy()
        agent_past_y_np = parsed['state/past/y'].numpy()
        agent_past_z_np = parsed['state/past/z'].numpy()
        agent_past_yaw_np = parsed['state/past/bbox_yaw'].numpy()
        agent_past_length_np = parsed['state/past/length'].numpy()
        agent_past_width_np = parsed['state/past/width'].numpy()
        agent_past_height_np = parsed['state/past/height'].numpy()
        agent_past_vx_np = parsed['state/past/velocity_x'].numpy()
        agent_past_vy_np = parsed['state/past/velocity_y'].numpy()
        agent_past_vel_yaw_np = parsed['state/past/vel_yaw'].numpy()
        agent_past_valid_np = parsed['state/past/valid'].numpy()

        agent_current_x_np = parsed['state/current/x'].numpy()
        agent_current_y_np = parsed['state/current/y'].numpy()
        agent_current_z_np = parsed['state/current/z'].numpy()
        agent_current_yaw_np = parsed['state/current/bbox_yaw'].numpy()
        agent_current_length_np = parsed['state/current/length'].numpy()
        agent_current_width_np = parsed['state/current/width'].numpy()
        agent_current_height_np = parsed['state/current/height'].numpy()
        agent_current_vx_np = parsed['state/current/velocity_x'].numpy()
        agent_current_vy_np = parsed['state/current/velocity_y'].numpy()
        agent_current_vel_yaw_np = parsed['state/current/vel_yaw'].numpy()
        agent_current_valid_np = parsed['state/current/valid'].numpy()

        agent_future_x_np = parsed['state/future/x'].numpy()
        agent_future_y_np = parsed['state/future/y'].numpy()
        agent_future_z_np = parsed['state/future/z'].numpy()
        agent_future_yaw_np = parsed['state/future/bbox_yaw'].numpy()
        agent_future_length_np = parsed['state/future/length'].numpy()
        agent_future_width_np = parsed['state/future/width'].numpy()
        agent_future_height_np = parsed['state/future/height'].numpy()
        agent_future_vx_np = parsed['state/future/velocity_x'].numpy()
        agent_future_vy_np = parsed['state/future/velocity_y'].numpy()
        agent_future_vel_yaw_np = parsed['state/future/vel_yaw'].numpy()
        agent_future_valid_np = parsed['state/future/valid'].numpy()

        skip = False
        interact_only = False
        if interact_only:
            skip = True

        agent_dic = {}
        type_anomaly = 0

        for i in range(len(agent_id_np)):
            agent_id = int(agent_id_np[i])
            agent_type = int(agent_type_np[i])
            agent_to_predict = int(agent_to_predict_np[i])
            agent_to_interact = int(agent_to_interact_np[i])
            agent_is_sdc = int(agent_is_sdc_np[i])
            poses_np = np.empty([1, 4])
            shapes_np = np.empty([1, 3])
            speeds_np = np.empty([1, 2])
            is_init = True

            if interact_only:
                if agent_to_interact == 1:
                    # invalid = -1, valid = 1
                    skip = False
            if agent_type not in [1, 2, 3] or agent_id == -1:
                type_anomaly += 1
                continue
            for time_state in text:
                if time_state == 'past':
                    pose_np = np.transpose(np.vstack(
                            (agent_past_x_np[i], agent_past_y_np[i], agent_past_z_np[i], agent_past_yaw_np[i])))
                    speed_np = velvector2value(agent_past_vx_np[i], agent_past_vy_np[i])
                    vyaw_np = agent_past_vel_yaw_np[i]
                    speed_vyaw_np = np.transpose(np.vstack((speed_np, vyaw_np)))

                elif time_state == 'future':
                    pose_np = np.transpose(np.vstack(
                            (agent_future_x_np[i], agent_future_y_np[i], agent_future_z_np[i],
                             agent_future_yaw_np[i])))
                    speed_np = velvector2value(agent_future_vx_np[i], agent_future_vy_np[i])
                    vyaw_np = agent_future_vel_yaw_np[i]
                    speed_vyaw_np = np.transpose(np.vstack((speed_np, vyaw_np)))

                elif time_state == 'current':
                    pose_np = np.transpose(np.vstack((agent_current_x_np[i], agent_current_y_np[i],
                                                      agent_current_z_np[i], agent_current_yaw_np[i])))
                    speed_np = velvector2value(agent_current_vx_np[i], agent_current_vy_np[i])
                    vyaw_np = agent_current_vel_yaw_np[i]
                    speed_vyaw_np = np.transpose(np.vstack((speed_np, vyaw_np)))

                poses_np = np.vstack((poses_np, pose_np))
                shapes_np = np.transpose(np.vstack(
                        (agent_current_width_np[i][0], agent_current_length_np[i][0], agent_current_height_np[i][0])))
                speeds_np = np.vstack((speeds_np, speed_vyaw_np))

                if is_init:
                    is_init = False
                    poses_np = np.delete(poses_np, 0, 0)
                    shapes_np = np.delete(shapes_np, 0, 0)
                    speeds_np = np.delete(speeds_np, 0, 0)

            if agent_id in agent_dic.keys():
                print("WARNING: duplicated agent")
                agent_dic[agent_id]['pose'] = np.vstack((agent_dic[agent_id]['pose'], poses_np))
                agent_dic[agent_id]['shape'] = np.vstack((agent_dic[agent_id]['shape'], shapes_np))
                agent_dic[agent_id]['speed'] = np.vstack((agent_dic[agent_id]['speed'], speeds_np))
            else:
                # filter logic
                filter = False
                if 'filter_static' in filter_config:
                    if pose_np[0, 0] == -1 or pose_np[-1, 0] == -1:
                        filter = True
                    elif euclidean_distance(poses_np[0, :2], poses_np[-1, :2]) < 1:
                        filter = True
                if 'filter_non_vehicle' in filter_config:
                    if agent_type != 1:
                        filter = True
                    elif shapes_np[0][0] == -1:
                        # filter out vehicles barely detected
                        filter = True
                if True:
                    # filter bus
                    if shapes_np[0][0] > 10 or shapes_np[0][1] > 10:
                        filter = True

                # print(f"debug: {agent_id} {agent_type} {filter} {filter_config} {agent_type_np[i]} {shapes_np}")

                if not filter:
                    new_dic = {'pose': poses_np, 'shape': shapes_np,
                               'speed': speeds_np, 'type': np.array(agent_type),
                               'to_predict': np.array(agent_to_predict),
                               'to_interact': np.array(agent_to_interact),
                               'is_sdc': agent_is_sdc,
                               'speed_xy': (agent_current_vx_np[i], agent_current_vy_np[i])}

                    agent_dic[agent_id] = new_dic

                    if not poses_np.shape[0] == 91:
                        print("frame number error: ", poses_np.shape)

        # get relation as edges [[A->B], ..]
        edges = []
        edge_type = []

        if include_relation and not skip:
            if loading_prediction_relation:
                # currently only work for one pair relation visualization
                if self.gt_relation_path is not None:
                    loading_file_name = self.gt_relation_path
                    with open(loading_file_name, 'rb') as f:
                        loaded_dictionary = pickle.load(f)

                # file_to_read = open(loading_file_name, "rb")
                # loaded_dictionary = pickle.load(file_to_read)
                # file_to_read.close()
                # old version
                # old_version = False
                old_version = True
                one_pair = False
                multi_time_edges = True

                if old_version:
                    if scenario_id in loaded_dictionary:
                        relation = loaded_dictionary[scenario_id]
                        edges = []
                        for each_info in relation:
                            if len(each_info) == 3:
                                agent_inf, agent_reactor, relation_label = each_info
                            elif len(each_info) == 4:
                                agent_inf, agent_reactor, inf_passing_frame, reactor_passing_frame = each_info
                            else:
                                assert False, f'Unknown relation format loaded {each_info}'
                        # for agent_inf, agent_reactor, relation_label in relation:
                            edges.append([agent_inf, agent_reactor, 0, 1])
                    else:
                        print("scenario_id not found in loaded dic 1:", scenario_id, list(loaded_dictionary.keys())[0])
                        # skip unrelated scenarios
                        skip = True
                elif one_pair:
                    threshold = 0.8
                    if scenario_id.encode() in loaded_dictionary:
                        relation = loaded_dictionary[scenario_id.encode()]
                        edges = []
                        for reactor_id in relation:
                            # print("debug: ", reactor_id, relation[reactor_id])
                            labels = relation[reactor_id]['pred_inf_label']
                            agent_ids = relation[reactor_id]['pred_inf_id']
                            scores = relation[reactor_id]['pred_inf_scores']
                            for i, label in enumerate(labels):
                                if label == 1 and scores[i][1] > threshold:
                                    edges.append([agent_ids[i], reactor_id, 0, 1])
                    else:
                        print("scenario_id not found in loaded dic 2:", scenario_id.encode(), list(loaded_dictionary.keys())[0])
                        skip = True
                elif multi_time_edges:
                    threshold = 0.8
                    if scenario_id.encode() in loaded_dictionary:
                        relation = loaded_dictionary[scenario_id.encode()]
                        edges = {}
                        for reactor_id in relation:
                            for time_offset in relation[reactor_id]:
                                # print("test: ", relation[reactor_id][time_offset])
                                labels = relation[reactor_id][time_offset]['pred_inf_label']
                                agent_ids = relation[reactor_id][time_offset]['pred_inf_id']
                                scores = relation[reactor_id][time_offset]['pred_inf_scores']
                                time_offset += 11
                                for i, label in enumerate(labels):
                                    if label == 1 and scores[i][1] > threshold:
                                        if time_offset not in edges:
                                            edges[time_offset] = []
                                        # rescale the score
                                        bottom = 0.6
                                        score = (scores[i][1] - threshold)/(1-threshold)*(1-bottom)+bottom
                                        edges[time_offset].append([agent_ids[i], reactor_id, 0, score])
                    else:
                        print("scenario_id not found in loaded dic 3:", scenario_id.encode(), list(loaded_dictionary.keys())[0])
                        skip = True
                else:
                    if scenario_id.encode() in loaded_dictionary:
                        relation = loaded_dictionary[scenario_id.encode()]
                        agent_ids = []
                        edges = []
                        for reactor_id in relation:
                            print("debug: ", reactor_id, relation[reactor_id])
                            inf_ids = relation[reactor_id]['pred_inf_ids']
                            inf_scores = relation[reactor_id]['pred_inf_scores']
                            inf_ids_list = inf_ids.tolist()
                            for k, inf_id in enumerate(inf_ids_list):
                                if int(inf_id) != 0 and inf_scores[k] > 0.5:
                                    edges.append([int(inf_id), int(reactor_id), 0, 1])
                                if inf_scores[k] > 0.8:
                                    print('skipping over 0.8 scenes')
                                    skip = True

                    else:
                        print(f"scenario_id not found in loaded dic: {scenario_id}. Loaded sample: {list(loaded_dictionary.keys())[0]}")
                        # skip unrelated scenarios
                        skip = True

                # inspect only on inconsistant cases
                if False:
                    edges_detected = get_relation_on_crossing(agent_dic=agent_dic, only_prediction_agents=only_predict_interest_agents)
                    if len(edges) > 0:
                        if len(edges_detected) > 0:
                            if edges_detected[0][0] == edges[0][0] and edges_detected[0][1] == edges[0][1]:
                                print("skip duplicate edges")
                                skip = True
                    # end of inspect codes
                    else:
                        if len(edges_detected) < 1:
                            print("skip no edge detected scenario")
                            skip = True
                    if not skip:
                        print("detected edge:", edges_detected, "\nprediction edge", edges)
            elif detect_gt_relation:
                edges = get_relation_on_crossing(agent_dic=agent_dic, only_prediction_agents=only_predict_interest_agents)
                form_a_tree = False
                if not only_predict_interest_agents and form_a_tree:
                    edges = form_tree_from_edges(edges)

                temp_loading_flag = True
                if temp_loading_flag and not skip:
                    for edge in edges:
                        if len(edge) != 4:
                            # [agent_id_influencer, agent_id_reactor, frame_idx_reactor_passing_cross_point, abs(frame_diff)]
                            print("invalid edge: ", edge)
                            skip = True
                            break
                        # one type per agent
                        # relation type: 1-vv, 2-vp, 3-vc, 4-others
                        agent_types = []
                        agent_id1, agent_id2, _, _ = edge
                        for agent_id in agent_dic:
                            if agent_id in [agent_id1, agent_id2]:
                                agent_types.append(agent_dic[agent_id]['type'])
                        if len(agent_types) != 2:
                            print("WARNING: Skipping an solo interactive agent scene - ", str(agent_types),
                                  str(scenario_id))
                            skip = True
                        else:
                            if agent_types[0] == 1 and agent_types[1] == 1:
                                edge_type.append(1)
                            elif 1 in agent_types and 2 in agent_types:
                                edge_type.append(2)
                            elif 1 in agent_types and 3 in agent_types:
                                edge_type.append(3)
                            else:
                                print("other type:", agent_types)
                                edge_type.append(4)

        if not agent_only:
            road_dic = {}
            parsed['roadgraph_samples/dir'] = parsed['roadgraph_samples/dir'].numpy()
            parsed['roadgraph_samples/id'] = parsed['roadgraph_samples/id'].numpy()
            parsed['roadgraph_samples/xyz'] = parsed['roadgraph_samples/xyz'].numpy()
            parsed['roadgraph_samples/type'] = parsed['roadgraph_samples/type'].numpy()
            parsed['roadgraph_samples/valid'] = parsed['roadgraph_samples/valid'].numpy()

            road_dir_np = parsed['roadgraph_samples/dir']
            road_id_np = parsed['roadgraph_samples/id']
            road_xy_np = parsed['roadgraph_samples/xyz']
            road_type_np = parsed['roadgraph_samples/type']
            road_valid_np = parsed['roadgraph_samples/valid']

            for i in range(len(road_dir_np)):
                road_id = int(road_id_np[i][0])
                if road_id in road_dic.keys():
                    if int(road_valid_np[i][0]) == 1:
                        road_dic[road_id]['dir'] = np.vstack(
                                (road_dic[road_id]['dir'], np.array(vector2radical(road_dir_np[i]))))
                        road_dic[road_id]['xyz'] = np.vstack((road_dic[road_id]['xyz'], road_xy_np[i][:3]))
                else:
                    # init a road_dic
                    if int(road_valid_np[i][0]) == 1:
                        new_dic = {'dir': np.array(vector2radical(road_dir_np[i])),
                                   # 'xy': road_xy_np[i][:2].reshape(1, 2),
                                   'type': road_type_np[i], 'turning': -1,
                                   'next_lanes': np.array([]), 'previous_lanes': np.array([]),
                                   'outbound': 0, 'marking': 0,
                                   'vector_dir': road_dir_np[i], 'xyz': road_xy_np[i][:3].reshape(1, 3), }
                        road_dic[road_id] = new_dic

            # NOTE: traffic needs to be processed after road
            traffic_dic = {}
            traffic_past_laneid_np = parsed['traffic_light_state/past/id'].numpy()
            traffic_past_valid_np = parsed['traffic_light_state/past/valid'].numpy()
            traffic_past_state_np = parsed['traffic_light_state/past/state'].numpy()

            traffic_current_laneid_np = parsed['traffic_light_state/current/id'].numpy()
            traffic_current_valid_np = parsed['traffic_light_state/current/valid'].numpy()
            traffic_current_state_np = parsed['traffic_light_state/current/state'].numpy()

            traffic_future_laneid_np = parsed['traffic_light_state/future/id'].numpy()
            traffic_future_valid_np = parsed['traffic_light_state/future/valid'].numpy()
            traffic_future_state_np = parsed['traffic_light_state/future/state'].numpy()

            for idx, laneid in enumerate(traffic_current_laneid_np[0]):
                valids_np = np.empty([1, 1])
                states_np = np.empty([1, 1])
                is_init = True
                for time_state in text:
                    if time_state == 'past':
                        valid_np = traffic_past_valid_np[:, idx].reshape(10, 1)
                        state_np = traffic_past_state_np[:, idx].reshape(10, 1)
                    elif time_state == 'future':
                        valid_np = traffic_future_valid_np[:, idx].reshape(80, 1)
                        state_np = traffic_future_state_np[:, idx].reshape(80, 1)
                    elif time_state == 'current':
                        valid_np = traffic_current_valid_np[:, idx].reshape(1, 1)
                        state_np = traffic_current_state_np[:, idx].reshape(1, 1)

                    valids_np = np.vstack((valids_np, valid_np))
                    states_np = np.vstack((states_np, state_np))

                    if is_init:
                        is_init = False
                        valids_np = np.delete(valids_np, 0, 0)
                        states_np = np.delete(states_np, 0, 0)

                new_dic = {'valid': valids_np, 'state': states_np}
                traffic_dic[laneid] = new_dic
                if laneid in road_dic.keys():
                    # mark 1 on outbound lanes
                    road_dic[laneid]["outbound"] = 1
                else:
                    if laneid != -1:
                        print("WARNING: no lane found for traffic light while parsing", laneid)

                if not states_np.shape[0] == 91:
                    print("ERROR PARSING TRAFFIC LIGHTS - frame number error: ", states_np.shape)

            # calculate turning directions, next lanes
            for road_key in road_dic.keys():
                lane_dic = road_dic[road_key]
                assert road_dic[road_key]["xyz"].shape[1] == 3
                dirs = lane_dic["dir"].flatten()
                lane_type = int(lane_dic["type"])
                lane_turning = is_turning_lane(dirs)
                if lane_type in [1, 2]:
                    next_lanes_np = get_next_laneid(current_lane_id=road_key,
                                                    road_dic=road_dic)
                    lane_dic["next_lanes"] = next_lanes_np
                    lane_dic["turning"] = lane_turning
                else:
                    lane_dic["turning"] = -1
                road_dic[road_key] = lane_dic

            for road_key in road_dic.keys():
                if int(road_dic[road_key]["type"]) not in [1, 2]:
                    continue
                next_lanes_np = road_dic[road_key]["next_lanes"]
                for next_lane_id in next_lanes_np:
                    road_dic[next_lane_id]["previous_lanes"] = np.insert(road_dic[next_lane_id]["previous_lanes"],
                                                                         0, int(road_key))
        else:
            road_dic = {}
            traffic_dic = {}

        # mark still agents is the past
        for agent_id in agent_dic:
            is_still = False
            for i in range(10):
                if agent_dic[agent_id]['pose'][i, 0] == -1:
                    continue
                if euclidean_distance(agent_dic[agent_id]['pose'][i, :2],
                                      agent_dic[agent_id]['pose'][10, :2]) < 1:
                    is_still = True
            agent_dic[agent_id]['still_in_past'] = is_still

        data_to_return = {
            "road": road_dic,
            "agent": agent_dic,
            "traffic_light": traffic_dic,
            'raw': parsed
        }

        # sanity check
        if agent_dic is None or road_dic is None or traffic_dic is None or parsed is None:
            print("Invalid Scenario Loaded: ")
            skip = True

        category = classify_scenario(data_to_return)
        data_to_return["category"] = category
        data_to_return['scenario_str'] = scenario_id

        data_to_return['edges'] = edges
        data_to_return['skip'] = skip
        data_to_return['edge_type'] = edge_type

        if process_intersection:
            intersection_dic = get_intersection(data_to_return)
            data_to_return["intersection"] = intersection_dic

        data_to_return['dataset'] = 'Waymo'

        # data_to_return['lanes_traveled'] = search_lanes_for_predict_agents(data_to_return)
        return data_to_return

    def get_next(self, process_intersection=True, relation=False, agent_only=False, only_predict_interest_agents=False,
                 filter_config={}, detect_gt_relation=False, load_prediction=True, seconds_in_future=None):
        new_files_loaded = False

        if self.current_dataset is None:
            # init the dataset and load file 0 from starting index
            self.load_new_file(first_file=True)
        else:
            self.current_scenario_index += 1

        if not self.current_scenario_index < self.current_file_total_scenario:
            self.current_file_index += 1
            self.load_new_file()
            new_files_loaded = True

        if self.end:
            return None, new_files_loaded

        if self.current_dataset is None:
            # if is still None, end the loop
            print("Ended with broken dataset")
            return None, new_files_loaded

        for idx, data in enumerate(self.current_dataset.as_numpy_iterator()):
            if idx != self.current_scenario_index:
                continue
            scenario_features = {
                'scenario/id':
                    tf.io.FixedLenFeature([], tf.string, default_value=None),
                'state/objects_of_interest': tf.io.FixedLenFeature([128], tf.int64, default_value=None)
            }
            features_description_scenario = {}
            features_description_scenario.update(scenario_features)
            parsed = tf.io.parse_single_example(data, features_description_scenario)
            scenario_id = parsed['scenario/id'].numpy()
            objects_of_interest = parsed['state/objects_of_interest'].numpy()

            # if online_planning:
            #     load_prediction = True
            #     calculate_gt_relation = False
            # else:
            #     load_prediction = True
            #     calculate_gt_relation = False

            data_to_return = self.get_datadic_fromTFRecord(data, process_intersection=process_intersection,
                                                           scenario_id=scenario_id.decode(),
                                                           include_relation=relation,
                                                           agent_to_interact_np=objects_of_interest,
                                                           agent_only=agent_only,
                                                           only_predict_interest_agents=only_predict_interest_agents,
                                                           loading_prediction_relation=load_prediction,
                                                           detect_gt_relation=detect_gt_relation,
                                                           filter_config=filter_config)
            data_to_return['scenario'] = scenario_id

            # loaded_dic = load('relation_inspected_scenarios_interVal.pickle')
            # # if scenario_id.decode() not in loaded_dic['scenarios_to_keep']:  # loaded_dic['scenarios_to_inspect']:
            # if scenario_id.decode() not in loaded_dic['scenarios_to_keep'] and scenario_id.decode() not in loaded_dic['scenarios_to_inspect']: #loaded_dic['scenarios_to_inspect']:
            # # # if scenario_id.decode() in loaded_dic['scenarios_to_exclude']:  # loaded_dic['scenarios_to_inspect']:
            # # # if scenario_id.decode() not in loaded_dic['scenarios_to_keep'] and scenario_id.decode() not in loaded_dic['scenarios_to_inspect']:
            #     data_to_return['skip'] = True
            #     print("Skipping scenario with tag: ", scenario_id.decode())
            # else:
            #     print("Loaded scenario with tag: ", scenario_id.decode())
                # print("Loaded scenario detail: ", loaded_dic['scenarios_to_inspect'][scenario_id.decode()])
            # if scenario_id.decode() not in ['76b3f3dad2e3a73f', '101442ea50efcadd', 'a3f02d58fb2b4d8e', '949a396596592c0e', '6c21e5f9b78b8efb', '5d22026d09cef6e1', '44bf09b1d43b233', 'd3f03eefbb18aeaa', '43b90281ed0d9ff5', '4ab7dfa97899fe1b', '8648a0b288464dd3', 'be0994a4fd79e08d']:
            #     data_to_return['skip'] = True

            # loaded_dic = load('target_scenes_Sep14.pickle')
            # if scenario_id.decode() not in loaded_dic:
            #     data_to_return['skip'] = True
            # else:
            #     print("Got Interesting Scenario: ", loaded_dic[scenario_id.decode()])

            # loaded_list = load('colliding_scenarios.pickle')
            # print("test:")
            # if scenario_id.decode() not in loaded_list:
            #     data_to_return['skip'] = True

            return data_to_return, new_files_loaded
        print("ERROR: found no scenario at all ", self.current_scenario_index)
        return None, new_files_loaded

    def get_next_from_playback(self, process_intersection=True, relation=True, agent_only=False,
                               only_predict_interest_agents=False,
                               filter_config={}, playback_dir=None):
        '''
        playback one file only
        '''
        new_files_loaded = False
        if self.end:
            return None, new_files_loaded

        if self.current_dataset is None:
            # dataset = tf.data.TFRecordDataset(
            #     "../Waymo/motion_dataset/tf_example/training/training_tfexample.tfrecord-00000-of-01000",
            #     compression_type='')
            file_name = playback_dir.split('/')[-1].split('.playback')[0]
            dataset = None
            for each_file in self.file_names:
                if each_file.find(file_name) == -1:
                    continue
                else:
                    print("file found: ", playback_dir, each_file)
                    dataset = tf.data.TFRecordDataset(each_file, compression_type='')
                    break
            assert dataset is not None, each_file
            self.current_dataset = dataset
            self.current_file_total_scenario = 0
            self.current_scenario_index = 0
            for _ in self.current_dataset.as_numpy_iterator():
                self.current_file_total_scenario += 1
        loaded_dictionary = load(playback_dir)
        for idx, data in enumerate(self.current_dataset.as_numpy_iterator()):
            if idx < self.current_scenario_index:
                continue
            scenario_features = {
                'scenario/id':
                    tf.io.FixedLenFeature([], tf.string, default_value=None),
                'state/objects_of_interest': tf.io.FixedLenFeature([128], tf.int64, default_value=None)
            }
            features_description_scenario = {}
            features_description_scenario.update(scenario_features)
            parsed = tf.io.parse_single_example(data, features_description_scenario)
            scenario_id = parsed['scenario/id'].numpy()
            objects_of_interest = parsed['state/objects_of_interest'].numpy()

            scenario_id_str = scenario_id.decode()
            first_key = list(loaded_dictionary.keys())[0]
            print("playing back: ", scenario_id_str, first_key, type(loaded_dictionary[first_key]))
            if self.current_scenario_index + 1 < self.current_file_total_scenario:
                self.current_scenario_index += 1
            else:
                self.end = True
                return None, new_files_loaded
                # self.load_new_file()
                # new_files_loaded = True

            if scenario_id_str not in loaded_dictionary:
                continue

            if isinstance(loaded_dictionary[scenario_id_str], list):
                loaded_scenario = loaded_dictionary[scenario_id_str]
            elif isinstance(loaded_dictionary[scenario_id_str], dict):
                loaded_scenario = loaded_dictionary[scenario_id_str]
            else:
                print("ERROR: Unknown type loaded. ", loaded_dictionary[scenario_id_str])

            load_prediction = False
            calculate_gt_relation = False
            data_to_return = self.get_datadic_fromTFRecord(data, process_intersection=process_intersection,
                                                           scenario_id=scenario_id.decode(),
                                                           include_relation=relation,
                                                           agent_to_interact_np=objects_of_interest,
                                                           agent_only=agent_only,
                                                           only_predict_interest_agents=only_predict_interest_agents,
                                                           loading_prediction_relation=load_prediction,
                                                           detect_gt_relation=calculate_gt_relation,
                                                           filter_config=filter_config)
            if isinstance(loaded_scenario, list):
                loaded_scenario = loaded_scenario[0]

            data_to_return['agent'] = loaded_scenario['agent']
            data_to_return['scenario'] = scenario_id
            data_to_return['predicting'] = loaded_scenario['predicting']
            data_to_return['predicting']['trajectory_to_mark'] = []
            for each_agent_id in loaded_scenario['predicting']['relevant_agents']:
                data_to_return['agent'][each_agent_id]['action'] = 'controlled'
            data_to_return['predicting']['points_to_mark'] = []

            # baseline_0616 00004
            # if scenario_id.decode() not in ['7ebfbf5231f8f84f', '784186b60285c8c8', 'd7668001dd7d25f5', 'e9e7208c55cbe2a3', '72c8b3f42a197ced', '488a48d08eb2f2b0', 'c65cf8e328f3bca0', '3e1a360fa4468e73', 'f193b3faca4112f3', '92c88821c818ef59', '6d3e5ccbeac05a79', '7bba17482771641e', 'cdb4c066c69277e6', 'a35bf629e4e15b19', 'eee2efd1ebf607', '5f3f83f8a0891f41', '3318add8b6f95a44', '3090a30281eb8c80', '3db7fdb52fc6f2e2', '4111e60c2a167c80', '53d71a137a11bed1']:
            #     data_to_return['skip'] = True

            return data_to_return, new_files_loaded
        print("ERROR: found no scenario at all ", self.current_scenario_index)
        return None, new_files_loaded


# from nuscenes import NuScenes
# class NuScenesDL(NuScenes):
#     def __init__(self,
#                  version: str='v1.0-mini',
#                  dataroot: str='./',
#                  verbose: bool=True,
#                  map_resolution: float=0.1,
#                  predict_class: int = 1,
#                  resolution_meters: float = 1.0,
#                  scenario_to_start: int=None,
#                  ending_scenario_num: int=None,
#                  gt_relation_path: str=None):
#         '''
#         :param predict_class: 1 for vehicle, including bicycle; 2 for human; 0 for objects and animals
#         '''
#         # Handling the NuScenes Packages
#         from nuscenes.prediction import PredictHelper
#         from nuscenes.eval.prediction.splits import get_prediction_challenge_split
#
#         super().__init__(version, dataroot, verbose, map_resolution)
#         # train: 32186 train_val: 8560 val: 9041
#         self.helper = PredictHelper(self)
#         self.datas = get_prediction_challenge_split("train", dataroot=dataroot)
#         # self.datas = get_prediction_challenge_split("mini_train", dataroot=dataroot)
#         self.dataroot = dataroot
#         self.predict_class = predict_class  # 需要预测轨迹的agent类别
#         self.resolution_meters = resolution_meters  # center_line的点间距
#
#         self.LineMapping = {
#             'LANE_CONNECTOR': 1,
#             'LANE': 2,
#             'NIL': 6,  # to be modified
#             'DOUBLE_DASHED_WHITE': 6,
#             'SINGLE_ZIGZAG_WHITE': 7,
#             'SINGLE_SOLID_WHITE': 7,
#             'DOUBLE_SOLID_WHITE': 8,
#             'SINGLE_SOLID_YELLOW': 11,
#             'ROAD_DIVIDER': 12,
#             'PED_CROSSING': 18,
#         }
#         self.ClassMapping = {
#             'vehicle.bicycle': 1,
#             'vehicle.bus.bendy': 1,
#             'vehicle.bus.rigid': 1,
#             'vehicle.car': 1,
#             'vehicle.construction': 1,
#             'vehicle.emergency.ambulance': 1,
#             'vehicle.emergency.police': 1,
#             'vehicle.motorcycle': 1,
#             'vehicle.trailer': 1,
#             'vehicle.truck': 1,
#             'human.pedestrian.adult': 2,
#             'human.pedestrian.child': 2,
#             'human.pedestrian.construction_worker': 2,
#             'human.pedestrian.personal_mobility': 2,
#             'human.pedestrian.police_officer': 2,
#             'human.pedestrian.stroller': 2,
#             'human.pedestrian.wheelchair': 2,
#             'animal': 0,
#             'movable_object.barrier': 0,
#             'movable_object.debris': 0,
#             'movable_object.pushable_pullable': 0,
#             'movable_object.trafficcone': 0,
#             'static_object.bicycle_rack': 0,
#         }
#
#         # four maps in Nuscenes: 'singapore-onenorth', 'boston-seaport', 'singapore-queenstown', 'singapore-hollandvillage'
#         # self.onenorth = self.load_from_map('singapore-onenorth')
#         # self.queenstown = self.load_from_map('singapore-queenstown')
#         # self.hollandvillage = self.load_from_map('singapore-hollandvillage')
#         # self.seaport = self.load_from_map('boston-seaport')
#
#         self.ending_scenario_num = ending_scenario_num
#         self.end = False
#         if scenario_to_start is not None:
#             self.data_pointer = scenario_to_start
#         else:
#             self.data_pointer = 0
#         self.loaded_playback = None
#         self.gt_relation_path = gt_relation_path
#
#         # dummy variables for logging
#         self.current_file_index = 0
#         # current data index over the whole datas
#
#     def get_next(self, process_intersection=True, relation=True, agent_only=False, only_predict_interest_agents=False,
#                  filter_config={}, calculate_gt_relation=False, load_prediction=True,
#                  detect_gt_relation=False, cut_into_8s=False):
#
#         from nuscenes import NuScenes
#         from nuscenes.map_expansion.map_api import NuScenesMap
#         from pyquaternion import Quaternion
#         from nuscenes.eval.common.utils import quaternion_yaw
#
#         more_data = False
#         if self.end:
#             print("DataLoader Ended")
#             return None, more_data
#
#         ins_sam_pair = []
#         current_scene = None
#
#         if self.data_pointer >= len(self.datas) or (self.ending_scenario_num is not None and self.data_pointer >= self.ending_scenario_num + 1):
#             self.end = True
#             return None, more_data
#
#         while True:
#             if self.data_pointer >= len(self.datas):
#                 break
#             cur_instance_token, cur_sample_token = self.datas[self.data_pointer].split("_")
#             current_ann = self.helper.get_sample_annotation(cur_instance_token, cur_sample_token)
#             assert 'vehicle' in current_ann['category_name'], 'predicting a non_vehicle'
#
#             # find whether it's different scene
#             scene_token = self.get('sample', current_ann['sample_token'])['scene_token']
#             if current_scene == None: current_scene = scene_token
#
#             # if it's a new scene, add the scene_token into dict and instancen_sample_pair
#             if scene_token == current_scene:
#                 ins_sam_pair.append((cur_instance_token, cur_sample_token))
#                 self.data_pointer += 1
#             else:
#                 break
#
#         # load all information in the new scene
#         scene_record = self.get('scene', current_scene)
#         map_name = self.get('log', scene_record['log_token'])['location']
#         self.nuscmap = NuScenesMap(dataroot=f'{self.dataroot}', map_name=map_name)
#
#         all_samples = self.field2token('sample', 'scene_token', current_scene)
#         ego_sample = {}
#         agents_sample = {}
#         sample_tokens = []
#         instance_tokens = []
#         for sample_token in all_samples:
#             sample_record = self.get('sample', sample_token)
#             # store ego information of the scene
#             ego_sample[sample_record['token']] = self.get('ego_pose', sample_record['data']['LIDAR_TOP'])
#
#             # store agents information of the scene
#             ann_tokens = sample_record['anns']
#             agents_instance = {}
#             for token in ann_tokens:
#                 ann_record = self.get('sample_annotation', token)
#                 instance_token = ann_record['instance_token']
#                 if instance_token not in instance_tokens:
#                     instance_tokens.append(instance_token)
#                 agents_instance[instance_token] = ann_record
#             agents_sample[sample_record['token']] = agents_instance
#             sample_tokens.append(sample_record['token'])
#
#         # init all the info tensors
#         len_ins = len(instance_tokens) + 1  # +1 to leave place for ego info
#         len_sam = len(sample_tokens)
#         state_xyzyaw = -1 * np.ones((len_ins, len_sam, 4))
#         state_shape = -1 * np.ones((len_ins, len_sam, 3))
#         # state_bbox_yaw = -1 * np.ones((len_ins, len_sam))
#         # state_velocity_x = -1 * np.ones((len_ins, len_sam))
#         # state_velocity_y = -1 * np.ones((len_ins, len_sam))
#         # gt_future_is_valid = np.zeros((len_ins, len_sam)) > 0 # Bool type
#         object_type = -1 * np.ones((len_ins))
#         tracks_to_predict = -1 * np.ones((len_ins))
#         interactive_tracks_to_predict = -1 * np.ones(len_ins)
#         sample_is_valid = np.zeros(len_ins) > 0 # Bool
#         agent_id = []
#         scenario_id = scene_record['name']
#         is_sdc = np.zeros(len_ins)
#
#         # load ego information: fix dim1 to 0
#         is_sdc[0] = 1
#         agent_id.append(0)
#         object_type[0] = 1  # ego is car
#         tracks_to_predict[0] = 1  # do we predict ego trajectory?
#         interactive_tracks_to_predict[0] = 1  # same question?
#         sample_is_valid[0] = True
#         for dim2, sample_token in enumerate(sample_tokens):
#             ego_record = ego_sample[sample_token]
#             # ego size: length 4.084 width 1.73 height 1.562
#             ego_record['size'] = [1.730, 4.084, 1.562]
#             state_xyzyaw[0, dim2, :3] = ego_record['translation']
#             state_shape[0, dim2, :] = ego_record['size']
#             state_xyzyaw[0, dim2, 3] = quaternion_yaw(Quaternion(ego_record['rotation']))
#             # state_velocity_x[0][dim2], state_velocity_y[0][dim2] = self.get_agent_velocity(ego_record, mode='ego')[:2]
#             # gt_future_is_valid[0][dim2] = True
#
#         for dim1, instance_token in enumerate(instance_tokens):
#             idx = dim1 + 1  # 0 is ego, the env_agents starts from index 1
#             for dim2, sample_token in enumerate(sample_tokens):
#                 agents_instance = agents_sample[sample_token]
#                 if instance_token not in agents_instance.keys(): continue
#                 ann_record = agents_instance[instance_token]
#                 state_xyzyaw[idx][dim2][:3] = ann_record['translation']
#                 state_shape[idx][dim2] = ann_record['size']
#                 state_xyzyaw[idx][dim2][3] = quaternion_yaw(Quaternion(ann_record['rotation']))
#                 # state_velocity_x[dim1][dim2], state_velocity_y[dim1][dim2] = self.get_agent_velocity(ann_record)[:2]
#                 # gt_future_is_valid[dim1][dim2] = True
#             object_type[idx] = self.ClassMapping[ann_record['category_name']]
#             # predict trajectories of certain classes
#             # tracks_to_predict[dim1] = 1 if self.ClassMapping[ann_record['category_name']] == self.predict_class else -1
#             # human and vehicles classes are labeled to have interaction
#             # interactive_tracks_to_predict[dim1] = 1 if self.ClassMapping[ann_record['category_name']] > 0 else -1
#             sample_is_valid[idx] = True
#             agent_id.append(idx)
#
#         decoded_example = {
#             'state/type': object_type,
#             'state/id': agent_id,
#             'scenario/id': scenario_id,
#             'state/objects_of_interest': interactive_tracks_to_predict,
#             'state/is_sdc': is_sdc
#         }
#
#         def downSample(org_state):
#             sample_scale = 5
#             total_frame = org_state.shape[0]
#             new_state_xyzyaw = -1 * np.ones((total_frame * sample_scale, 4))
#             for idx in range(total_frame * sample_scale):
#                 idx_offset = idx % 5
#                 idx_number = int(idx / 5)
#                 current_state = org_state[idx_number, :]
#                 if current_state[0] == -1 or current_state[1] == -1:
#                     new_state_xyzyaw[idx, :] = current_state[:]
#                 elif idx_number + 1 != total_frame:
#                     next_state = org_state[idx_number+1, :]
#                     if next_state[0] == -1 or next_state[1] == -1:
#                         new_state_xyzyaw[idx, :] = current_state[:]
#                     else:
#                         new_state_xyzyaw[idx, 0] = current_state[0] + (next_state - current_state)[0] * idx_offset / 5
#                         new_state_xyzyaw[idx, 1] = current_state[1] + (next_state - current_state)[1] * idx_offset / 5
#                         new_state_xyzyaw[idx, 3] = current_state[3]
#                 else:
#                     new_state_xyzyaw[idx, :] = current_state[:]
#             return new_state_xyzyaw
#
#         agent_dic = {}
#         for idx, each_agent_id in enumerate(agent_id):
#             if idx == 0:
#                 to_pred = 1
#             else:
#                 to_pred = 0
#             agent_dic[each_agent_id] = {
#                 # 'pose': state_xyzyaw[idx],
#                 'pose': downSample(state_xyzyaw[idx]),
#                 'shape': state_shape[idx],
#                 'type': object_type[idx],
#                 'to_predict': np.array(to_pred), 'to_interact': np.array(to_pred), 'is_sdc': idx == 0
#             }
#
#         # add map_info
#         # if map_name == 'singapore-onenorth': decoded_example.update(self.onenorth)
#         # if map_name == 'boston-seaport': decoded_example.update(self.seaport)
#         # if map_name == 'singapore-queenstown': decoded_example.update(self.queenstown)
#         # if map_name == 'singapore-hollandvillage': decoded_example.update(self.hollandvillage)
#
#         skip = False
#         if detect_gt_relation:
#             edges = get_relation_on_crossing(agent_dic=agent_dic, only_prediction_agents=only_predict_interest_agents)
#             form_a_tree = False
#             if not only_predict_interest_agents and form_a_tree:
#                 edges = form_tree_from_edges(edges)
#
#             temp_loading_flag = True
#             if temp_loading_flag:
#                 for edge in edges:
#                     if len(edge) != 4:
#                         # [agent_id_influencer, agent_id_reactor, frame_idx_reactor_passing_cross_point, abs(frame_diff)]
#                         print("invalid edge: ", edge)
#                         skip = True
#                         break
#                     # one type per agent
#                     # relation type: 1-vv, 2-vp, 3-vc, 4-others
#                     agent_types = []
#                     agent_id1, agent_id2, _, _ = edge
#                     for agent_id in agent_dic:
#                         if agent_id in [agent_id1, agent_id2]:
#                             agent_types.append(agent_dic[agent_id]['type'])
#                     if len(agent_types) != 2:
#                         print("WARNING: Skipping an solo interactive agent scene - ", str(agent_types),
#                               str(scenario_id))
#                         skip = True
#         else:
#             edges = []
#
#         if not agent_only:
#             ego_post = state_xyzyaw[0, 0, :]
#             patch = (ego_post[0] - 500, ego_post[1] - 500, ego_post[0] + 500, ego_post[1] + 500)
#
#             decoded_example.update(self.load_from_map(map_name, patch))
#
#             road_dic = {}
#             road_dir_np = decoded_example['roadgraph_samples/dir']
#             road_id_np = decoded_example['roadgraph_samples/id']
#             road_xy_np = decoded_example['roadgraph_samples/xyz']
#             road_type_np = decoded_example['roadgraph_samples/type']
#             road_valid_np = decoded_example['roadgraph_samples/valid']
#
#             for i in range(len(road_dir_np)):
#                 road_id = int(road_id_np[i][0])
#                 if road_id in road_dic.keys():
#                     if int(road_valid_np[i][0]) == 1:
#                         road_dic[road_id]['dir'] = np.vstack(
#                                 (road_dic[road_id]['dir'], np.array(vector2radical(road_dir_np[i]))))
#                         road_dic[road_id]['xyz'] = np.vstack((road_dic[road_id]['xyz'], road_xy_np[i][:3]))
#                 else:
#                     # init a road_dic
#                     if int(road_valid_np[i][0]) == 1:
#                         new_dic = {'dir': np.array(vector2radical(road_dir_np[i])),
#                                    # 'xy': road_xy_np[i][:2].reshape(1, 2),
#                                    'type': road_type_np[i], 'turning': -1,
#                                    'next_lanes': np.array([]), 'previous_lanes': np.array([]),
#                                    'outbound': 0, 'marking': 0,
#                                    'vector_dir': road_dir_np[i], 'xyz': road_xy_np[i][:3].reshape(1, 3), }
#                         road_dic[road_id] = new_dic
#             # no traffic lights loaded
#             traffic_dic = {}
#             # calculate turning directions, next lanes
#             for road_key in road_dic.keys():
#                 lane_dic = road_dic[road_key]
#                 assert road_dic[road_key]["xyz"].shape[1] == 3
#                 dirs = lane_dic["dir"].flatten()
#                 lane_type = int(lane_dic["type"])
#                 lane_turning = is_turning_lane(dirs)
#                 if lane_type in [1, 2]:
#                     next_lanes_np = get_next_laneid(current_lane_id=road_key,
#                                                     road_dic=road_dic)
#                     lane_dic["next_lanes"] = next_lanes_np
#                     lane_dic["turning"] = lane_turning
#                 else:
#                     lane_dic["turning"] = -1
#                 road_dic[road_key] = lane_dic
#
#             for road_key in road_dic.keys():
#                 if int(road_dic[road_key]["type"]) not in [1, 2]:
#                     continue
#                 next_lanes_np = road_dic[road_key]["next_lanes"]
#                 for next_lane_id in next_lanes_np:
#                     road_dic[next_lane_id]["previous_lanes"] = np.insert(road_dic[next_lane_id]["previous_lanes"],
#                                                                          0, road_key)
#         else:
#             road_dic = {}
#             traffic_dic = {}
#
#         if not cut_into_8s:
#             data_to_return = {"road": road_dic, "agent": agent_dic, "traffic_light": traffic_dic, 'raw': decoded_example,
#                               'skip': skip, 'scenario_str': scenario_id, 'scenario': scenario_id, 'category': 1,
#                               'edges': edges}
#             data_to_return[scenario_id] = data_to_return
#             return data_to_return, True
#         else:
#             import copy
#             data_list_to_return = []
#             random_agent_id = list(agent_dic.keys())[0]
#             total_time = agent_dic[random_agent_id]['pose'].shape[0]
#             # if we have 150, then we cut them into [0, 91], [10, 101], ..., [40, 141]
#             total_num = int((total_time - 91) / 10)
#             for i in range(total_num):
#                 new_agent_dic = copy.deepcopy(agent_dic)
#                 if i != 0:
#                     for each_agent_id in new_agent_dic:
#                         new_pose = new_agent_dic[each_agent_id]['pose'][i * 10 - 1:i * 10 + 91, :]
#                         new_agent_dic[each_agent_id]['pose'] = new_pose
#                 data_to_return = {"road": road_dic, "agent": new_agent_dic, "traffic_light": traffic_dic,
#                                   'raw': decoded_example,
#                                   'skip': skip, 'scenario_str': scenario_id, 'scenario': f'{scenario_id}-{i}', 'category': 1,
#                                   'edges': edges}
#                 # data_to_return[f'{scenario_id}-{i}'] = data_to_return
#                 data_list_to_return.append(data_to_return)
#             return data_list_to_return, True
#
#
#     def load_from_map(self, map_name, patch=None):
#         from nuscenes.map_expansion.arcline_path_utils import discretize_lane
#         # self.nuscmap = NuScenesMap(dataroot=f'{self.dataroot}', map_name=map_name)
#
#         xyz = []
#         type = []
#         valid = []
#         id = []
#         global_id = 1
#         id2token = {}
#         map_info = {}
#
#         def get_direction(xyz):
#             xyz = np.array(xyz)
#             vector = xyz[1:] - xyz[:-1]
#             norm = np.linalg.norm(vector, axis=0).flatten()
#             vector = vector / norm
#             return np.concatenate([vector, vector[-1][np.newaxis, :]])
#
#         if patch is not None:
#             lane_dividers = self.nuscmap.get_records_in_patch(patch)['lane_divider']
#             # processing lane_divider:
#             for each_token in lane_dividers:
#                 for ld_node in self.nuscmap.get('lane_divider', each_token)['lane_divider_segments']:
#                     node_record = self.nuscmap.get('node', ld_node['node_token'])
#                     xyz.append([node_record['x'], node_record['y'], 0])  # all z coor fixed to 0
#                     type.append(self.LineMapping[ld_node['segment_type']])
#                     valid.append(1)
#                     id.append(global_id)
#                 id2token[global_id] = {
#                     'type': 'lane_divider',
#                     'token': each_token
#                 }
#                 global_id += 1
#
#             road_dividers = self.nuscmap.get_records_in_patch(patch)['road_divider']
#             # processing road_divider
#             for each_token in road_dividers:
#                 for node_token in self.nuscmap.get('road_divider', each_token)['node_tokens']:
#                     node_record = self.nuscmap.get('node', node_token)
#                     xyz.append([node_record['x'], node_record['y'], 0])  # all z coor fixed to 0
#                     type.append(self.LineMapping['ROAD_DIVIDER'])
#                     valid.append(1)
#                     id.append(global_id)
#                 id2token[global_id] = {
#                     'type': 'road_divider',
#                     'token': each_token
#                 }
#                 global_id += 1
#
#             # lane_connectors = self.nuscmap.get_records_in_patch(patch)['lane_connector']
#             # processing center_lines
#             # for lane in lane_connectors:
#             #     my_lane = self.nuscmap.arcline_path_3.get(lane['token'], [])
#             #     discretized = np.array(discretize_lane(my_lane, resolution_meters=self.resolution_meters))
#             #     for node in discretized:
#             #         xyz.append([node[0], node[1], 0])  # all z coor fixed to 0
#             #         type.append(self.LineMapping['LANE_CONNECTOR'])
#             #         valid.append(1)
#             #         id.append(global_id)
#             #     id2token[global_id] = {
#             #         'type': 'lane_connector',
#             #         'token': lane['token']
#             #     }
#             #     global_id += 1
#
#             lanes = self.nuscmap.get_records_in_patch(patch)['lane']
#             for lane_token in lanes:
#                 my_lane = self.nuscmap.arcline_path_3.get(lane_token, [])
#                 discretized = np.array(discretize_lane(my_lane, resolution_meters=self.resolution_meters))
#                 for node in discretized:
#                     xyz.append([node[0], node[1], 0])  # all z coor fixed to 0
#                     type.append(self.LineMapping['LANE'])
#                     valid.append(1)
#                     id.append(global_id)
#                 id2token[global_id] = {
#                     'type': 'lane',
#                     'token': lane_token
#                 }
#                 global_id += 1
#
#             ped_crossing = self.nuscmap.get_records_in_patch(patch)['ped_crossing']
#             # processing ped_crossing
#             for cross_record in ped_crossing:
#                 for node_token in self.nuscmap.get('ped_crossing', cross_record)['exterior_node_tokens']:
#                     node_record = self.nuscmap.get('node', node_token)
#                     xyz.append([node_record['x'], node_record['y'], 0])  # all z coor fixed to 0
#                     type.append(self.LineMapping['PED_CROSSING'])
#                     valid.append(1)
#                     id.append(global_id)
#                 id2token[global_id] = {
#                     'type': 'ped_crossing',
#                     'token': cross_record
#                 }
#                 global_id += 1
#         else:
#             # processing lane_divider:
#             for ld_record in self.nuscmap.lane_divider:
#                 for ld_node in ld_record['lane_divider_segments']:
#                     node_record = self.nuscmap.get('node', ld_node['node_token'])
#                     xyz.append([node_record['x'], node_record['y'], 0]) # all z coor fixed to 0
#                     type.append(self.LineMapping[ld_node['segment_type']])
#                     valid.append(1)
#                     id.append(global_id)
#                 id2token[global_id] = {
#                     'type': 'lane_divider',
#                     'token': ld_record['token']
#                 }
#                 global_id += 1
#
#             # processing road_divider
#             for rd_record in self.nuscmap.road_divider:
#                 for node_token in rd_record['node_tokens']:
#                     node_record = self.nuscmap.get('node', node_token)
#                     xyz.append([node_record['x'], node_record['y'], 0]) # all z coor fixed to 0
#                     type.append(self.LineMapping['ROAD_DIVIDER'])
#                     valid.append(1)
#                     id.append(global_id)
#                 id2token[global_id] = {
#                     'type': 'road_divider',
#                     'token': rd_record['token']
#                 }
#                 global_id += 1
#
#             # processing center_lines
#             for lane in self.nuscmap.lane_connector:
#                 my_lane = self.nuscmap.arcline_path_3.get(lane['token'], [])
#                 discretized = np.array(discretize_lane(my_lane, resolution_meters = self.resolution_meters))
#                 for node in discretized:
#                     xyz.append([node[0], node[1], 0]) # all z coor fixed to 0
#                     type.append(self.LineMapping['LANE_CONNECTOR'])
#                     valid.append(1)
#                     id.append(global_id)
#                 id2token[global_id] = {
#                     'type': 'lane_connector',
#                     'token': lane['token']
#                 }
#                 global_id += 1
#             for lane in self.nuscmap.lane:
#                 my_lane = self.nuscmap.arcline_path_3.get(lane['token'], [])
#                 discretized = np.array(discretize_lane(my_lane, resolution_meters = self.resolution_meters))
#                 for node in discretized:
#                     xyz.append([node[0], node[1], 0]) # all z coor fixed to 0
#                     type.append(self.LineMapping['LANE'])
#                     valid.append(1)
#                     id.append(global_id)
#                 id2token[global_id] = {
#                     'type': 'lane',
#                     'token': lane['token']
#                 }
#                 global_id += 1
#
#             # processing ped_crossing
#             for cross_record in self.nuscmap.ped_crossing:
#                 for node_token in cross_record['exterior_node_tokens']:
#                     node_record = self.nuscmap.get('node', node_token)
#                     xyz.append([node_record['x'], node_record['y'], 0]) # all z coor fixed to 0
#                     type.append(self.LineMapping['PED_CROSSING'])
#                     valid.append(1)
#                     id.append(global_id)
#                 id2token[global_id] = {
#                     'type': 'ped_crossing',
#                     'token': cross_record['token']
#                 }
#                 global_id += 1
#
#         map_info['roadgraph_samples/xyz'] = np.array(xyz, dtype=np.float32)  # 20000, 3
#         map_info['roadgraph_samples/dir'] = np.array(get_direction(np.array(xyz)), dtype=np.float32)  # 20000, 3
#         map_info['roadgraph_samples/type'] = np.array(type, dtype=np.float32)[:, np.newaxis]  # 20000, 1
#         map_info['roadgraph_samples/valid'] = np.array(valid, dtype=np.float32)[:, np.newaxis]  # 20000, 1
#         map_info['roadgraph_samples/id'] = np.array(id, dtype=np.float32)[:, np.newaxis]  # 20000, 1
#
#         return map_info

############# uncomment below to load submit proto prediction data ###############
#
# import os
# import multiprocessing
# import joblib
# from joblib import Parallel, delayed
# import tensorflow as tf
#
# from waymo_open_dataset.protos import motion_submission_pb2 as sub_proto
#
#
# def get_FDE(gt_path, pred_path):
#     loss = 0
#     assert gt_path.shape == (91, 2), str(gt_path.shape)
#     assert pred_path.shape == (16, 2), str(pred_path.shape)
#     for i in range(pred_path.shape[0]):
#         pred_pt = pred_path[i, :]
#         gt_pt = gt_path[11 + i * 5, :]
#         if float(gt_pt[0]) != -1.0:
#             loss += euclidean_distance(pred_pt, gt_pt)
#     return loss
#
#
# def get_predictions_for_agent(original_scenario_index, target_agent_id, target_scenario_id, gt_dataDic):
#     if target_agent_id not in gt_dataDic['agent']:
#         print("agent not found in groud truth: ", target_scenario_id, target_agent_id)
#         return None
#     last_yaw = float(gt_dataDic['agent'][target_agent_id]['pose'][10, 3])
#
#     prediction_file_path = 'interactive_predict_all.submission'
#     # val_source_path = 'val_interactive_source.npy'
#     # source_np = np.load(val_source_path)
#     # print("loaded prep source to predict shape: ", source_np.shape)
#     with open(prediction_file_path, 'rb') as predictions_bytes:
#         prediction_data = predictions_bytes.read()
#         # Further file processing goes here
#         submission = sub_proto.MotionChallengeSubmission()
#         submission.ParseFromString(prediction_data)
#         scenario_predictions_list = submission.scenario_predictions
#         total_predictions = len(scenario_predictions_list)
#         # 14426 scenarios in validation_interactive
#         for scene_index, ChallengeScenarioPredictions in enumerate(scenario_predictions_list):
#             if scene_index != original_scenario_index:
#                 continue
#
#             prediction_trajectories = ChallengeScenarioPredictions.single_predictions.predictions
#             for prediction in prediction_trajectories:
#                 # each agent
#                 agent_id = prediction.object_id
#                 if int(agent_id) != target_agent_id:
#                     continue
#                 pred_trajs = prediction.trajectories
#                 # shape: prediction_num, 91, 2
#                 trajs_list = []
#                 # shape: prediction_num, 91
#                 yaws_list = []
#                 # shape: prediction_num
#                 scores_list = []
#                 # shape: prediction_num
#                 # FDE_list = []
#                 # prediction_num = len(pred_trajs)
#                 # prediction num = 6
#                 for ScoredTrajectory in pred_trajs:
#                     # each prediction for an agent
#                     traj = ScoredTrajectory.trajectory
#                     score = ScoredTrajectory.confidence
#                     scores_list.append(math.exp(score))
#                     traj_list = []
#                     for idx, x in enumerate(traj.center_x):
#                         traj_list.append([x, traj.center_y[idx]])
#                     yaw_list = []
#                     for idx, pose in enumerate(traj_list):
#                         if idx == 0:
#                             continue
#                         d_x = pose[1] - traj_list[idx - 1][1]
#                         d_y = pose[0] - traj_list[idx - 1][0]
#                         if abs(d_x) < 0.1 and abs(d_y) < 0.1:
#                             if len(yaw_list) < 1:
#                                 yaw_list.append(last_yaw)
#                             else:
#                                 yaw_list.append(yaw_list[-1])
#                         else:
#                             yaw_list.append(normalize_angle((math.atan2(d_x, d_y))))
#                     yaw_list.insert(0, yaw_list[0])
#                     # trajs_list.append(traj_list)
#                     # yaws_list.append(yaw_list)
#
#                     #     FDE_list.append(get_FDE(gt_path=gt_path, pred_path=traj_list))
#                     # minimal_path_index = FDE_list.index(min(FDE_list))
#                     # return traj_list[minimal_path_index], yaws_list[minimal_path_index]
#                     # populate data
#
#                     two_herz_list = traj_list.copy()
#                     two_herz_yaw = yaw_list.copy()
#                     for i in range(16):
#                         traj = two_herz_list[i]
#                         yaw = two_herz_yaw[i]
#                         for j in range(4):
#                             traj_list.insert(i * 5, traj)
#                             yaw_list.insert(i * 5, yaw)
#                     assert len(traj_list) == 80, str(len(traj_list))
#                     assert len(yaw_list) == 80, str(len(yaw_list))
#                     trajs_list.append(traj_list)
#                     yaws_list.append(yaw_list)
#                 return np.array(trajs_list), np.array(yaws_list), np.array(scores_list)
#
#         print("prediction not found: ", target_scenario_id, target_agent_id)
#
# class WaymoDL_LoadTargetScenes(WaymoDL):
#     def __init__(self, filepath):
#         self.tf_example_dir = filepath
#         self.prefix = 'validation_interactive_tfexample.tfrecord-00'
#         self.surfix = '-of-00150'
#         worst = True
#         if not worst:
#             # best scenarios
#             self.targets = [(b'11', 'b0837c10b5f9c51', 2262, 4086), (b'45', '2bd62502524c2d1e', 493, 10464),
#                             (b'17', '4e7882596771cde6', 2515, 13080), (b'41', '50ddee537763579a', 192, 14056),
#                             (b'25', 'c6553fb818cd5212', 5316, 13235), (b'39', '5ea10ed01efd404d', 4, 8512),
#                             (b'2', 'be0e083e6f3b7a0c', 91, 2221), (b'20', '12041cd6a46c7fcb', 4856, 11177),
#                             (b'33', 'abecf151ab0d4493', 1572, 13746), (b'11', '448be4c3a722913a', 145, 4121),
#                             (b'5', '7318349e22bab361', 484, 9023), (b'16', 'e5dc645543f6d7da', 2349, 792),
#                             (b'44', '19f5d501faec4d95', 2481, 12318), (b'44', 'd85c4288ef13add2', 490, 12305),
#                             (b'2', '31a2a41513404eb5', 812, 2242), (b'31', '31cfe1d1c997cabd', 572, 8192),
#                             (b'39', 'b2428b54e59ada01', 199, 8406), (b'2', 'e3a74151a4be1207', 528, 2191),
#                             (b'45', '537337e436d2e31e', 1794, 10372), (b'36', 'aa65d1fb18deb756', 30, 11963),
#                             (b'8', '39da8c72d4c297b5', 9, 335), (b'16', 'd5e30b0f695cc5ce', 2717, 697),
#                             (b'45', 'aefcf199f27a5956', 417, 10360), (b'35', '98f957b91e88bb4f', 2828, 4838),
#                             (b'5', '8c99545901adbd43', 1589, 9036), (b'34', 'f0e319bdcc3cae90', 925, 3249),
#                             (b'16', '6b1f450f36d87656', 911, 721), (b'45', '7d2c18085b049e8a', 1244, 10534),
#                             (b'37', '2cea43d7acdd9bac', 151, 10211), (b'2', 'e5c0bb9dee88977a', 249, 2261),
#                             (b'42', '8a07f5c4c88a188b', 119, 3650), (b'35', 'e04750587b315224', 2188, 5060),
#                             (b'29', '82bbb8a9504b8e06', 1758, 10004), (b'33', '1401508fe4485189', 169, 13552),
#                             (b'35', '56a7d2f97dc61dba', 1494, 4969), (b'24', 'd52d2247a062e476', 20, 954),
#                             (b'41', '3e9d92b492799b3f', 1001, 14012), (b'11', 'cdc4fe7569337719', 43, 4222),
#                             (b'31', '217c4562ac54eb6e', 385, 8065), (b'30', '4479956b1cca81a8', 1626, 6474),
#                             (b'24', '505aa532f4500678', 733, 1042), (b'8', '98c06371763912ce', 642, 553),
#                             (b'9', 'fd50d252845859a2', 425, 12718), (b'3', '2aa1d3e39afa64dc', 211, 3823),
#                             (b'3', '88eeeb833edacc51', 2579, 3871), (b'20', '7db2d91dbff3acdc', 5046, 11293),
#                             (b'21', 'b70c6415df1e0cd8', 584, 9624), (b'39', '36bcdeb2868d53c6', 1748, 8477),
#                             (b'42', 'b1927f219341db5e', 527, 3686), (b'23', 'e33464daa50cdd4b', 681, 7887)]
#         else:
#             # worst scenarios
#             self.targets = [(b'43', '3805690e1a1f2ee2', 2945, 5182), (b'20', 'b40b307f4b22510d', 15, 11407),
#                             (b'26', '9d1897526152be4f', 578, 2956), (b'5', 'c34d128477d1411b', 380, 9034),
#                             (b'33', '2f413ddf2a81d0a8', 376, 13742), (b'22', '3dad0b2318d36046', 2911, 5977),
#                             (b'40', '2d707562abef08e', 604, 1642), (b'31', '34c66592642d3daa', 825, 8159),
#                             (b'9', '4aa53a497164481b', 0, 12655), (b'28', '24e195e07a7105d', 29, 11695),
#                             (b'39', 'aa3dda82ff1346d6', 491, 8344), (b'9', '1f3a1fcd1c6e3ae1', 1508, 12753),
#                             (b'2', 'c05b473fc1b98dc5', 769, 2186), (b'25', '849c8a8868380ea3', 43, 13363),
#                             (b'15', 'f30d673689c3974c', 2614, 7649), (b'19', '30da4d7f4667c26f', 373, 4535),
#                             (b'19', '757710384872e114', 2385, 4528), (b'41', '7fa58c7d6c466ea1', 1253, 13981),
#                             (b'40', '4e0aff5487abdaa2', 1049, 1716), (b'44', '2c129fe4f9c0b390', 464, 12106),
#                             (b'28', 'cad0b194202d4260', 830, 11683), (b'16', '6f9a99944247ce41', 1550, 688),
#                             (b'42', '5ce4a1dacbb92239', 590, 3608), (b'0', '2fe3347ce5691e17', 1774, 172),
#                             (b'9', '80b54b6b552c0b49', 164, 12882), (b'27', 'b3bcd719d736104d', 1097, 4758),
#                             (b'1', '86dc1b8823a1f88d', 239, 12535), (b'30', '33600132510c8c13', 48, 6403),
#                             (b'1', '710e61166b69be52', 1097, 12510), (b'4', '3ddb16b5c0ff6159', 351, 10648),
#                             (b'44', '29dfb0520ef7c5a7', 261, 12100), (b'38', 'cc0532c52e9f1bd9', 578, 6790),
#                             (b'8', '201ebe30be2f5fb1', 1349, 556), (b'37', 'dbb04aea54936408', 1354, 10161),
#                             (b'40', 'd2a644fb90464c6b', 1737, 1649), (b'24', '82019aa9878b49ae', 1097, 921),
#                             (b'17', 'edc45b8e390bf0b', 1607, 13024), (b'35', '76b712f834ba66a6', 16, 5106),
#                             (b'44', '19f5d501faec4d95', 1480, 12318), (b'8', '894c85f074942522', 2092, 585),
#                             (b'3', '7d87343246cd6573', 2110, 3828), (b'27', '2014fe2139d98ca8', 284, 4586),
#                             (b'1', '297f7f6c02140ecb', 380, 12527), (b'24', 'b081f846fd3b6324', 1161, 1085),
#                             (b'27', '682489bbb2efce1a', 499, 4705), (b'23', 'a87cb50c40761580', 1371, 7842),
#                             (b'36', '9287cb951de29412', 1737, 11790), (b'18', '346847af01697762', 3303, 2602),
#                             (b'40', '861dfa1d71c39bd6', 839, 1481), (b'39', '55a476456c08cc22', 665, 8430)]
#         self.current_scene_index = 0
#         self.current_file_total_scenario = len(self.targets)
#         self.end = False
#         self.current_scene_pred = {}
#
#         # dummy vaiables for printing logs
#         self.current_file_index = FILE_TO_START - 1
#         self.current_scenario_index = 0
#
#     def get_next(self, process_intersection=True):
#         if self.end:
#             return None
#         # dataset = tf.data.TFRecordDataset(
#         #     "../Waymo/motion_dataset/tf_example/training/training_tfexample.tfrecord-00000-of-01000",
#         #     compression_type='')
#         # reinitialize prediction dictionary
#         self.current_scene_pred = {}
#
#         infos = self.targets[self.current_scene_index]
#         current_file_index = int(infos[0].decode())
#         current_scenario_id = infos[1]
#         current_agent_ids = [int(infos[2])]
#         original_scenario_index = int(infos[3])
#         file_path = self.tf_example_dir + '/' + self.prefix + f'{current_file_index:03d}' + self.surfix
#         dataset = tf.data.TFRecordDataset(file_path, compression_type='')
#         for idx, data in enumerate(dataset.as_numpy_iterator()):
#             scenario_features = {
#                 'scenario/id':
#                     tf.io.FixedLenFeature([], tf.string, default_value=None)
#             }
#             features_description_scenario = {}
#             features_description_scenario.update(scenario_features)
#             parsed = tf.io.parse_single_example(data, features_description_scenario)
#             scenario_id = parsed['scenario/id'].numpy()
#
#             if str(scenario_id.decode()) != current_scenario_id:
#                 continue
#             data_to_return = self.get_datadic_fromTFRecord(data, process_intersection=process_intersection)
#             data_to_return['id'] = scenario_id
#
#             offset = 1
#             while self.current_scene_index + offset < len(self.targets) and \
#                             self.targets[self.current_scene_index + offset][1] == current_scenario_id:
#                 offset += 1
#                 current_agent_ids.append(int(self.targets[self.current_scene_index + offset][2]))
#             # for each agent in current_agent, populate from 2Hz to 10Hz, and put them in the prediction dic
#             for agent_id in current_agent_ids:
#                 # load prediction for
#                 print("loading prediction: ", original_scenario_index, agent_id, current_scenario_id)
#                 prediction_rst = get_predictions_for_agent(original_scenario_index=original_scenario_index,
#                                                            target_agent_id=agent_id,
#                                                            target_scenario_id=current_scenario_id,
#                                                            gt_dataDic=data_to_return)
#                 if prediction_rst is not None:
#                     pred_trajectories, pred_yaws, pred_scores = prediction_rst
#                 else:
#                     continue
#                 info_dic = {'pred_trajectory': pred_trajectories,
#                             'pred_yaw': pred_yaws,
#                             'pred_scores': pred_scores}
#                 self.current_scene_pred[agent_id] = info_dic
#
#             self.current_scene_index += offset
#             if self.current_scene_index >= self.current_file_total_scenario:
#                 self.end = True
#             return data_to_return
#         print("ERROR: found no scenario ", infos)
#         return None

