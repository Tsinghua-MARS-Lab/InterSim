import tensorflow as tf
from interactive_sim.envs.util import *
import math
import os
import numpy as np
from waymo_open_dataset.protos import scenario_pb2 as open_dataset

# set to 0 to iterate all files
FILE_TO_START = 1
# interesting scenes:
# 185 see 4 ways stop line, 201 roundabout, 223 & 230 for a huge intersection
SCENE_TO_START = 305
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
    diff_dir = new_end_dir - new_start_dir

    if abs(new_start_dir - new_end_dir) > (math.pi / 180 * 30):
        return True
    return False


def get_next_laneid(current_lane_id, road_dic):
    lane_id_list = []
    current_lane_pts = road_dic[current_lane_id]["xy"]
    lane_type = road_dic[current_lane_id]["type"]
    if len(current_lane_pts.shape) > 1 and current_lane_pts.shape[0] > 2 and lane_type in [1, 2]:
        ending_pt = current_lane_pts[-1]
        for road_seg_id in road_dic.keys():
            if road_seg_id == current_lane_id:
                continue
            road_seg = road_dic[road_seg_id]
            road_pts = road_seg["xy"]
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
                ego_way["direction"] = road_dic[outbound_lanes[0]]["dir"][-2]
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
                    intersection_dic["way"][index]["direction"] = road_dic[outbound_lanes[0]]["dir"][-2]

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
                            tl_states_unknown_np = np.array(tl_states_unknown).reshape(total_frames)
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
    xy_np = road_dic[one_inbound_lane_id]["xy"]
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
        target_xy_np = road_dic[road_seg_key]["xy"]
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


def handvdistance(pt1, pt2, direction):
    new_pt2_x, new_pt2_y = rotate(pt1, pt2, -direction)
    return pt1[0] - new_pt2_x, pt1[1] - new_pt2_y


class WaymoDL_Proto:
    def __init__(self, filepath):
        scenario_dir = filepath
        self.file_names = [os.path.join(scenario_dir, f) for f in os.listdir(scenario_dir) if
                           os.path.isfile(os.path.join(scenario_dir, f))]
        self.current_file_index = FILE_TO_START - 1
        self.current_file_total_scenario = 0
        self.load_new_file()
        self.current_scenario_index = SCENE_TO_START
        self.end = False

    def load_new_file(self):
        self.current_file_index += 1
        if self.current_file_index < len(self.file_names):
            # self.current_file_index += 1
            self.current_file_total_scenario = 0
            dataset = tf.data.TFRecordDataset(self.file_names[self.current_file_index], compression_type='')
            for _ in dataset:
                self.current_file_total_scenario += 1
            self.current_scenario_index = 0
            print("Loading file from: ", self.file_names[self.current_file_index], " with ",
                  self.current_file_total_scenario, " scenarios ")
        else:
            self.end = True

    def get_next(self, process_intersection=True):
        if self.end:
            return None
        # dataset = tf.data.TFRecordDataset(
        #     "../Waymo/motion_dataset/tf_example/training/training_tfexample.tfrecord-00000-of-01000",
        #     compression_type='')
        dataset = tf.data.TFRecordDataset(self.file_names[self.current_file_index], compression_type='')
        for idx, data in enumerate(dataset):
            if idx != self.current_scenario_index:
                continue
            scenario = open_dataset.Scenario()
            scenario.ParseFromString(bytearray(data.numpy()))
            scenario_id = scenario.scenario_id

            agent_dic = {}
            objects_of_interest = scenario.objects_of_interest
            agents_to_predict = []
            for track_info in scenario.tracks_to_predict:
                agents_to_predict.append(track_info.track_index)
            sdc_track_index = scenario.sdc_track_index

            # print("test1: ", objects_of_interest)
            # print("test2: ", agents_to_predict)
            # print("test3: ", sdc_track_index)
            """
            dirty data:
            agents_to_predict might be empty or just [0]
            """

            for agent_obj in scenario.tracks:
                agent_type = agent_obj.object_type
                agent_id = agent_obj.id
                # if agent_type_str == "TYPE_VECHICLE":
                #     agent_type = 1
                # elif agent_type_str == "TYPE_PEDESTRIAN":
                #     agent_type = 2
                # elif agent_type_str == "TYPE_CYCLIST":
                #     agent_type = 3
                # elif agent_type_str == "TYPE_OTHER":
                #     agent_type = 4
                # elif agent_type_str == "TYPE_UNSET":
                #     agent_type = 0
                # else:
                #     agent_type = -1
                #     print("ERROR: Unknown agent type: ", agent_type)

                agent_of_interest = 1 if agent_id in objects_of_interest else 0
                agent_to_predict = 1 if agent_id in agents_to_predict else 0
                sdc = 1 if int(agent_id) == int(sdc_track_index) else 0
                # loop pose
                poses = []
                shape = [-1, -1, -1]
                speed = []
                for states in agent_obj.states:
                    if states.valid:
                        poses.append([states.center_x,
                                      states.center_y,
                                      states.center_z,
                                      states.heading])
                        shape = [[states.width, states.length, states.height]]
                        v = math.sqrt(states.velocity_x * states.velocity_x + states.velocity_y * states.velocity_y)
                        speed.append([v, 0])
                    else:
                        poses.append([-1, -1, -1, -1])
                        speed.append([-1, 0])
                new_dic = {'pose': np.array(poses), 'shape': np.array(shape),
                           'speed': np.array(speed), 'type': np.array(agent_type),
                           'to_predict': np.array(agent_to_predict),
                           'objects_of_interest': np.array(agent_of_interest),
                           'is_sdc': sdc}
                agent_dic[agent_id] = new_dic

            road_dic = {}
            for road_obj in scenario.map_features:
                road_id = road_obj.id
                road_pts = []
                road_speed_limit = -1
                if road_obj.HasField('lane'):
                    road_type = road_obj.lane.type
                    road_speed_limit = road_obj.lane.speed_limit_mph
                    # road_type_dictionary = {
                    #     'TYPE_UNDEFINED': 0,
                    #     'TYPE_FREEWAY': 1,
                    #     'TYPE_SURFACE_STREET': 2,
                    #     'TYPE_BIKE_LANE': 3,
                    # }
                    # print("test: ", road_type_str)
                    # road_type = road_type_dictionary[road_type_str]
                    for pts in road_obj.lane.polyline:
                        road_pts.append([pts.x, pts.y])
                elif road_obj.HasField('road_edge'):
                    # UNKNOWN ERROR: type can only get 1 or 2, probably because of the type conflicting system keywords
                    road_type = road_obj.road_edge.type
                    # road_type_dictionary = {
                    #     'TYPE_UNKNOWN': 0,
                    #     'TYPE_ROAD_EDGE_BOUNDARY': 15,
                    #     'TYPE_ROAD_EDGE_MEDIAN': 16
                    # }
                    # road_type = road_type_dictionary[road_type_str]
                    for pts in road_obj.road_edge.polyline:
                        road_pts.append([pts.x, pts.y])
                elif road_obj.HasField('road_line'):
                    road_type = road_obj.road_line.type
                    # road_type_dictionary = {
                    #     'TYPE_UNKNOWN': 0,
                    #     'TYPE_BROKEN_SINGLE_WHITE': 6,
                    #     'TYPE_SOLID_SINGLE_WHITE': 7,
                    #     'TYPE_SOLID_DOUBLE_WHITE': 8,
                    #     'TYPE_BROKEN_SINGLE_YELLOW': 9,
                    #     'TYPE_BROKEN_DOUBLE_YELLOW': 10,
                    #     'TYPE_SOLID_SINGLE_YELLOW': 11,
                    #     'TYPE_SOLID_DOUBLE_YELLOW': 12,
                    #     'TYPE_PASSING_DOUBLE_YELLOW': 13,
                    # }
                    # road_type = road_type_dictionary[road_type_str]
                    for pts in road_obj.road_line.polyline:
                        road_pts.append([pts.x, pts.y])
                elif road_obj.HasField('stop_sign'):
                    stop_sign_lanes = road_obj.stop_sign.lane
                    road_type = 17
                    road_pts.append([road_obj.stop_sign.position.x,
                                     road_obj.stop_sign.position.y])
                elif road_obj.HasField('crosswalk'):
                    road_type = 18
                    for pts in road_obj.crosswalk.polygon:
                        road_pts.append([pts.x, pts.y])
                elif road_obj.HasField('speed_bump'):
                    road_type = 19
                    for pts in road_obj.crosswalk.polygon:
                        road_pts.append([pts.x, pts.y])
                else:
                    # print("ERROR: Unknown road type, ", road_obj)
                    # there are a lot of empty road_obj only with an id
                    continue

                if road_type not in [17, 18, 19]:
                    if len(road_pts) < 3:
                        direction = np.array([-1])
                    else:
                        """
                          dirty data example:
                          lane {
                          speed_limit_mph: 15.0
                          type: TYPE_SURFACE_STREET
                          polyline {
                            x: 8191.389785565264
                            y: 9040.54171031861
                            z: -13.649861083984348
                          }
                        }
                        """
                        direction = np.ones(len(road_pts))
                        for i in range(len(road_pts)):
                            if i == 0:
                                continue

                            delta_x = road_pts[i][0] - road_pts[i-1][0]
                            delta_y = road_pts[i][1] - road_pts[i-1][1]
                            direction_framei = vector2radical((delta_x, delta_y))
                            direction[i-1] = np.array(direction_framei)
                        direction[-1] = direction[-2]
                else:
                    direction = np.array([-1])
                road_pts_np = np.array(road_pts)
                if len(road_pts_np.shape) != 2:
                    continue
                total_frames = road_pts_np.shape[0]
                new_dic = {'dir': direction,
                           'xy': np.array(road_pts),
                           'type': road_type, 'turning': -1,
                           'next_lanes': np.array([]), 'previous_lanes': np.array([]),
                           'outbound': 0, 'marking': 0}
                road_dic[road_id] = new_dic


            # NOTE: traffic needs to be processed after road
            traffic_dic = {}
            total_frames = len(scenario.dynamic_map_states)
            for frame_id, dm_obj in enumerate(scenario.dynamic_map_states):
                if True: # dm_obj.HasField('lane_states'):
                    for lane_obj in dm_obj.lane_states:
                        lane_id = lane_obj.lane
                        state = lane_obj.state
                        stop_pt = lane_obj.stop_point
                        # if state_str == 'LANE_STATE_UNKNOWN':
                        #     state = 0
                        # elif state_str == 'LANE_STATE_ARROW_STOP':
                        #     state = 1
                        # elif state_str == 'LANE_STATE_ARROW_CAUTION':
                        #     state = 2
                        # elif state_str == 'LANE_STATE_ARROW_GO':
                        #     state = 3
                        # elif state_str == 'LANE_STATE_STOP':
                        #     state = 4
                        # elif state_str == 'LANE_STATE_CAUTION':
                        #     state = 5
                        # elif state_str == 'LANE_STATE_GO':
                        #     state = 6
                        # elif state_str == 'LANE_STATE_FLASHING_STOP':
                        #     state = 7
                        # elif state_str == 'LANE_STATE_FLASHING_CAUTION':
                        #     state = 8
                        # else:
                        #     state = -1

                        if not lane_id in traffic_dic.keys():
                            states = np.ones(total_frames) * -1
                            valids = np.zeros(total_frames)
                            traffic_dic[lane_id] = {'valid': valids,
                                                    'state': states}

                        states_dic = traffic_dic[lane_id]
                        states = states_dic['state']
                        valids = states_dic['valid']

                        states[frame_id] = state
                        valids[frame_id] = 1
                        traffic_dic[lane_id] = {'valid': valids,
                                                'state': states}

                        if lane_id in road_dic.keys():
                            # mark 1 on outbound lanes
                            road_dic[lane_id]["outbound"] = 1
                        else:
                            if lane_id != -1:
                                print("WARNING: no lane found for traffic light while parsing", lane_id)

            # calculate turning directions, next lanes
            for road_key in road_dic.keys():
                lane_dic = road_dic[road_key]
                assert road_dic[road_key]["xy"].shape[1] == 2
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
                                                                         0, road_key)

            data_to_return = {
                "scenario": scenario_id,
                "road": road_dic,
                "agent": agent_dic,
                "traffic_light": traffic_dic
            }

            category = classify_scenario(data_to_return)
            data_to_return["category"] = category

            if process_intersection:
                intersection_dic = get_intersection(data_to_return)
                data_to_return["intersection"] = intersection_dic

            if self.current_scenario_index + 1 < self.current_file_total_scenario:
                self.current_scenario_index += 1
            else:
                self.load_new_file()

            return data_to_return
        print("ERROR: found no scenario ", self.current_scenario_index)
        return None





