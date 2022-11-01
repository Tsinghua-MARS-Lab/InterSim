import numpy as np
from interactive_sim.envs.util import *
import time


def get_relation_for_two_trajectories(traj1, traj2, threshold=0.1):
    '''
    You need to be sure traj1 and traj2 start at the same time
    :param traj1: [x, 2]
    :param traj2: [x, 2]
    :param threshold: collision detection threshold
    :return: 2=agent2 first, 1=agent1 first, 0=no relation
    '''
    # detect relationship based on ground truth data
    for idx1, pt1 in enumerate(traj1):
        for idx2, pt2 in enumerate(traj2):
            diff = pt1 - pt2
            dist = np.sqrt(diff[0] ** 2 + diff[1] ** 2)
            if dist < threshold:
                if idx1 > idx2:
                    return 2
                else:
                    return 1
    return 0


def get_agent_from_dic(agent_in_dic, frame_idx1, agent_id):
    x, y, _, yaw = agent_in_dic["pose"][frame_idx1]
    yaw = normalize_angle(yaw + math.pi / 2)
    width, length, _ = agent_in_dic["shape"][0]
    if 'soeed' in agent_in_dic:
        v, v_yaw = agent_in_dic["speed"][frame_idx1]
    else:
        v, v_yaw = 0, 0
    agent = Agent(x=x,
                  y=y, yaw=-yaw,
                  vx=v, length=length, width=width,
                  agent_id=agent_id)
    return agent


def get_relation_on_crossing(agent_dic, only_prediction_agents=False, to_predict=True, total_frame_number=91):
    # edge: [influencer, reactor]
    edges = []
    start_from_frame = 5
    tic = time.perf_counter()
    for agent_id1 in agent_dic:
        if only_prediction_agents and not int(agent_dic[agent_id1]['to_interact']):
            continue
        if to_predict:
            current_pair_has_to_predict_flag = False
        else:
            # always be true to skip none
            current_pair_has_to_predict_flag = True
        if int(agent_dic[agent_id1]['to_predict']):
            current_pair_has_to_predict_flag = True
        cross = False
        exist_edge_index = None
        agent_in_dic1 = agent_dic[agent_id1]
        if check_static_agent(agent_in_dic1):
            continue
        for agent_id2 in agent_dic:
            last_detection_frame_diff = 99999
            if agent_id1 == agent_id2:
                continue
            if only_prediction_agents and not int(agent_dic[agent_id2]['to_interact']):
                continue
            if int(agent_dic[agent_id2]['to_predict']):
                current_pair_has_to_predict_flag = True
            if not current_pair_has_to_predict_flag:
                # skip pairs with no agent to predict at all
                continue
            if [agent_id2, agent_id1] in edges:
                exist_edge_index = edges.index([agent_id2, agent_id1])
            elif [agent_id1, agent_id2] in edges:
                exist_edge_index = edges.index([agent_id1, agent_id2])

            edge_to_add = []
            agent_in_dic2 = agent_dic[agent_id2]
            if check_static_agent(agent_in_dic2):
                continue
            for frame_idx1 in range(total_frame_number):
                if frame_idx1 < start_from_frame:
                    continue
                if cross:
                    break
                agent1 = get_agent_from_dic(agent_in_dic1, frame_idx1, agent_id1)
                if agent1.x == -1 or agent1.y == -1:
                    continue
                for frame_idx2 in range(total_frame_number):
                    if frame_idx2 < start_from_frame:
                        continue
                    if cross:
                        break
                    agent2 = get_agent_from_dic(agent_in_dic2, frame_idx2, agent_id2)
                    if agent2.x == -1 or agent2.y == -1:
                        continue
                    cross1 = check_collision_for_two_agents_rotate_and_dist_check(checking_agent=agent1,
                                                                                  target_agent=agent2)
                    cross2 = check_collision_for_two_agents_rotate_and_dist_check(checking_agent=agent2,
                                                                                  target_agent=agent1)
                    cross = cross1 | cross2
                    if cross:
                        # print("cross detected: ", agent_id1, agent_id2, frame_idx1, frame_idx2)
                        if abs(frame_idx1 - frame_idx2) < last_detection_frame_diff:
                            last_detection_frame_diff = abs(frame_idx1 - frame_idx2)
                            frame_diff = frame_idx1 - frame_idx2
                            if frame_diff > 0:
                                edge_to_add = [agent_id2, agent_id1, frame_idx1, abs(frame_diff)]
                            elif frame_diff < 0:
                                edge_to_add = [agent_id1, agent_id2, frame_idx2, abs(frame_diff)]
                            else:
                                print("collide at same frame", agent_id1, agent_id2)
                                cross = False  # most likely a false detection
            if len(edge_to_add) > 0:
                if exist_edge_index is not None:
                    edges[exist_edge_index] = edge_to_add
                else:
                    edges.append(edge_to_add)
    toc = time.perf_counter()
    agent_num = len(list(agent_dic.keys()))
    print(f"edge detection time per scenario: {toc - tic:04f} seconds with {agent_num} agents and {(toc - tic)/agent_num :04f} seconds per agent")
    return edges


def check_static_agent(agent_dic):
    poses_np = agent_dic['pose']
    valid_poses = poses_np[poses_np[:, 0] != -1]
    if abs(valid_poses[0, 0] - valid_poses[-1, 0]) < 0.1 and abs(valid_poses[0, 1] - valid_poses[-1, 1]) < 0.1:
        return True
    else:
        return False


def form_tree_from_edges(edges):
    edges_np = np.array(edges)
    if len(edges_np.shape) == 1:
        edges_np = np.expand_dims(edges_np, axis=0)
        return edges_np
    assert len(edges_np.shape) == 2 and edges_np.shape[1] == 3, str(edges_np.shape)
    np_to_return = np.empty((0, 3))
    children = np.unique(edges_np[:, 1])
    for child in children:
        selected_edges = edges_np[edges_np[:, 1] == child]
        if selected_edges.shape[0] > 1:
            max_idx = np.argmin(selected_edges[:, 2])
            np_to_return = np.vstack((np_to_return, edges_np[max_idx]))
        else:
            np_to_return = np.vstack((np_to_return, selected_edges))
    # return edges_np.tolist()
    return np_to_return.tolist()



class Agent:
    def __init__(self,
                 # init location, angle, velocity
                 x=0.0, y=0.0, yaw=0.0, vx=0.01, vy=0, length=4.726, width=1.842, agent_id=None):
        self.x = x  # px
        self.y = y
        self.yaw = yaw
        self.vx = vx  # px/frame
        self.vy = vy
        self.length = length  # px
        self.width = width  # px
        self.agent_polys = []
        self.crashed = False
        self.agent_id = agent_id