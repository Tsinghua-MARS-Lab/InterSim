from prediction.M2I.relation_predictor.predict import RelationPredictor
from prediction.M2I.goal_setter.static_goal import GoalSetter
from prediction.M2I.marginal_prediction.predict import MarginalTrajectoryPredictor
# from prediction.M2I.marginal_prediction_tnt.predict import MarginalTrajectoryPredictorTNT
# # from prediction.M2I.reactor_prediction.predict import ReactorTrajectoryPredictor
# from prediction.M2I.xpt_prediction.predict import XPtPredictor
# from prediction.M2I.guilded_m_pred.predict import GuildedTrajectoryPredictor
# from prediction.M2I.variety_loss_prediction.predict import VarietyLossTrajectoryPredictor
import numpy as np
import copy
import time

"""
This is the main entrance of the M2I predictor
Predicting related data will be stored in self.data['predicting']
"""

import math
def euclidean_distance(pt1, pt2):
    x_1, y_1 = pt1
    x_2, y_2 = pt2
    return math.sqrt((x_1-x_2)**2+(y_1-y_2)**2)

def get_normalized(line, x, y, angle):
    cos_ = math.cos(angle)
    sin_ = math.sin(angle)
    new_line = np.zeros_like(line, dtype=np.float32)
    assert new_line.shape[1] == 2, new_line.shape
    for i, each_pt in enumerate(line):
        line[i, 0] -= x
        line[i, 1] -= y
        new_line[i, 0] = line[i, 0] * cos_ - line[i , 1] * sin_
        new_line[i, 1] = line[i, 0] * sin_ + line[i, 1] * cos_
    return new_line

def _raster_float_to_int(a, is_y, scale):
    if not is_y:
        return int(a * scale + 10000 + 0.5) - 10000 + 112
    else:
        return int(a * scale + 10000 + 0.5) - 10000 + 56

def _raster_int_to_float(a, is_y, scale):
    if not is_y:
        return (a - 112) / scale
    else:
        return (a - 56) / scale

def _in_image(x, y):
    return 0 <= x < 224 and 0 <= y < 224

def vectorize_agent(gt_trajectory, gt_future_is_valid, tracks_type, args):
    history_frame_num = 11
    max_vector_num = 10000
    scale = 1
    raster_scale = 1
    len_vectors = 0
    polyline_num = 0

    vectors = np.zeros([max_vector_num, 128], dtype=np.float32)
    polyline_spans = np.zeros([max_vector_num, 2], dtype=np.int32)
    image = args.image
    agent_num = gt_trajectory.shape[0]
    for i in range(agent_num):
        start = len_vectors
        for j in range(history_frame_num):
            if not gt_future_is_valid[i, j]:
                gt_trajectory[i, j, :] = 0
        # do_raster
        for j in range(history_frame_num):
            if gt_future_is_valid[i, j]:
                x = _raster_float_to_int(gt_trajectory[i, j, 0], 0, raster_scale)
                y = _raster_float_to_int(gt_trajectory[i, j, 1], 1, raster_scale)
                if _in_image(x, y):
                    if i == 0:
                        image[x, y, j] = 1
                    else:
                        image[x, y, j + 20] = 1
        # do_vector
        for j in range(history_frame_num - 1):
            cur = 0
            for k in range(7):
                vectors[len_vectors, cur + k] = gt_trajectory[i, j, k] * scale if k < 2 else 1
            cur = 20
            for k in range(7):
                vectors[len_vectors, cur + k] = gt_trajectory[i, j + 1, k] * scale if k < 2 else 1
            cur = 30
            vectors[len_vectors, cur + 0] = j
            vectors[len_vectors, cur + 1 + j] = 1

            cur = 50
            vectors[len_vectors, cur + 0] = tracks_type[i]
            vectors[len_vectors, cur + 1 + int(tracks_type[i])] = 1

            len_vectors += 1
            assert len_vectors < max_vector_num
        polyline_spans[polyline_num, 0], polyline_spans[polyline_num, 1] = start, len_vectors
        polyline_num += 1
    assert len_vectors <=max_vector_num
    args.image = image
    return vectors, polyline_spans, args

def vectorize_roads(road_dic, normalizer, args):
    max_vector_num = 10000
    polyline_num = 0
    max_point_num = 2500
    max_lane_num = 1000
    len_vectors = 0
    max_goals_2D_num = 100000
    goals_2d_len = 0
    raster_scale = 1
    lane_types = [0, 11]  # for NuPlan

    image = args.image
    vectors = np.zeros([max_vector_num, 128], dtype=np.float32)
    polyline_spans = np.zeros([max_vector_num, 2], dtype=np.int32)
    goals_2d = np.zeros([max_goals_2D_num, 2], dtype=np.float32)  # not used in relation prediction decoder

    lanes = []
    stride = 10
    scale = 0.03

    for each_lane_id in road_dic:
        # if road_dic[each_lane_id]['type'] not in lane_types:
        #     continue
        road_type = int(road_dic[each_lane_id]['type'])
        xyz = road_dic[each_lane_id]['xyz']
        if isinstance(xyz, list):
            length = len(xyz)
            xyz_np = np.array(xyz)
        else:
            xyz_np = xyz.copy()
            if len(xyz_np.shape) < 2:
                print('skipping road element: \n', road_dic[each_lane_id])
                continue
            length = xyz.shape[0]
        xyz_np = get_normalized(xyz_np[:, :2], normalizer.x, normalizer.y, normalizer.yaw)
        for each_pt in xyz_np:
            # rasterize
            x_int = _raster_float_to_int(each_pt[0], 0, raster_scale)
            y_int = _raster_float_to_int(each_pt[1], 0, raster_scale)
            if _in_image(x_int, y_int):
                image[x_int, y_int, 40 + road_type] = 1
        # do vector
        start = len_vectors
        len_vectors += 1
        polyline_spans[polyline_num, 0], polyline_spans[polyline_num, 1] = start, len_vectors
        polyline_num += 1

        lane = np.zeros([1, 2], dtype=np.float32)
        lanes.append(lane)

    args.image = image
    return vectors, polyline_spans, args, lanes


class M2IPredictor:
    def __init__(self, **kwargs):
        self.data = None
        self.relation_predictor = RelationPredictor()
        self.goal_setter = GoalSetter()
        self.marginal_predictor = MarginalTrajectoryPredictor()
        # self.marginal_predictor = MarginalTrajectoryPredictorTNT()
        # self.reactor_predictor = ReactorTrajectoryPredictor()
        # self.xpt_predictor = XPtPredictor()
        # self.guilded_predictor = GuildedTrajectoryPredictor()
        self.conditional_predictor = None
        self.predicting_horizon = kwargs['time_horizon'] if 'time_horizon' in kwargs else 80
        self.prediction_data = None

        self.dataset = 'Waymo'

        # self.variety_predictor = VarietyLossTrajectoryPredictor()

    def __call__(self, **kwargs):
        self.data = kwargs['new_data']
        predictor_list = kwargs['predictor_list']
        use_prediction = kwargs['use_prediction']
        if True: # 'predicting' not in self.data:
            self.data['predicting'] = {
                'ego_id': None,  # [select, selected_agent_id]
                'goal_pts': {
                    # 'sample_agent_id': [[0.0, 0.0], 3.14]
                },
                'follow_goal': {
                    # 'sample_agent_id': True
                },
                'relevant_agents': [],
                'relation': np.array([]),
                'colliding_pairs': [],
                'marginal_trajectory': {
                    # 'sample_agent_id': {
                    #     'rst': np.zeros([self.predicting_horizon, 6, 2]),
                    #     'score': np.zeros([self.predicting_horizon, 6])
                    # }
                },
                'conditional_trajectory': {
                    # 'sample_agent_id': {
                    #     'rst': np.zeros([self.predicting_horizon, 6, 2]),
                    #     'score': np.zeros([self.predicting_horizon, 6])
                    # }
                },
                'guilded_trajectory': {
                    # 'sample_agent_id': {
                    #     'rst': np.zeros([self.predicting_horizon, 6, 2]),
                    #     'score': np.zeros([self.predicting_horizon, 6])
                    # }
                },
                'XPt': {
                    # (reactor_id, inf_id): {
                    # 'pred_collision_pt': [0.0, 0.0],
                    # 'pred_cp_scores': 0.0,
                    # 'scenarios_id': scenario_ids[i],
                    # 'all_agents': all_agent_ids[i]}
                },
                'original_trajectory': copy.deepcopy(self.data['agent']),
                'points_to_mark': [],
                'trajectory_to_mark': [],
                'emergency_stopping': False,
                'route': {},
                'all_relations_last_step': [],
                'goal_fit': None,  # True if goal fit the assigned route, calculate goal past
                'terminate_log': []  # [terminate_frame_id, terminate_reason] terminate_reason: 'collision', 'offroad'
            }
        self.predicting_horizon = kwargs['time_horizon']
        # self.goal_setter()
        if predictor_list is not None:
            _, relation_predictor, marginal_predictor = predictor_list
            # self.relation_predictor = relation_predictor
            # if use_prediction:
            #     self.marginal_predictor = marginal_predictor
            self.relation_predictor(model_path=kwargs['model_path'],
                                    predict_device=kwargs['predict_device'],
                                    model=relation_predictor.model)
            if use_prediction:
                self.marginal_predictor(model_path=kwargs['model_path'],
                                        time_horizon=kwargs['time_horizon'],
                                        predict_device=kwargs['predict_device'],
                                        model=marginal_predictor.model)

        else:
            self.relation_predictor(model_path=kwargs['model_path'], predict_device=kwargs['predict_device'])
            if use_prediction:
                self.marginal_predictor(model_path=kwargs['model_path'], time_horizon=kwargs['time_horizon'], predict_device=kwargs['predict_device'])
            # self.reactor_predictor(new_data=self.data)
            # self.xpt_predictor(new_data=self.data)
            # self.guilded_predictor(new_data=self.data, model_path=kwargs['model_path'], time_horizon=kwargs['time_horizon'], predict_device=kwargs['predict_device'])
            # self.variety_predictor(new_data=self.data, model_path=kwargs['model_path'], time_horizon=kwargs['time_horizon'], predict_device=kwargs['predict_device'])

        # select one ego vehicle
        agent_dic = self.data['agent']
        select = -1
        selected_agent_id = -1

        if self.dataset == 'Waymo':
            interact_only = False

            for i, agent_id in enumerate(agent_dic):
                # select the ego agent from the two interactive agents once per scenario
                # TODO: extend ego selection for the validation set
                if agent_dic[agent_id]['type'] != 1:
                    continue
                if interact_only:
                    if agent_dic[agent_id]['to_interact']:
                        if np.max(agent_dic[agent_id]['pose'][:11, :2]) == -1:
                            print("skip invalid")
                        else:
                            select = i
                            selected_agent_id = agent_id
                else:
                    if agent_dic[agent_id]['to_predict']:
                        if np.max(agent_dic[agent_id]['pose'][:11, :2]) == -1:
                            print("skip invalid")
                        else:
                            select = i
                            selected_agent_id = agent_id

            # selected_agent_id = 857
            # selecting ego agent for planning
            self.data['predicting']['ego_id'] = [select, selected_agent_id]

            # if len(list(agent_dic.keys())) > 30:
            #     self.data['skip'] = True
            #     print("skipping large agent number scenarios")

            if selected_agent_id == -1:
                print("Predictor Skip without ego agent")
                self.data['skip'] = True

        elif self.dataset == 'NuPlan':
            if 'ego' in agent_dic:
                selected_agent_id = 'ego'
                select = list(agent_dic.keys()).index('ego')
            else:
                print("Predictor Skip without ego agent")
                self.data['skip'] = True

            self.data['predicting']['ego_id'] = [select, selected_agent_id]


    def relationship_pred(self, current_frame=0):
        # return self.relation_predictor.predict_oneframe()
        self.data['predicting']['relation'], relevant_agents = self.relation_predictor.predict(current_frame)
        for each_agent_id in relevant_agents:
            if each_agent_id not in self.data['predicting']['relevant_agents']:
                self.data['predicting']['relevant_agents'].append(each_agent_id)

    def setting_goal_points(self, current_data):
        # select one ego vehicle
        agent_dic = current_data['agent']
        for i, agent_id in enumerate(agent_dic):
            # set the goal point for each agent
            self.data['predicting']['goal_pts'][agent_id] = self.goal_setter.get_goal(current_data=current_data,
                                                                                      agent_id=agent_id,
                                                                                      dataset=self.dataset)
            if self.data['predicting']['goal_pts'][agent_id][0] is not None:
                self.data['predicting']['follow_goal'][agent_id] = True
            else:
                self.data['predicting']['follow_goal'][agent_id] = False
        print('Goal points settled')

    def variety_predict(self, current_frame=0):
        marginal_pred = self.variety_predictor.predict(current_frame)
        for each_agent_id in marginal_pred:
            self.data['predicting']['marginal_trajectory'][each_agent_id] = marginal_pred[each_agent_id]

    def marginal_predict(self, current_data, current_frame=0, selected_agent_ids=None):
        # pass selected agent ids to predict these agents, not passing selected ids to predict relevant agents
        marginal_pred = self.marginal_predictor.predict(current_frame=current_frame, selected_agent_ids=selected_agent_ids, current_data=current_data)
        for each_agent_id in marginal_pred:
            self.data['predicting']['marginal_trajectory'][each_agent_id] = marginal_pred[each_agent_id]

    def reactor_predict(self, current_frame=0):
        reactor_pred = self.reactor_predictor.predict(current_frame)
        for each_agent_id in reactor_pred:
            self.data['predicting']['conditional_trajectory'][each_agent_id] = reactor_pred[each_agent_id]
        self.update_state(self.data)

    def xpt_predict(self, current_frame=0):
        xPt_pred = self.xpt_predictor.predict(current_frame)
        for each_pair in xPt_pred:
            self.data['predicting']['XPt'][each_pair] = xPt_pred[each_pair]

    # def guilded_predict(self, ending_points, current_frame=0):
    #     guilded_pred = self.guilded_predictor.predict(ending_points=ending_points, current_frame=current_frame)
    #     for each_agent_id in guilded_pred:
    #         # guilded_pred[each_agent_id]['rst'] = guilded_pred[each_agent_id]['rst'][:self.predicting_horizon, :]
    #         for i in range(6):
    #             guilded_pred[each_agent_id]['rst'][i] = self.filter_trajectory_after_goal_point(traj=guilded_pred[each_agent_id]['rst'][i],
    #                                                                                             goal_point=ending_points[each_agent_id])
    #         self.data['predicting']['guilded_trajectory'][each_agent_id] = guilded_pred[each_agent_id]
    #         # self.data['predicting']['marginal_trajectory'][each_agent_id] = guilded_pred[each_agent_id]

    # def guilded_marginal_predict(self, ending_points, current_frame=0):
    #     guilded_pred = self.guilded_predictor.predict(ending_points=ending_points, current_frame=current_frame)
    #     for each_agent_id in guilded_pred:
    #         # guilded_pred[each_agent_id]['rst'] = guilded_pred[each_agent_id]['rst'][:self.predicting_horizon, :]
    #         for i in range(6):
    #             guilded_pred[each_agent_id]['rst'][i] = self.filter_trajectory_after_goal_point(traj=guilded_pred[each_agent_id]['rst'][i],
    #                                                                                             goal_point=ending_points[each_agent_id])
    #         if euclidean_distance(guilded_pred[each_agent_id]['rst'][0, 0, :2], guilded_pred[each_agent_id]['rst'][0, 50, :2]) < 10:
    #             pass
    #         else:
    #             self.data['predicting']['marginal_trajectory'][each_agent_id] = guilded_pred[each_agent_id]

    def relation_pred_onetime(self, each_pair, current_data, current_frame=0, clear_history=False, with_rules=True):
        if clear_history:
            self.data['predicting']['relation'] = self.relation_predictor.predict_one_time(each_pair=each_pair, current_frame=current_frame, current_data=current_data, predict_with_rules=with_rules)
        else:
            self.data['predicting']['relation'] += self.relation_predictor.predict_one_time(each_pair=each_pair, current_frame=current_frame, current_data=current_data, predict_with_rules=with_rules)
        # if [857, 996] in self.data['predicting']['relation']:
        # self.data['predicting']['relation'] = [[996, 857]]

    # def update_state(self, current_state):
    #     self.marginal_predictor.data = current_state
    #     # self.reactor_predictor.data = current_state
    #     self.relation_predictor.data = current_state
    #     # self.xpt_predictor.data = current_state
    #     # self.guilded_predictor.data = current_state

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
