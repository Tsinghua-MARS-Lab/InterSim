import os
import time
import math

import numpy as np
import torch

import prediction.M2I.guilded_m_pred.src.utils as utils
from prediction.M2I.guilded_m_pred.src.modeling.vectornet import VectorNet
import prediction.M2I.guilded_m_pred.src.utils_cython as utils_cython

import copy
import pickle
from tqdm import tqdm

# import sys
# os.chdir(src_path)
# sys.path[-1] = src_path
# # sys.path.append('//relations/reactor_git_repo/src')
# print(sys.path)


Normalizer = utils.Normalizer

from enum import IntEnum


class AgentType(IntEnum):
    unset = 0
    vehicle = 1
    pedestrian = 2
    cyclist = 3
    other = 4

    @staticmethod
    def to_string(a: int):
        return str(AgentType(a)).split('.')[1]


class TrajectoryType(IntEnum):
    STATIONARY = 0
    STRAIGHT = 1
    STRAIGHT_LEFT = 2
    STRAIGHT_RIGHT = 3
    LEFT_U_TURN = 4
    LEFT_TURN = 5
    RIGHT_U_TURN = 6
    RIGHT_TURN = 7


rare_data = [
    'STATIONARY',
    'STRAIGHT_LEFT',
    'STRAIGHT_RIGHT',
    'LEFT_U_TURN',
]


def compile_pyx_files():
    if True:
        src_path = 'prediction/M2I/guilded_m_pred/src/'
        current_path = os.getcwd()
        os.chdir(src_path)
        # os.system('cython -a utils_cython.pyx && python setup.py build_ext --inplace')
        if not os.path.exists('utils_cython.c') or not os.path.exists('utils_cython.cpython-36m-x86_64-linux-gnu.so') or \
                os.path.getmtime('utils_cython.pyx') > os.path.getmtime('utils_cython.cpython-36m-x86_64-linux-gnu.so'):
            os.system('cython -a utils_cython.pyx && python setup.py build_ext --inplace')
        os.chdir(current_path)


# Comment out this line if pyx files have been compiled manually.
# compile_pyx_files()

def get_normalized(polygons, x, y, angle):
    cos_ = math.cos(angle)
    sin_ = math.sin(angle)
    n = polygons.shape[1]
    new_polygons = np.zeros_like(polygons, dtype=np.float32)
    assert new_polygons.shape[2] == 2, new_polygons.shape
    for polygon_idx in range(polygons.shape[0]):
        for i in range(n):
            polygons[polygon_idx, i, 0] -= x
            polygons[polygon_idx, i, 1] -= y
            new_polygons[polygon_idx, i, 0] = polygons[polygon_idx, i, 0] * cos_ - polygons[polygon_idx, i, 1] * sin_
            new_polygons[polygon_idx, i, 1] = polygons[polygon_idx, i, 0] * sin_ + polygons[polygon_idx, i, 1] * cos_
    return new_polygons


def predict_guilded_trajectory(all_agent_trajectories, all_agent_speeds, track_type_int, time_offset, raw_data,
                               history_frame_num, future_frame_num, objects_id, args,
                               model, device, relevant_agent_ids, ending_points, threshold=0.1):
    mapping_to_return = []
    for selected_agent_id in relevant_agent_ids:
        # predict trajectory for each relevant agents per iteration
        select = np.where(objects_id == selected_agent_id)[0]

        def swap_1(tensor, index):
            if isinstance(tensor[0], int):
                tensor[index], tensor[0] = tensor[0], tensor[index]
            else:
                tensor[index], tensor[0] = tensor[0].copy(), tensor[index].copy()

        all_agent_trajectories_this_batch = all_agent_trajectories.copy()
        objects_id_this_batch = objects_id.copy()

        for each in [all_agent_trajectories_this_batch, objects_id_this_batch]:
            swap_1(each, select)
        assert objects_id_this_batch[0] == selected_agent_id, f'{objects_id_this_batch} and {selected_agent_id}'


        # default without reactor's intentions
        last_valid_index = history_frame_num - 1
        speed = all_agent_speeds[select, history_frame_num - 1]
        # speed = utils.get_dis_point2point((all_agent_trajectories_this_batch[0, history_frame_num - 1, 5], all_agent_trajectories_this_batch[0, history_frame_num - 1, 6]))
        waymo_yaw = all_agent_trajectories_this_batch[0, last_valid_index, 3]
        headings = all_agent_trajectories_this_batch[0, history_frame_num:, 3].copy()
        angle = -waymo_yaw + math.radians(90)
        normalizer = utils.Normalizer(all_agent_trajectories_this_batch[0, last_valid_index, 0],
                                      all_agent_trajectories_this_batch[0, last_valid_index, 1],
                                      angle)
        # print("Tag guilded before normalized")
        # all_agent_trajectories_this_batch[:, :, :] = utils_cython.get_normalized(all_agent_trajectories_this_batch[:, :, :], normalizer)
        all_agent_trajectories_this_batch[:, :, :2] = get_normalized(all_agent_trajectories_this_batch[:, :, :2],
                                                                     normalizer.x,
                                                                     normalizer.y,
                                                                     normalizer.yaw)
        # print("Tag guilded after normalized")
        labels = all_agent_trajectories_this_batch[0, history_frame_num:history_frame_num + future_frame_num, :2].copy()

        image = np.zeros([224, 224, 60], dtype=np.int8)
        args.image = image

        # create some dummies just to check if it works
        gt_future_is_valid = np.ones_like(all_agent_trajectories_this_batch)[:, :, 0]
        # print("Tag guilded before classify")
        trajectory_type = utils_cython.classify_track(gt_future_is_valid[0], all_agent_trajectories_this_batch[0])
        # print("Tag guilded after classify")
        tracks_type = np.ones_like(all_agent_trajectories_this_batch)[:, 0, 0]

        if selected_agent_id not in ending_points:
            # skip relevant but not reguilded agents
            continue
        # # normalize
        # ending_pt2 = np.array(ending_points[selected_agent_id], dtype=np.float32).copy()
        # ending_pt2 = utils_cython.get_normalized(ending_pt2[np.newaxis, np.newaxis, :], normalizer)[0][0]

        ending_pt = np.array(ending_points[selected_agent_id], dtype=np.float32).copy()
        ending_pt[0] -= normalizer.x
        ending_pt[1] -= normalizer.y
        new_ending_pt = ending_pt.copy()
        new_ending_pt[0] = ending_pt[0] * math.cos(normalizer.yaw) - ending_pt[1] * math.sin(normalizer.yaw)
        new_ending_pt[1] = ending_pt[0] * math.sin(normalizer.yaw) + ending_pt[1] * math.cos(normalizer.yaw)

        if time_offset is None:
            time_offset = 0
        # print("Tag guilded before get agents")
        vectors, polyline_spans, trajs = utils_cython.get_agents(all_agent_trajectories_this_batch, gt_future_is_valid, tracks_type,
                                                                 False, args,
                                                                 new_ending_pt)
        # print("Tag guilded after get agents")
        # vectors, polyline_spans, trajs = utils_cython.get_agents(all_agent_trajectories_this_batch, gt_future_is_valid,
        #                                                          tracks_type, False, args)
        map_start_polyline_idx = len(polyline_spans)
        # print("Tag guilded before get roads")
        vectors_, polyline_spans_, goals_2D, lanes = utils_cython.get_roads(raw_data, normalizer, args)
        # print("Tag guilded after get roads")
        polyline_spans_ = polyline_spans_ + len(vectors)
        vectors = np.concatenate([vectors, vectors_])
        polyline_spans = np.concatenate([polyline_spans, polyline_spans_])
        polyline_spans = [slice(each[0], each[1]) for each in polyline_spans]

        stage_one_label = np.argmin(
            [utils.get_dis(lane, all_agent_trajectories_this_batch[0, -1, :2]).min() for lane in lanes]) if len(
            lanes) > 0 else 0

        labels_is_valid = np.array([1])

        mapping = {
            'matrix': vectors,
            'polyline_spans': polyline_spans,
            'map_start_polyline_idx': map_start_polyline_idx,
            'labels': labels,
            'labels_is_valid': labels_is_valid,
            'normalizer': normalizer,
            'goals_2D': goals_2D,
            'polygons': lanes,
            'stage_one_label': stage_one_label,
            'waymo_yaw': waymo_yaw,
            'speed': speed,
            'headings': headings,
            'track_type_int': track_type_int,
            # 'track_type_string': AgentType.to_string(track_type_int),
            'trajectory_type': trajectory_type,
            'tracks_type': tracks_type,
            'eval_time': 80,
            'scenario_id': '001',
            # 'object_id': tf.convert_to_tensor(objects_id)[0],
            # 'inf_id': objects_id_this_batch[1],
            'all_agent_ids': objects_id_this_batch.copy(),
            # 'inf_label': inf_label,
            'image': args.image,
        }
        # if eval_time < 80:
        #     mapping['final_idx'] = eval_time - 1

        final_idx = mapping.get('final_idx', -1)

        mapping['goals_2D_labels'] = np.argmin(utils.get_dis(goals_2D, labels[final_idx]))
        mapping_to_return.append(mapping)

    # print(f"***** predicting {len(mapping_to_return)} samples for guilded trajectory prediction *****")
    if len(mapping_to_return) < 1:
        return {}
    pred_trajectory, pred_score, _ = model(mapping_to_return, device)
    # print(f"***** predicted for offset-{time_offset} *****")
    result_to_return = {}
    for i, each_mapping in enumerate(mapping_to_return):
        agent_id = int(each_mapping['all_agent_ids'][0])
        result_to_return[agent_id] = {
            'rst': pred_trajectory[i],
            'score': pred_score[i]
        }
    # print(f"***** {len(list(result_to_return.keys()))} reguilded result unpacked for {time_offset} *****")
    return result_to_return


class GuildedTrajectoryPredictor:
    def __init__(self):
        self.data = None
        self.last_predict_frame = None
        self.device = None
        self.model = None
        self.args = None
        self.model_path = None
        self.predict_device = 'cuda'

    def __call__(self, **kwargs):
        self.data = kwargs['new_data']
        if self.device is None:
            if kwargs['predict_device']:
                self.device = "cpu"
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"predicting guilded trajectory with {self.device}")
        if self.args is None:
            utils.args = utils.Args
            self.args = utils.args
            self.args.future_frame_num = kwargs['time_horizon']
            assert 'laneGCN-4' in self.args.other_params and 'raster' in self.args.other_params, self.args.other_params
        # if self.model is None:
        #     # feed mappings_to_return to the gpus
        #     model = VectorNet(self.args).to(self.device)
        #     model_recover_path = kwargs['model_path']['guilded_m_pred']
        #     model.eval()
        #     torch.no_grad()
        #     # model_recover_path = '/Users/qiaosun/Documents/PyTrafficSim_Git/driving_simulator/prediction/M2I/guilded_m_pred/pretrained/GoalDn.drop0-9.model.28.bin'
        #     # optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
        #     # print("***** Recover model: %s *****", model_recover_path)
        #     if not torch.cuda.is_available():
        #         model_recover = torch.load(model_recover_path, map_location='cpu')
        #     else:
        #         model_recover = torch.load(model_recover_path)
        #     model.load_state_dict(model_recover)
        #     # model.module.load_state_dict(model_recover)
        #     self.model = model
        if self.model is None:
            self.model_path = kwargs['model_path']['guilded_m_pred']
            self.predict_device = kwargs['predict_device']
            self.load_model()

    def load_model(self):
        # feed mappings_to_return to the gpus
        model = VectorNet(self.args).to(self.device)
        model_recover_path = self.model_path
        model.eval()
        torch.no_grad()
        # print("***** Recover model: %s *****", model_recover_path)
        if not torch.cuda.is_available() or self.predict_device == 'cpu':
            model_recover = torch.load(model_recover_path, map_location='cpu')
        elif self.predict_device == 'mps':
            model_recover = torch.load(model_recover_path, map_location='mps')
        else:
            model_recover = torch.load(model_recover_path)
        model.load_state_dict(model_recover)
        # model.module.load_state_dict(model_recover)
        self.model = model

    # def predict(self, time_interval=5):
    def predict_w_all(self, ending_points, current_frame=11, time_horizon=80):
        """
        :param current_frame:
        :return: trajectory predicted for target agent
        """
        data = copy.deepcopy(self.data)
        args = self.args
        future_frame_num = self.args.future_frame_num
        device = self.device
        model = self.model

        skip_flag = data['skip']
        agent_dic = data['agent']
        assert not skip_flag

        history_frame_num = 11

        all_agent_trajectories = []
        track_type_int = []
        all_agent_speeds = []
        objects_id = np.array(list(agent_dic.keys()))

        select, selected_id = self.data['predicting']['ego_id']
        for i, agent_id in enumerate(agent_dic):
            # add all agent pose and type to a numpy array
            all_agent_trajectories.append(agent_dic[agent_id]['pose'].copy())
            track_type_int.append(agent_dic[agent_id]['type'])
            all_agent_speeds.append(agent_dic[agent_id]['speed'][:, 0])
        assert select != -1, 'no interact agent found, failed to select ego in the predictor'
        all_agent_trajectories = np.array(all_agent_trajectories, dtype=np.float32)
        track_type_int = np.array(track_type_int)
        all_agent_speeds = np.array(all_agent_speeds, dtype=np.float32)

        trajectory_predicted = {
            'sample_agent_id': np.zeros([80, 2])
        }
        # for current_frame in range(0, future_frame_num, time_interval):
        if current_frame > 0:
            all_agent_trajectories = np.concatenate([all_agent_trajectories[:, current_frame:, :],
                                                     np.zeros_like(all_agent_trajectories[:, :current_frame, :])],
                                                    axis=1)

        relevant_agent_ids = self.data['predicting']['relevant_agents']
        predicted_m_trajectories = predict_guilded_trajectory(all_agent_trajectories=all_agent_trajectories,
                                                              all_agent_speeds=all_agent_speeds,
                                                              track_type_int=track_type_int,
                                                              time_offset=current_frame,
                                                              history_frame_num=history_frame_num,
                                                              future_frame_num=future_frame_num,
                                                              objects_id=objects_id,
                                                              raw_data=data['raw'],
                                                              args=args,
                                                              model=model, device=device,
                                                              relevant_agent_ids=relevant_agent_ids,
                                                              ending_points=ending_points)
        torch.cuda.empty_cache()
        self.args.image = None

        # print(f"debug in marginal predictor: {predicted_m_trajectories[752]}")

        return predicted_m_trajectories


    def predict(self, ending_points, current_frame=11, time_horizon=80):
        """
        :param current_frame:
        :return: trajectory predicted for target agent
        """
        data = copy.deepcopy(self.data)
        args = self.args
        future_frame_num = self.args.future_frame_num
        device = self.device
        model = self.model

        skip_flag = data['skip']
        agent_dic = data['agent']
        assert not skip_flag

        history_frame_num = 11

        all_agent_trajectories = []
        track_type_int = []
        all_agent_speeds = []
        objects_id = []
        select, selected_id = self.data['predicting']['ego_id']
        for i, agent_id in enumerate(agent_dic):
            # add all agent pose and type to a numpy array
            if agent_id not in self.data['predicting']['relevant_agents']:
                continue
            objects_id.append(agent_id)
            all_agent_trajectories.append(agent_dic[agent_id]['pose'].copy())
            track_type_int.append(agent_dic[agent_id]['type'])
            all_agent_speeds.append(agent_dic[agent_id]['speed'][:, 0])
        objects_id = np.array(objects_id)
        assert select != -1, 'no interact agent found, failed to select ego in the predictor'
        all_agent_trajectories = np.array(all_agent_trajectories, dtype=np.float32)
        track_type_int = np.array(track_type_int)
        all_agent_speeds = np.array(all_agent_speeds, dtype=np.float32)

        trajectory_predicted = {
            'sample_agent_id': np.zeros([80, 2])
        }
        # for current_frame in range(0, future_frame_num, time_interval):
        if current_frame > 0:
            all_agent_trajectories = np.concatenate([all_agent_trajectories[:, current_frame:, :],
                                                     np.zeros_like(all_agent_trajectories[:, :current_frame, :])],
                                                    axis=1)

        relevant_agent_ids = self.data['predicting']['relevant_agents']
        predicted_m_trajectories = predict_guilded_trajectory(all_agent_trajectories=all_agent_trajectories,
                                                              all_agent_speeds=all_agent_speeds,
                                                              track_type_int=track_type_int,
                                                              time_offset=current_frame,
                                                              history_frame_num=history_frame_num,
                                                              future_frame_num=future_frame_num,
                                                              objects_id=objects_id,
                                                              raw_data=data['raw'],
                                                              args=args,
                                                              model=model, device=device,
                                                              relevant_agent_ids=relevant_agent_ids,
                                                              ending_points=ending_points)
        torch.cuda.empty_cache()
        self.args.image = None

        # print(f"debug in marginal predictor: {predicted_m_trajectories[752]}")

        return predicted_m_trajectories
    


