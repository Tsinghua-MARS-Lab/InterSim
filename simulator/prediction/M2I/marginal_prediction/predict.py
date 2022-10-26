import os
import time
import math

import numpy as np
import torch

import prediction.M2I.marginal_prediction.src.utils as utils
from prediction.M2I.marginal_prediction.src.modeling.vectornet import VectorNet
import prediction.M2I.marginal_prediction.src.utils_cython as utils_cython

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
        src_path = 'prediction/M2I/marginal_prediction/src/'
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


def predict_marginal_trajectory(all_agent_trajectories, all_agent_speeds, track_type_int, time_offset, raw_data,
                                history_frame_num, future_frame_num, objects_id, args,
                                model, device, relevant_agent_ids, threshold=0.1):
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
        # print("Tag: marginal prediction before normalize")
        # all_agent_trajectories_this_batch[:, :, :] = utils_cython.get_normalized(all_agent_trajectories_this_batch[:, :, :], normalizer)
        all_agent_trajectories_this_batch[:, :, :2] = get_normalized(all_agent_trajectories_this_batch[:, :, :2],
                                                                     normalizer.x, normalizer.y, normalizer.yaw)
        # print("Tag: marginal prediction after normalize")
        labels = all_agent_trajectories_this_batch[0, history_frame_num:history_frame_num + future_frame_num, :2].copy()

        image = np.zeros([224, 224, 60], dtype=np.int8)
        args.image = image

        # create some dummies just to check if it works
        gt_future_is_valid = np.ones_like(all_agent_trajectories_this_batch)[:, :, 0]
        # print("Tag: marginal prediction before classify track")
        trajectory_type = utils_cython.classify_track(gt_future_is_valid[0], all_agent_trajectories_this_batch[0])
        # print("Tag: marginal prediction after classify track")
        tracks_type = np.ones_like(all_agent_trajectories_this_batch)[:, 0, 0]
        # print("Tag: marginal prediction before get agents")

        # all_agent_trajectories_this_batch = all_agent_trajectories_this_batch[:1, :, :]
        # gt_future_is_valid = gt_future_is_valid[:1, :]

        vectors, polyline_spans, trajs = utils_cython.get_agents(all_agent_trajectories_this_batch, gt_future_is_valid,
                                                                 tracks_type, False, args)
        # print("Tag: marginal prediction after get agents")
        map_start_polyline_idx = len(polyline_spans)
        # print("Tag: marginal prediction before get roads")
        vectors_, polyline_spans_, goals_2D, lanes = utils_cython.get_roads(raw_data, normalizer, args)
        # print("Tag: marginal prediction after get roads")
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
            'image': args.image
        }
        # if eval_time < 80:
        #     mapping['final_idx'] = eval_time - 1

        final_idx = mapping.get('final_idx', -1)

        mapping['goals_2D_labels'] = np.argmin(utils.get_dis(goals_2D, labels[final_idx]))
        mapping_to_return.append(mapping)

    # print(f"***** predicting {len(mapping_to_return)} samples for marginal trajectory prediction *****")
    if len(mapping_to_return) < 1:
        return {}
    result_to_return = {}
    for each_mapping in mapping_to_return:
        # calculate per agent to save GPU memory
        pred_trajectory, pred_score, _ = model([each_mapping], device)
        agent_id = int(each_mapping['all_agent_ids'][0])
        result_to_return[agent_id] = {
            'rst': pred_trajectory[0],  # k results included (already), 0 is for batch index
            'score': pred_score[0]
        }
    # pred_trajectory, pred_score, _ = model(mapping_to_return, device)
    # # print(f"***** predicted for {time_offset} *****")
    # for i, each_mapping in enumerate(mapping_to_return):
    #     agent_id = int(each_mapping['all_agent_ids'][0])
    #     result_to_return[agent_id] = {
    #         'rst': pred_trajectory[i],
    #         'score': pred_score[i]
    #     }
    # print(f"***** {len(list(result_to_return.keys()))} marginal result unpacked for {time_offset} *****")
    return result_to_return


class MarginalTrajectoryPredictor:
    def __init__(self):
        self.data = None
        self.last_predict_frame = None
        self.model = None
        self.device = None
        self.args = None
        self.model_path = None
        self.predict_device = 'cuda'
        self.max_prediction_num = 128
        self.rank = None

        self.predicting_lock = False

        self.call_and_init = False

    def __call__(self, **kwargs):
        self.call_and_init = True
        if self.device is None:
            predict_device = kwargs['predict_device']
            if predict_device in ['cpu', 'mps']:
                self.device = predict_device
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu", 0)
                self.rank = 0
                print(f"predicting marginal trajectory with {self.device} and max at {self.max_prediction_num}")
        if self.args is None:
            utils.args = utils.Args
            self.args = utils.args
            self.args.future_frame_num = kwargs['time_horizon']
            assert 'laneGCN-4' in self.args.other_params and 'raster' in self.args.other_params, self.args.other_params

        if self.model is None:
            if 'model' in kwargs and kwargs['model'] is not None:
                self.model = kwargs['model']
            else:
                self.model_path = kwargs['model_path']['marginal_pred']
                self.predict_device = kwargs['predict_device']
                self.load_model()
        #
        # if self.model is None:
        #     # feed mappings_to_return to the gpus
        #     model = VectorNet(self.args).to(self.device)
        #     model_recover_path = kwargs['model_path']['marginal_pred']
        #     model.eval()
        #     torch.no_grad()
        #     # model_recover_path = '/Users/qiaosun/Documents/PyTrafficSim_Git/driving_simulator/prediction/M2I/marginal_prediction/pretrained/marginalPred.drop3-10.model.25.bin'
        #     # model_recover_path = '/Users/qiaosun/Documents/PyTrafficSim_Git/driving_simulator/prediction/M2I/marginal_prediction/pretrained/marginalPred.drop0-9.model.30.bin'
        #     # model_recover_path = '/Users/qiaosun/Documents/PyTrafficSim_Git/driving_simulator/prediction/M2I/marginal_prediction/pretrained/marginalPred.tfR.model.21.bin'
        #     # optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
        #     # print("***** Recover model: %s *****", model_recover_path)
        #     if not torch.cuda.is_available() or kwargs['predict_device']:
        #         model_recover = torch.load(model_recover_path, map_location='cpu')
        #     else:
        #         model_recover = torch.load(model_recover_path)
        #     model.load_state_dict(model_recover)
        #     # model.module.load_state_dict(model_recover)
        #     self.model = model

    def load_model(self):
        print("Start loading marginal prediction model")
        # feed mappings_to_return to the gpus
        if self.rank is not None:
            # setup(self.rank, 512)
            torch.cuda.set_device(self.rank)
        model = VectorNet(self.args).to(self.device)
        model_recover_path = self.model_path
        model.eval()
        torch.no_grad()
        # print("***** Recover model: %s *****", model_recover_path)
        if not torch.cuda.is_available() or self.predict_device:
            model_recover = torch.load(model_recover_path, map_location='cpu')
        else:
            model_recover = torch.load(model_recover_path)
        model.load_state_dict(model_recover)
        # model.module.load_state_dict(model_recover)
        self.model = model
        print("Model loaded for marginal prediction")

    # def predict(self, time_interval=5):
    def predict(self, current_data, current_frame=11, selected_agent_ids=None):
        """
        :param selected_agent_ids:
        1. pass None to predict for all relevant agents detected by env planner
        2. pass a LIST of selected agent_ids to predict one trajectory for each agent in the list
        3. pass one agent_id to predict only for this agent
        4. pass 'all' to predict trajectory for all agents non-stop agents
        :return: trajectory predicted for target agent
        """
        self.predicting_lock = True
        assert self.call_and_init
        data = copy.deepcopy(current_data)
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

        select, selected_id = current_data['predicting']['ego_id']

        relevant_agent_ids = []
        for i, agent_id in enumerate(agent_dic):
            if len(all_agent_trajectories) >= self.max_prediction_num:
                break
            # add all agent pose and type to a numpy array
            all_agent_trajectories.append(agent_dic[agent_id]['pose'].copy())
            track_type_int.append(agent_dic[agent_id]['type'])
            all_agent_speeds.append(agent_dic[agent_id]['speed'][:, 0])
            if selected_agent_ids == 'all':
                # filter steady ones to save memory
                if utils.get_dis_point2point(agent_dic[agent_id]['pose'][current_frame, :2], agent_dic[agent_id]['pose'][current_frame-10, :2]) < 0.5:
                    continue
                relevant_agent_ids.append(agent_id)
            elif isinstance(selected_agent_ids, list) and agent_id in selected_agent_ids:
                if utils.get_dis_point2point(agent_dic[agent_id]['pose'][current_frame, :2],
                                             agent_dic[agent_id]['pose'][current_frame - 10, :2]) < 0.5:
                    continue
                relevant_agent_ids.append(agent_id)

        if selected_agent_ids is None:
            relevant_agent_ids = current_data['predicting']['relevant_agents']
        elif isinstance(selected_agent_ids, int):
            relevant_agent_ids = [selected_agent_ids]


        if len(relevant_agent_ids) == 0:
            # nothing in the scenario to predict at all
            print("Warning: Marginal predictor got nothing to predict")
            return {}

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

        # if selected_agent_ids is None:
        #     relevant_agent_ids = self.data['predicting']['relevant_agents']
        # elif selected_agent_ids == 'all':
        #     pass
        #     # relevant_agent_ids = list(agent_dic.keys())
        # else:
        #     relevant_agent_ids = selected_agent_ids

        # relevant_agent_ids = [731, 932, 752, 750]
        predicted_m_trajectories = predict_marginal_trajectory(all_agent_trajectories=all_agent_trajectories,
                                                               all_agent_speeds=all_agent_speeds,
                                                               track_type_int=track_type_int,
                                                               time_offset=current_frame,
                                                               history_frame_num=history_frame_num,
                                                               future_frame_num=future_frame_num,
                                                               objects_id=objects_id,
                                                               raw_data=data['raw'],
                                                               args=args,
                                                               model=model, device=device,
                                                               relevant_agent_ids=relevant_agent_ids)

        # print(f"debug in marginal predictor: {predicted_m_trajectories[752]}")
        torch.cuda.empty_cache()
        self.args.image = None
        self.predicting_lock = False
        return predicted_m_trajectories
