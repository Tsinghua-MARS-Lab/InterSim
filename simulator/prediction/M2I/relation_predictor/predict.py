import os
import time
import math

import numpy as np
import torch

import prediction.M2I.relation_predictor.src.utils as utils
from prediction.M2I.relation_predictor.src.modeling.vectornet import VectorNet
import prediction.M2I.relation_predictor.src.utils_cython as utils_cython
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

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
        src_path = 'prediction/M2I/relation_predictor/src/'
        current_path = os.getcwd()
        os.chdir(src_path)
        # os.system('cython -a utils_cython.pyx && python setup.py build_ext --inplace')
        if not os.path.exists('utils_cython.c') or not os.path.exists('utils_cython.cpython-36m-x86_64-linux-gnu.so') or \
                os.path.getmtime('utils_cython.pyx') > os.path.getmtime('utils_cython.cpython-36m-x86_64-linux-gnu.so'):
            os.system('cython -a utils_cython.pyx && python setup.py build_ext --inplace')
        os.chdir(current_path)


# Comment out this line if pyx files have been compiled manually.
# compile_pyx_files()


def predict_influencer(all_agent_trajectories, track_type_int, time_offset, raw_data,
                       history_frame_num, future_frame_num, objects_id, args, device):
    # default without reactor's intentions
    gt_reactor = np.zeros_like(all_agent_trajectories[0])
    last_valid_index = history_frame_num - 1
    # speed = utils.get_dis_point2point((gt_trajectory[0, history_frame_num - 1, 5], gt_trajectory[0, history_frame_num - 1, 6]))
    waymo_yaw = all_agent_trajectories[0, last_valid_index, 3]
    # headings = gt_trajectory[0, history_frame_num:, 4].copy()
    angle = -waymo_yaw + math.radians(90)
    normalizer = utils.Normalizer(all_agent_trajectories[0, last_valid_index, 0],
                                  all_agent_trajectories[0, last_valid_index, 1],
                                  angle)
    all_agent_trajectories[:, :, :] = utils_cython.get_normalized(all_agent_trajectories[:, :, :], normalizer)
    gt_reactor[:, :] = utils_cython.get_normalized(gt_reactor[:, :][np.newaxis, :], normalizer)[0]

    labels = all_agent_trajectories[0, history_frame_num:history_frame_num + future_frame_num, :2].copy()

    mapping_to_return = []

    image = np.zeros([224, 224, 60 + 90], dtype=np.int8)
    args.image = image

    def swap_1(tensor, index):
        if isinstance(tensor[0], int):
            tensor[index], tensor[1] = tensor[1], tensor[index]
        else:
            tensor[index], tensor[1] = tensor[1].copy(), tensor[index].copy()

    for i in range(1, objects_id.shape[0]):
        target_id = objects_id[i]
        all_agent_trajectories_this_batch = all_agent_trajectories.copy()
        objects_id_this_batch = objects_id.copy()
        for each in [all_agent_trajectories_this_batch, objects_id_this_batch]:
            swap_1(each, i)
        assert objects_id_this_batch[1] == target_id
        # create some dummies just to check if it works
        gt_future_is_valid = np.ones_like(all_agent_trajectories_this_batch)[:, :, 0]
        trajectory_type = utils_cython.classify_track(gt_future_is_valid[i], all_agent_trajectories_this_batch[i])
        if trajectory_type == 'STATIONARY':
            continue
        tracks_type = np.ones_like(all_agent_trajectories_this_batch)[:, 0, 0]
        vectors, polyline_spans, trajs = utils_cython.get_agents(all_agent_trajectories_this_batch, gt_future_is_valid,
                                                                 tracks_type, False, args, gt_reactor)
        map_start_polyline_idx = len(polyline_spans)

        vectors_, polyline_spans_, goals_2D, lanes = utils_cython.get_roads(raw_data, normalizer, args)
        polyline_spans_ = polyline_spans_ + len(vectors)
        vectors = np.concatenate([vectors, vectors_])
        polyline_spans = np.concatenate([polyline_spans, polyline_spans_])
        polyline_spans = [slice(each[0], each[1]) for each in polyline_spans]

        stage_one_label = np.argmin(
            [utils.get_dis(lane, all_agent_trajectories_this_batch[0, -1, :2]).min() for lane in lanes]) if len(
            lanes) > 0 else 0

        mapping = {
            'matrix': vectors,
            'polyline_spans': polyline_spans,
            'map_start_polyline_idx': map_start_polyline_idx,
            'labels': labels,
            'normalizer': normalizer,
            'goals_2D': goals_2D,
            'polygons': lanes,
            'stage_one_label': stage_one_label,
            'waymo_yaw': waymo_yaw,
            'track_type_int': track_type_int,
            # 'track_type_string': AgentType.to_string(track_type_int),
            'trajectory_type': trajectory_type,
            'tracks_type': tracks_type,
            'eval_time': 80,
            'scenario_id': '001',
            # 'object_id': tf.convert_to_tensor(objects_id)[0],
            'inf_id': objects_id_this_batch[1],
            'all_agent_ids': objects_id_this_batch.copy(),
            # 'inf_label': inf_label,
            'image': args.image
        }
        # if eval_time < 80:
        #     mapping['final_idx'] = eval_time - 1

        final_idx = mapping.get('final_idx', -1)

        mapping['goals_2D_labels'] = np.argmin(utils.get_dis(goals_2D, labels[final_idx]))
        mapping_to_return.append(mapping)

    # feed mappings_to_return to the gpus
    model = VectorNet(args).to(device)
    model_recover_path = '/Users/qiaosun/Documents/PyTrafficSim_Git/driving_simulator/prediction/M2I/relation_predictor/pretrained/infPred.bin.IA.v2x.noIntention.1221.model.30.bin'
    # optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    # print("***** Recover model: %s *****", model_recover_path)
    model_recover = torch.load(model_recover_path, map_location='cpu')
    model.load_state_dict(model_recover)
    # model.module.load_state_dict(model_recover)
    # print(f"***** model loaded, predicting {len(mapping_to_return)} samples *****")
    scores, all_agent_ids, scenario_ids = model(mapping_to_return, device)
    # print(f"***** model predicted for {time_offset} *****")
    # result_to_return = {}
    # for i, each_mapping in enumerate(mapping_to_return):
    #     reactor_id = all_agent_ids[i][0]
    #     inf_id = all_agent_ids[i][1]
    #     if reactor_id not in result_to_return:
    #         result_to_return[reactor_id] = {}
    #     if time_offset not in result_to_return[reactor_id]:
    #         result_to_return[reactor_id][time_offset] = {
    #             'pred_inf_id': [],
    #             'pred_inf_scores': [],
    #             'pred_inf_label': []
    #         }
    #     result_to_return[reactor_id][time_offset]['pred_inf_id'].append(int(all_agent_ids[i][1]))
    #     result_to_return[reactor_id][time_offset]['pred_inf_scores'].append(scores[i])
    #     result_to_return[reactor_id][time_offset]['pred_inf_label'].append(np.argmax(scores[i]))
    result_to_return = []
    for i, each_mapping in enumerate(mapping_to_return):
        reactor_id = int(all_agent_ids[i][0])
        inf_id = int(all_agent_ids[i][1])
        r_pred = np.argmax(scores[i])
        if r_pred == 1:  # add more filter logics here
            if [inf_id, reactor_id] not in result_to_return:
                result_to_return.append([inf_id, reactor_id])
    # print(f"***** result unpacked for {time_offset} *****")
    return result_to_return


def predict_reactor(all_agent_trajectories, track_type_int, time_offset, raw_data,
                    history_frame_num, future_frame_num, objects_id, args, model, device, threshold=0.5):

    mapping_to_return = []
    for i in range(1, objects_id.shape[0]):
        def swap_1(tensor, index):
            if isinstance(tensor[0], int):
                tensor[index], tensor[0] = tensor[0], tensor[index]
            else:
                tensor[index], tensor[0] = tensor[0].copy(), tensor[index].copy()

        all_agent_trajectories_this_batch = all_agent_trajectories.copy()
        objects_id_this_batch = objects_id.copy()
        # loop all other agents as reactor candidates to detect relationships
        if i == 1:
            reactor_index = 0
        else:
            reactor_index = i
            for each in [all_agent_trajectories_this_batch, objects_id_this_batch]:
                # swap agents (reactor candidates) to index 0
                swap_1(each, i)

        # default without reactor's intentions
        gt_reactor = np.zeros_like(all_agent_trajectories_this_batch[0])
        last_valid_index = history_frame_num - 1
        # speed = utils.get_dis_point2point((gt_trajectory[0, history_frame_num - 1, 5], gt_trajectory[0, history_frame_num - 1, 6]))
        waymo_yaw = all_agent_trajectories_this_batch[0, last_valid_index, 3]
        # headings = gt_trajectory[0, history_frame_num:, 4].copy()
        angle = -waymo_yaw + math.radians(90)
        normalizer = utils.Normalizer(all_agent_trajectories_this_batch[0, last_valid_index, 0],
                                      all_agent_trajectories_this_batch[0, last_valid_index, 1],
                                      angle)
        all_agent_trajectories_this_batch[:, :, :] = utils_cython.get_normalized(all_agent_trajectories_this_batch[:, :, :], normalizer)
        gt_reactor[:, :] = utils_cython.get_normalized(gt_reactor[:, :][np.newaxis, :], normalizer)[0]

        labels = all_agent_trajectories_this_batch[0, history_frame_num:history_frame_num + future_frame_num, :2].copy()

        image = np.zeros([224, 224, 60 + 90], dtype=np.int8)
        args.image = image

        # create some dummies just to check if it works
        gt_future_is_valid = np.ones_like(all_agent_trajectories_this_batch)[:, :, 0]
        trajectory_type = utils_cython.classify_track(gt_future_is_valid[0], all_agent_trajectories_this_batch[0])
        if trajectory_type == 'STATIONARY':
            continue
        tracks_type = np.ones_like(all_agent_trajectories_this_batch)[:, 0, 0]
        vectors, polyline_spans, trajs = utils_cython.get_agents(all_agent_trajectories_this_batch, gt_future_is_valid,
                                                                 tracks_type, False, args, gt_reactor)
        map_start_polyline_idx = len(polyline_spans)

        vectors_, polyline_spans_, goals_2D, lanes = utils_cython.get_roads(raw_data, normalizer, args)
        polyline_spans_ = polyline_spans_ + len(vectors)
        vectors = np.concatenate([vectors, vectors_])
        polyline_spans = np.concatenate([polyline_spans, polyline_spans_])
        polyline_spans = [slice(each[0], each[1]) for each in polyline_spans]

        stage_one_label = np.argmin(
            [utils.get_dis(lane, all_agent_trajectories_this_batch[0, -1, :2]).min() for lane in lanes]) if len(
            lanes) > 0 else 0

        mapping = {
            'matrix': vectors,
            'polyline_spans': polyline_spans,
            'map_start_polyline_idx': map_start_polyline_idx,
            'labels': labels,
            'normalizer': normalizer,
            'goals_2D': goals_2D,
            'polygons': lanes,
            'stage_one_label': stage_one_label,
            'waymo_yaw': waymo_yaw,
            'track_type_int': track_type_int,
            # 'track_type_string': AgentType.to_string(track_type_int),
            'trajectory_type': trajectory_type,
            'tracks_type': tracks_type,
            'eval_time': 80,
            'scenario_id': '001',
            # 'object_id': tf.convert_to_tensor(objects_id)[0],
            'inf_id': objects_id_this_batch[1],
            'all_agent_ids': objects_id_this_batch.copy(),
            # 'inf_label': inf_label,
            'image': args.image
        }
        # if eval_time < 80:
        #     mapping['final_idx'] = eval_time - 1

        final_idx = mapping.get('final_idx', -1)

        mapping['goals_2D_labels'] = np.argmin(utils.get_dis(goals_2D, labels[final_idx]))
        mapping_to_return.append(mapping)

    # print(f"***** model loaded, predicting {len(mapping_to_return)} samples *****")
    if len(mapping_to_return) == 0:
        return None
    scores, all_agent_ids, scenario_ids = model(mapping_to_return, device)
    # print(f"***** model predicted for {time_offset} *****")
    # result_to_return = {}
    # for i, each_mapping in enumerate(mapping_to_return):
    #     reactor_id = all_agent_ids[i][0]
    #     inf_id = all_agent_ids[i][1]
    #     if reactor_id not in result_to_return:
    #         result_to_return[reactor_id] = {}
    #     if time_offset not in result_to_return[reactor_id]:
    #         result_to_return[reactor_id][time_offset] = {
    #             'pred_inf_id': [],
    #             'pred_inf_scores': [],
    #             'pred_inf_label': []
    #         }
    #     result_to_return[reactor_id][time_offset]['pred_inf_id'].append(int(all_agent_ids[i][1]))
    #     result_to_return[reactor_id][time_offset]['pred_inf_scores'].append(scores[i])
    #     result_to_return[reactor_id][time_offset]['pred_inf_label'].append(np.argmax(scores[i]))
    result_to_return = []
    for i, each_mapping in enumerate(mapping_to_return):
        reactor_id = int(all_agent_ids[i][0])
        inf_id = int(all_agent_ids[i][1])
        r_pred = np.argmax(scores[i])
        # print("test 000000: ", scores[i], reactor_id, inf_id)
        if r_pred == 1 and scores[i][r_pred] > threshold:  # add more filter logics here
            if [inf_id, reactor_id] not in result_to_return:
                result_to_return.append([inf_id, reactor_id])
    # print(f"***** result unpacked for {time_offset} *****")
    return result_to_return


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


def predict_reactor_for_onepair(target_reactor_id, all_agent_trajectories, track_type_int, time_offset, raw_data,
                                history_frame_num, future_frame_num, objects_id, args, model, device, threshold=0.5):

    mapping_to_return = []
    for i in range(0, objects_id.shape[0]):
        if objects_id[i] != target_reactor_id:
            continue
        def swap_1(tensor, index):
            if isinstance(tensor[0], int):
                tensor[index], tensor[0] = tensor[0], tensor[index]
            else:
                tensor[index], tensor[0] = tensor[0].copy(), tensor[index].copy()

        all_agent_trajectories_this_batch = all_agent_trajectories.copy()
        objects_id_this_batch = objects_id.copy()
        # loop all other agents as reactor candidates to detect relationships
        if i == 1:
            reactor_index = 0
        else:
            reactor_index = i
            for each in [all_agent_trajectories_this_batch, objects_id_this_batch]:
                # swap agents (reactor candidates) to index 0
                swap_1(each, i)

        # default without reactor's intentions
        gt_reactor = np.zeros_like(all_agent_trajectories_this_batch[0])
        last_valid_index = history_frame_num - 1
        # speed = utils.get_dis_point2point((gt_trajectory[0, history_frame_num - 1, 5], gt_trajectory[0, history_frame_num - 1, 6]))
        waymo_yaw = all_agent_trajectories_this_batch[0, last_valid_index, 3]
        # headings = gt_trajectory[0, history_frame_num:, 4].copy()
        angle = -waymo_yaw + math.radians(90)

        normalizer = utils.Normalizer(all_agent_trajectories_this_batch[0, last_valid_index, 0],
                                      all_agent_trajectories_this_batch[0, last_valid_index, 1],
                                      angle)
        # normalize without Cython



        # print("Tag relation before normalize")
        # all_agent_trajectories_this_batch[:, :, :] = utils_cython.get_normalized(all_agent_trajectories_this_batch[:, :, :], normalizer)
        all_agent_trajectories_this_batch[:, :, :2] = get_normalized(all_agent_trajectories_this_batch[:, :, :2],
                                                                    normalizer.x, normalizer.y, normalizer.yaw)
        # print("Tag relation after normalize1")
        # gt_reactor[:, :] = utils_cython.get_normalized(gt_reactor[:, :][np.newaxis, :], normalizer)[0]
        # print("Tag relation after normalize2")
        labels = all_agent_trajectories_this_batch[0, history_frame_num:history_frame_num + future_frame_num, :2].copy()

        image = np.zeros([224, 224, 60 + 90], dtype=np.int8)
        args.image = image

        # create some dummies just to check if it works
        gt_future_is_valid = np.ones_like(all_agent_trajectories_this_batch)[:, :, 0]
        # print("Tag relation before classify")
        trajectory_type = utils_cython.classify_track(gt_future_is_valid[0], all_agent_trajectories_this_batch[0])
        # print("Tag relation after classify")
        # if trajectory_type == 'STATIONARY':
        #     continue
        tracks_type = np.ones_like(all_agent_trajectories_this_batch)[:, 0, 0]
        # print("Tag relation before get agents")
        vectors, polyline_spans, trajs = utils_cython.get_agents(all_agent_trajectories_this_batch, gt_future_is_valid,
                                                                 tracks_type, False, args, gt_reactor)
        # print("Tag relation after get agents")
        map_start_polyline_idx = len(polyline_spans)
        # print("Tag relation before get roads")
        vectors_, polyline_spans_, goals_2D, lanes = utils_cython.get_roads(raw_data, normalizer, args)
        # print("Tag relation after get roads")
        polyline_spans_ = polyline_spans_ + len(vectors)
        vectors = np.concatenate([vectors, vectors_])
        polyline_spans = np.concatenate([polyline_spans, polyline_spans_])
        polyline_spans = [slice(each[0], each[1]) for each in polyline_spans]

        stage_one_label = np.argmin(
            [utils.get_dis(lane, all_agent_trajectories_this_batch[0, -1, :2]).min() for lane in lanes]) if len(
            lanes) > 0 else 0

        mapping = {
            'matrix': vectors,
            'polyline_spans': polyline_spans,
            'map_start_polyline_idx': map_start_polyline_idx,
            'labels': labels,
            'normalizer': normalizer,
            'goals_2D': goals_2D,
            'polygons': lanes,
            'stage_one_label': stage_one_label,
            'waymo_yaw': waymo_yaw,
            'track_type_int': track_type_int,
            # 'track_type_string': AgentType.to_string(track_type_int),
            'trajectory_type': trajectory_type,
            'tracks_type': tracks_type,
            'eval_time': 80,
            'scenario_id': '001',
            # 'object_id': tf.convert_to_tensor(objects_id)[0],
            'inf_id': objects_id_this_batch[1],
            'all_agent_ids': objects_id_this_batch.copy(),
            # 'inf_label': inf_label,
            'image': args.image
        }
        # if eval_time < 80:
        #     mapping['final_idx'] = eval_time - 1

        final_idx = mapping.get('final_idx', -1)

        mapping['goals_2D_labels'] = np.argmin(utils.get_dis(goals_2D, labels[final_idx]))
        mapping_to_return.append(mapping)

    # print(f"***** model loaded, predicting {len(mapping_to_return)} samples *****")
    if len(mapping_to_return) == 0:
        return None
    scores, all_agent_ids, scenario_ids = model(mapping_to_return, device)
    # print(f"***** model predicted for {time_offset} *****")
    # result_to_return = {}
    # for i, each_mapping in enumerate(mapping_to_return):
    #     reactor_id = all_agent_ids[i][0]
    #     inf_id = all_agent_ids[i][1]
    #     if reactor_id not in result_to_return:
    #         result_to_return[reactor_id] = {}
    #     if time_offset not in result_to_return[reactor_id]:
    #         result_to_return[reactor_id][time_offset] = {
    #             'pred_inf_id': [],
    #             'pred_inf_scores': [],
    #             'pred_inf_label': []
    #         }
    #     result_to_return[reactor_id][time_offset]['pred_inf_id'].append(int(all_agent_ids[i][1]))
    #     result_to_return[reactor_id][time_offset]['pred_inf_scores'].append(scores[i])
    #     result_to_return[reactor_id][time_offset]['pred_inf_label'].append(np.argmax(scores[i]))
    result_to_return = []
    for i, each_mapping in enumerate(mapping_to_return):
        reactor_id = int(all_agent_ids[i][0])
        inf_id = int(all_agent_ids[i][1])
        r_pred = np.argmax(scores[i])
        # print("test 000000: ", scores[i], reactor_id, inf_id)
        if r_pred == 1 and scores[i][r_pred] > threshold:  # add more filter logics here
            if [inf_id, reactor_id] not in result_to_return:
                result_to_return.append([inf_id, reactor_id])
    # print(f"***** result unpacked for {time_offset} *****")
    return result_to_return


def swap(tensor, index):
    if isinstance(tensor[0], int) or isinstance(tensor[0], str):
        tensor[index], tensor[0] = tensor[0], tensor[index]
    else:
        tensor[index], tensor[0] = tensor[0].copy(), tensor[index].copy()
    return tensor

def predict_reactor_for_onepair_NuPlan(target_reactor_id, all_agent_trajectories, track_type_int, time_offset, road_dic,
                                       history_frame_num, future_frame_num, objects_id, args, model, device, threshold=0.5):
    mapping_to_return = []
    for i in range(0, objects_id.shape[0]):
        if objects_id[i] != target_reactor_id:
            continue
        all_agent_trajectories_this_batch = all_agent_trajectories.copy()
        objects_id_this_batch = objects_id.copy()
        # loop all other agents as reactor candidates to detect relationships
        if i == 1:
            reactor_index = 0
            print("ERROR: Predict relation index not arranged before for ", target_reactor_id)
        else:
            reactor_index = i
            # swap agents (reactor candidates) to index 0
            all_agent_trajectories_this_batch = swap(all_agent_trajectories_this_batch, i)
            objects_id_this_batch = swap(objects_id_this_batch, i)


        # default without reactor's intentions
        gt_reactor = np.zeros_like(all_agent_trajectories_this_batch[0])
        last_valid_index = history_frame_num - 1
        # speed = utils.get_dis_point2point((gt_trajectory[0, history_frame_num - 1, 5], gt_trajectory[0, history_frame_num - 1, 6]))
        waymo_yaw = all_agent_trajectories_this_batch[0, last_valid_index, 3]
        # headings = gt_trajectory[0, history_frame_num:, 4].copy()
        angle = -waymo_yaw + math.radians(90)

        normalizer = utils.Normalizer(all_agent_trajectories_this_batch[0, last_valid_index, 0],
                                      all_agent_trajectories_this_batch[0, last_valid_index, 1],
                                      angle)
        # normalize without Cython



        # print("Tag relation before normalize")
        # all_agent_trajectories_this_batch[:, :, :] = utils_cython.get_normalized(all_agent_trajectories_this_batch[:, :, :], normalizer)
        all_agent_trajectories_this_batch[:, :, :2] = get_normalized(all_agent_trajectories_this_batch[:, :, :2],
                                                                     normalizer.x, normalizer.y, normalizer.yaw)
        # print("Tag relation after normalize1")
        # gt_reactor[:, :] = utils_cython.get_normalized(gt_reactor[:, :][np.newaxis, :], normalizer)[0]
        # print("Tag relation after normalize2")
        labels = all_agent_trajectories_this_batch[0, history_frame_num:history_frame_num + future_frame_num, :2].copy()

        image = np.zeros([224, 224, 60 + 90], dtype=np.int8)
        args.image = image

        # create some dummies just to check if it works
        gt_future_is_valid = np.ones_like(all_agent_trajectories_this_batch)[:, :, 0]
        # print("Tag relation before classify")
        trajectory_type = 'STRAIGHT'
        # print("Tag relation after classify")
        # if trajectory_type == 'STATIONARY':
        #     continue
        tracks_type = np.ones_like(all_agent_trajectories_this_batch)[:, 0, 0]
        # print("Tag relation before get agents")
        from prediction.M2I.predictor import vectorize_agent, vectorize_roads
        vectors, polyline_spans, args = vectorize_agent(gt_trajectory=all_agent_trajectories_this_batch,
                                                               gt_future_is_valid=gt_future_is_valid,
                                                               tracks_type=tracks_type,
                                                               args=args)
        # vectors, polyline_spans, trajs = utils_cython.get_agents(all_agent_trajectories_this_batch, gt_future_is_valid,
        #                                                          tracks_type, False, args, gt_reactor)
        # print("Tag relation after get agents")
        map_start_polyline_idx = len(polyline_spans)
        # print("Tag relation before get roads")
        vectors_, polyline_spans_, args, lanes = vectorize_roads(road_dic=road_dic,
                                                                 normalizer=normalizer,
                                                                 args=args)
        # vectors_, polyline_spans_, goals_2D, lanes = utils_cython.get_roads(raw_data, normalizer, args)
        # print("Tag relation after get roads")
        polyline_spans_ = polyline_spans_ + len(vectors)
        vectors = np.concatenate([vectors, vectors_])
        polyline_spans = np.concatenate([polyline_spans, polyline_spans_])
        polyline_spans = [slice(each[0], each[1]) for each in polyline_spans]

        # stage_one_label = np.argmin(
        #     [utils.get_dis(lane, all_agent_trajectories_this_batch[0, -1, :2]).min() for lane in lanes]) if len(
        #     lanes) > 0 else 0
        stage_one_label = 0
        goals_2D = np.zeros([10000, 2])

        mapping = {
            'matrix': vectors,
            'polyline_spans': polyline_spans,
            'map_start_polyline_idx': map_start_polyline_idx,
            'labels': labels,
            'normalizer': normalizer,
            'goals_2D': goals_2D,
            'polygons': lanes,
            'stage_one_label': stage_one_label,
            'waymo_yaw': waymo_yaw,
            'track_type_int': track_type_int,
            # 'track_type_string': AgentType.to_string(track_type_int),
            'trajectory_type': trajectory_type,
            'tracks_type': tracks_type,
            'eval_time': 80,
            'scenario_id': '001',
            # 'object_id': tf.convert_to_tensor(objects_id)[0],
            'inf_id': objects_id_this_batch[1],
            'all_agent_ids': objects_id_this_batch.copy(),
            # 'inf_label': inf_label,
            'image': args.image
        }
        # if eval_time < 80:
        #     mapping['final_idx'] = eval_time - 1

        final_idx = mapping.get('final_idx', -1)

        # mapping['goals_2D_labels'] = np.argmin(utils.get_dis(goals_2D, labels[final_idx]))
        mapping['goals_2D_labels'] = 0
        mapping_to_return.append(mapping)

    # print(f"***** model loaded, predicting {len(mapping_to_return)} samples *****")
    if len(mapping_to_return) == 0:
        return None
    scores, all_agent_ids, scenario_ids = model(mapping_to_return, device)
    # print(f"***** model predicted for {time_offset} *****")
    # result_to_return = {}
    # for i, each_mapping in enumerate(mapping_to_return):
    #     reactor_id = all_agent_ids[i][0]
    #     inf_id = all_agent_ids[i][1]
    #     if reactor_id not in result_to_return:
    #         result_to_return[reactor_id] = {}
    #     if time_offset not in result_to_return[reactor_id]:
    #         result_to_return[reactor_id][time_offset] = {
    #             'pred_inf_id': [],
    #             'pred_inf_scores': [],
    #             'pred_inf_label': []
    #         }
    #     result_to_return[reactor_id][time_offset]['pred_inf_id'].append(int(all_agent_ids[i][1]))
    #     result_to_return[reactor_id][time_offset]['pred_inf_scores'].append(scores[i])
    #     result_to_return[reactor_id][time_offset]['pred_inf_label'].append(np.argmax(scores[i]))
    result_to_return = []
    for i, each_mapping in enumerate(mapping_to_return):
        reactor_id = all_agent_ids[i][0]
        inf_id = all_agent_ids[i][1]
        r_pred = np.argmax(scores[i])
        if r_pred == 1 and scores[i][r_pred] > threshold:  # add more filter logics here
            if [inf_id, reactor_id] not in result_to_return:
                result_to_return.append([inf_id, reactor_id])
    # print(f"***** result unpacked for {time_offset} *****")
    print("test: ", result_to_return)
    return result_to_return


class RelationPredictor:
    def __init__(self):
        self.data = None
        self.threshold = 0.5
        self.model = None
        self.device = None
        self.args = None
        self.model_path = None
        self.predict_device = 'cpu'
        self.max_prediction_num = 128
        self.rank = None

        self.predicting_lock = False

    def __call__(self, **kwargs):
        # init predictor and load model
        if self.device is None:
            predict_device = kwargs['predict_device']
            if predict_device in ['cpu', 'mps']:
                self.device = predict_device
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu", 0)
                self.rank = 0
                print(f"predicting relation with {self.device}")
        if self.args is None:
            utils.args = utils.Args
            self.args = utils.args
            assert 'laneGCN-4' in self.args.other_params and 'raster' in self.args.other_params, self.args.other_params

        if self.model is None:
            if 'model' in kwargs and kwargs['model'] is not None:
                self.model = kwargs['model']
            else:
                self.model_path = kwargs['model_path']['relation_pred']
                self.predict_device = kwargs['predict_device']
                self.load_model()

    # def __call__(self, **kwargs):
    #     self.data = kwargs['new_data']
    #     if self.device is None:
    #         predict_device = kwargs['predict_device']
    #         if predict_device in ['cpu', 'mps']:
    #             self.device = predict_device
    #         else:
    #             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu", 0)
    #             self.rank = 0
    #             print(f"predicting relation with {self.device}")
    #     if self.args is None:
    #         utils.args = utils.Args
    #         self.args = utils.args
    #         assert 'laneGCN-4' in self.args.other_params and 'raster' in self.args.other_params, self.args.other_params
    #
    #     if self.model is None:
    #         if 'model' in kwargs and kwargs['model'] is not None:
    #             self.model = kwargs['model']
    #         else:
    #             self.model_path = kwargs['model_path']['relation_pred']
    #             self.predict_device = kwargs['predict_device']
    #             self.load_model()
        # if self.model is None:
        #     # feed mappings_to_return to the gpus
        #     model = VectorNet(self.args).to(self.device)
        #     model_recover_path = kwargs['model_path']['relation_pred']
        #     model.eval()
        #     torch.no_grad()
        #     # model_recover_path = '/Users/qiaosun/Documents/PyTrafficSim_Git/driving_simulator/prediction/M2I/relation_predictor/pretrained/infPred.bin.IA.v2x.noIntention.1221.model.30.bin'
        #     # optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
        #     print("***** Recover model: %s *****", model_recover_path)
        #     if not torch.cuda.is_available():
        #         model_recover = torch.load(model_recover_path, map_location='cpu')
        #     else:
        #         model_recover = torch.load(model_recover_path)
        #     model.load_state_dict(model_recover)
        #     # model.module.load_state_dict(model_recover)
        #     self.model = model

    def load_model(self):
        print("Start loading relation prediction model")
        # def setup(rank, world_size):
        #     os.environ['MASTER_ADDR'] = 'localhost'
        #     os.environ['MASTER_PORT'] = '12345'
        #     # initialize the process group
        #     dist.init_process_group("nccl", rank=rank, world_size=world_size)
        # feed mappings_to_return to the gpus
        if self.rank is not None:
            # setup(self.rank, 512)
            torch.cuda.set_device(self.rank)
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
        print("Model loaded for relation prediction")


    def predict_one_time(self, current_data, each_pair, current_frame=11, predict_with_rules=True):

        self.predicting_lock = True
        data = copy.deepcopy(current_data)
        future_frame_num = 80
        args = self.args
        device = self.device
        model = self.model

        skip_flag = data['skip']
        agent_dic = data['agent']
        assert not skip_flag

        history_frame_num = 11

        all_agent_trajectories = []
        track_type_int = []
        objects_id = np.array(list(agent_dic.keys()))

        # relevant_agent_ids = self.data['predicting']['relevant_agents']
        # assert len(relevant_agent_ids) > 0
        ego_agent, target_agent = each_pair

        # select, selected_id = self.data['predicting']['ego_id']
        for i, agent_id in enumerate(agent_dic):
            if len(all_agent_trajectories) >= self.max_prediction_num:
                break
            # add all agent pose and type to a numpy array
            pose = agent_dic[agent_id]['pose']
            if isinstance(pose, list):
                short = 91 - len(pose)
                print(f'short in prediction - {short} ; {agent_id}')
                if short > 0:
                    pose += (np.ones([short, 4]) * -1).tolist()
            all_agent_trajectories.append(pose[:91, :])
            track_type_int.append(agent_dic[agent_id]['type'])
        # assert select != -1, 'no interact agent found, failed to select ego in the predictor'
        all_agent_trajectories = np.array(all_agent_trajectories, dtype=np.float32)
        track_type_int = np.array(track_type_int)

        relationship_predicted = []
        # for current_frame in range(0, future_frame_num, time_interval):
        if current_frame > 0:
            all_agent_trajectories = np.concatenate([all_agent_trajectories[:, current_frame:, :],
                                                     np.zeros_like(all_agent_trajectories[:, :current_frame, :])],
                                                    axis=1)

        # WARNING: Adding a non-learning rear collision solver
        def normalize_angle(angle):
            """
            Normalize an angle to [-pi, pi].
            :param angle: (float)
            :return: (float) Angle in radian in [-pi, pi]
            """
            while angle > np.pi:
                angle -= 2.0 * np.pi

            while angle < -np.pi:
                angle += 2.0 * np.pi

            return angle
        same_direction_threshold = 30

        ego_idx = np.where(objects_id == ego_agent)[0]
        target_idx = np.where(objects_id == target_agent)[0]
        ego_yaw = all_agent_trajectories[ego_idx, 0, 3]
        target_yaw = all_agent_trajectories[target_idx, 0, 3]
        yaw_diff = normalize_angle(ego_yaw - target_yaw)

        def get_angle_of_a_line(pt1, pt2):
            # angle from horizon to the right, counter-clockwise,
            x1, y1 = pt1
            x2, y2 = pt2
            angle = math.atan2(y2 - y1, x2 - x1)
            return angle

        if predict_with_rules:
            if -math.pi / 180 * same_direction_threshold < yaw_diff < math.pi / 180 * same_direction_threshold:
                # check rear collision relationship instead of making a prediction
                ego_to_target = get_angle_of_a_line(all_agent_trajectories[ego_idx, 0, :2][0],
                                                    all_agent_trajectories[target_idx, 0, :2][0])
                ego_yaw_diff = normalize_angle(ego_yaw - ego_to_target)
                if -math.pi / 180 * same_direction_threshold < ego_yaw_diff < math.pi / 180 * same_direction_threshold:
                    return [[target_agent, ego_agent]]
                target_to_ego = get_angle_of_a_line(all_agent_trajectories[target_idx, 0, :2][0],
                                                    all_agent_trajectories[ego_idx, 0, :2][0])
                target_yaw_diff = normalize_angle(target_yaw - target_to_ego)
                if -math.pi / 180 * same_direction_threshold < target_yaw_diff < math.pi / 180 * same_direction_threshold:
                    return [[ego_agent, target_agent]]

            # always yield to non-vehicle agents
            if agent_dic[target_agent]['type'] != 1:
                return [[target_agent, ego_agent]]

        objects_id = np.array(objects_id)
        # check forward relation
        select = np.where(objects_id == ego_agent)[0]
        if not isinstance(select, int):
            select = select[0]
        print(f'predicting {ego_agent} at {select} with {each_pair}')
        # swap the ego to index 1
        def swap(tensor):
            if isinstance(tensor[0], int) or isinstance(tensor[0], str):  # id for Waymo is int and id for NuPlan is a string
                tensor[select], tensor[1] = tensor[1], tensor[select]
            else:
                tensor[select], tensor[1] = tensor[1].copy(), tensor[select].copy()
        for each in [all_agent_trajectories, objects_id]:
            swap(each)
        assert objects_id[1] == ego_agent, objects_id

        # not predict
        # return []

        if data['dataset'] == 'NuPlan':
            predicted_relationships = predict_reactor_for_onepair_NuPlan(target_reactor_id=target_agent,
                                                                  all_agent_trajectories=all_agent_trajectories,
                                                                  track_type_int=track_type_int,
                                                                  time_offset=current_frame,
                                                                  history_frame_num=history_frame_num,
                                                                  future_frame_num=future_frame_num,
                                                                  objects_id=objects_id,
                                                                  road_dic=data['road'],
                                                                  args=args,
                                                                  model=model, device=device, threshold=self.threshold)
        elif data['dataset'] == 'Waymo':
            predicted_relationships = predict_reactor_for_onepair(target_reactor_id=target_agent,
                                                                  all_agent_trajectories=all_agent_trajectories,
                                                                  track_type_int=track_type_int,
                                                                  time_offset=current_frame,
                                                                  history_frame_num=history_frame_num,
                                                                  future_frame_num=future_frame_num,
                                                                  objects_id=objects_id,
                                                                  raw_data=data['raw'],
                                                                  args=args,
                                                                  model=model, device=device, threshold=self.threshold)
        else:
            assert False, f'Unknown dataset: '+ str(data['dataset'])

        if predicted_relationships is not None:
            relationship_predicted += predicted_relationships
            # print(f'predicted forward adding {predicted_relationships} to {relationship_predicted}')

        # check backward relation
        select = np.where(objects_id == target_agent)[0]
        if not isinstance(select, int):
            select = select[0]
        # print(f'predicting {target_agent} at {select} with {each_pair}')

        # swap the ego to index 1
        def swap2(tensor):
            if isinstance(tensor[0], int) or isinstance(tensor[0], str):  # id for Waymo is int and id for NuPlan is a string
                tensor[select], tensor[1] = tensor[1], tensor[select]
            else:
                tensor[select], tensor[1] = tensor[1].copy(), tensor[select].copy()

        for each in [all_agent_trajectories, objects_id]:
            swap2(each)
        assert objects_id[1] == target_agent, objects_id

        if data['dataset'] == 'NuPlan':
            predicted_relationships = predict_reactor_for_onepair_NuPlan(target_reactor_id=ego_agent,
                                                                  all_agent_trajectories=all_agent_trajectories,
                                                                  track_type_int=track_type_int,
                                                                  time_offset=current_frame,
                                                                  history_frame_num=history_frame_num,
                                                                  future_frame_num=future_frame_num,
                                                                  objects_id=objects_id,
                                                                  road_dic=data['road'],
                                                                  args=args,
                                                                  model=model, device=device, threshold=self.threshold)
        elif data['dataset'] == 'Waymo':
            predicted_relationships = predict_reactor_for_onepair(target_reactor_id=ego_agent,
                                                                  all_agent_trajectories=all_agent_trajectories,
                                                                  track_type_int=track_type_int,
                                                                  time_offset=current_frame,
                                                                  history_frame_num=history_frame_num,
                                                                  future_frame_num=future_frame_num,
                                                                  objects_id=objects_id,
                                                                  raw_data=data['raw'],
                                                                  args=args,
                                                                  model=model, device=device, threshold=self.threshold)

        if predicted_relationships is not None:
            relationship_predicted += predicted_relationships
        # print(f'predicted backwards adding {predicted_relationships} to {relationship_predicted}')
        torch.cuda.empty_cache()
        self.args.image = None

        self.predicting_lock = False
        return relationship_predicted


    # def predict(self, time_interval=5):
    def predict(self, current_data, current_frame=11):
        data = copy.deepcopy(current_data)
        future_frame_num = 80
        args = self.args
        device = self.device
        model = self.model

        skip_flag = data['skip']
        agent_dic = data['agent']
        assert not skip_flag

        history_frame_num = 11

        all_agent_trajectories = []
        track_type_int = []
        objects_id = np.array(list(agent_dic.keys()))

        select, selected_id = data['predicting']['ego_id']
        for i, agent_id in enumerate(agent_dic):
            # add all agent pose and type to a numpy array
            all_agent_trajectories.append(agent_dic[agent_id]['pose'])
            track_type_int.append(agent_dic[agent_id]['type'])
        assert select != -1, 'no interact agent found, failed to select ego in the predictor'
        all_agent_trajectories = np.array(all_agent_trajectories, dtype=np.float32)
        track_type_int = np.array(track_type_int)

        relationship_predicted = []
        # for current_frame in range(0, future_frame_num, time_interval):
        if current_frame > 0:
            all_agent_trajectories = np.concatenate([all_agent_trajectories[:, current_frame:, :],
                                                     np.zeros_like(all_agent_trajectories[:, :current_frame, :])],
                                                    axis=1)
        iteration_ended = False
        undetected_reactor_pile = [objects_id[select]]
        detected_influencers = []
        while not iteration_ended:
            # predict relationship for each time step
            selected_agent_id = undetected_reactor_pile.pop()
            detected_influencers.append(selected_agent_id)
            select = np.where(objects_id==selected_agent_id)[0]
            # print(f'predicting {selected_agent_id} at {select} with {undetected_reactor_pile}')

            # swap the ego to index 1
            def swap(tensor):
                if isinstance(tensor[0], int):
                    tensor[select], tensor[1] = tensor[1], tensor[select]
                else:
                    tensor[select], tensor[1] = tensor[1].copy(), tensor[select].copy()
            for each in [all_agent_trajectories, objects_id]:
                swap(each)
            assert objects_id[1] == selected_agent_id, objects_id

            predicted_relationships = predict_reactor(all_agent_trajectories=all_agent_trajectories,
                                                      track_type_int=track_type_int,
                                                      time_offset=current_frame,
                                                      history_frame_num=history_frame_num,
                                                      future_frame_num=future_frame_num,
                                                      objects_id=objects_id,
                                                      raw_data=data['raw'],
                                                      args=args,
                                                      model=model, device=device, threshold=self.threshold)
            if len(predicted_relationships) > 0:
                influencer_id = predicted_relationships[0][0]
                for each_pair in predicted_relationships:
                    assert influencer_id == each_pair[0], predicted_relationships
                    reactor_id = each_pair[1]
                    if reactor_id not in undetected_reactor_pile and reactor_id not in detected_influencers:
                        undetected_reactor_pile.append(reactor_id)
                # if current_frame not in relationship_predicted:
                #     relationship_predicted[current_frame] = []
                # relationship_predicted[current_frame] += predicted_relationships
                relationship_predicted += predicted_relationships
                # print(f'predicted adding {predicted_relationships} with {undetected_reactor_pile}')

            if len(undetected_reactor_pile) == 0:
                iteration_ended = True
        # print("test relationships detected on one time_step: ", relationship_predicted)

        torch.cuda.empty_cache()
        self.args.image = None

        return relationship_predicted, detected_influencers


# def main():
#     parser = argparse.ArgumentParser()
#     utils.add_argument(parser)
#     args: utils.Args = parser.parse_args()
#     utils.init(args, logger)
#
#     data = load_data(file_path)
#     run(data, args=args)
#
#
# if __name__ == "__main__":
#     main()
