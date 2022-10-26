import math
import os
import pickle
import random
from functools import partial

import numpy as np
import tensorflow as tf
import torch
from tqdm import tqdm

import globals, structs, utils, utils_cython

tqdm = partial(tqdm, dynamic_ncols=True)

from waymo_tutorial import _parse
from waymo_open_dataset.protos import motion_submission_pb2

from collections import defaultdict

_False = False
if _False:
    import utils_cython

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


def extract_from_inputs(inputs, decoded_example, args, select, idx_in_K):
    sample_is_valid = inputs['sample_is_valid']
    gt_trajectory = tf.boolean_mask(inputs['gt_future_states'], sample_is_valid)
    gt_future_is_valid = tf.boolean_mask(inputs['gt_future_is_valid'], sample_is_valid)
    tracks_to_predict = tf.boolean_mask(inputs['tracks_to_predict'], sample_is_valid)
    tracks_type = tf.boolean_mask(decoded_example['state/type'], sample_is_valid)
    objects_id = tf.boolean_mask(decoded_example['state/id'], sample_is_valid)
    scenario_id = decoded_example['scenario/id'].numpy()[0]

    # For interactive dataset, map select from indices to [0, 1], since interactive agents are not always the first 2 agents.
    if 'train_pair_interest' in args.other_params:
        objects_of_interest = tf.boolean_mask(decoded_example['state/objects_of_interest'], sample_is_valid).numpy()
        indices = np.nonzero(objects_of_interest)[0]
        if select == 0:
            select = indices[0]
        elif select == 1:
            select = indices[1]
        else:
            raise NotImplementedError

    # print(decoded_example['scenario/id'].numpy()[0], type(decoded_example['scenario/id'].numpy()))

    mapping_eval = {
        'gt_trajectory': gt_trajectory[select],
        'gt_is_valid': gt_future_is_valid[select],
        'object_type': tracks_type[select],
        'object_id': objects_id[select],
        'scenario_id': scenario_id,

        'idx_in_predict_num': select,
        'idx_in_K': idx_in_K,
    } if args.do_eval else None

    gt_trajectory = gt_trajectory.numpy().copy()
    gt_future_is_valid = gt_future_is_valid.numpy().copy()
    tracks_to_predict = tracks_to_predict.numpy().copy()
    tracks_type = tracks_type.numpy().copy().reshape(-1)
    objects_id = objects_id.numpy().copy()
    sample_is_valid = sample_is_valid.numpy().copy()

    predict_agent_num = tracks_to_predict.sum()
    for i in range(predict_agent_num):
        assert tracks_to_predict[i]
    assert len(gt_trajectory) == len(gt_future_is_valid) == len(tracks_type)

    return sample_is_valid, gt_trajectory, gt_future_is_valid, tracks_to_predict, tracks_type, objects_id, scenario_id, mapping_eval




def load_scenario_from_dictionary(dictionary_to_load, scenario_id):
    if bytes.decode(scenario_id) in dictionary_to_load.keys():
        return dictionary_to_load[bytes.decode(scenario_id)]
    elif scenario_id in dictionary_to_load.keys():
        return dictionary_to_load[scenario_id]
    else:
        # print(f"scenario {scenario_id} or {bytes.decode(scenario_id)} not found, key sample: {list(dictionary_to_load.keys())[0]}")
        return None


def get_instance(args: utils.Args, inputs, decoded_example, file_name,
                 select=None, time_offset=None, idx_in_K=None):
    sample_is_valid, gt_trajectory, gt_future_is_valid, tracks_to_predict, tracks_type, objects_id, scenario_id, mapping_eval = \
        extract_from_inputs(inputs, decoded_example, args, select, idx_in_K)

    # For interactive dataset, map select from indices to [0, 1], since interactive agents are not always the first 2 agents.
    if args.agent_selection[0] == 'I':
        objects_of_interest = tf.boolean_mask(decoded_example['state/objects_of_interest'], sample_is_valid).numpy()
        indices = np.nonzero(objects_of_interest)[0]
        select = indices[select]
        predict_agent_num = 2
    elif args.agent_selection[0] == 'P':
        indices = np.nonzero(tracks_to_predict)[0]
        select = indices[select]
        predict_agent_num = tracks_to_predict.sum()
    elif args.agent_selection[0] == 'S':
        # only select sdv as the reactor
        sdc = tf.boolean_mask(decoded_example['state/is_sdc'], sample_is_valid).numpy()
        indices = np.nonzero(sdc)[0]
        select = indices[0]
        predict_agent_num = 1
    elif args.agent_selection[0] == 'A':
        # already select all valid samples at extract_from_inputs
        predict_agent_num = sample_is_valid.sum()
    else:
        print(f'Unknown agent selection mode: {args.agent_slection}')
        raise NotImplementedError

    # check target agent type
    if not type_is_ok(tracks_type[select], args):
        # print("test type return None")
        return None

    mapping_eval = {
        'gt_trajectory': gt_trajectory[select],
        'gt_is_valid': gt_future_is_valid[select],
        'object_type': tracks_type[select],
        'object_id': objects_id[select],
        'scenario_id': scenario_id,

        'idx_in_predict_num': select,
        'idx_in_K': idx_in_K,
    } if args.do_eval else None

    # predict_agent_num = tracks_to_predict.sum()
    # use future_frame_num as default value for eval_time
    eval_time = args.other_params.get('eval_time', args.future_frame_num)
    history_frame_num = 11
    whole_final_idx_eval = -1 if eval_time == 80 else history_frame_num + eval_time - 1
    whole_final_idx_training = -1 if args.future_frame_num == 80 else history_frame_num + args.future_frame_num - 1

    if time_offset is not None:
        if time_offset > 0:
            gt_trajectory = np.concatenate([gt_trajectory[:, time_offset:, :], gt_trajectory[:, :time_offset, :]], axis=1)
            gt_future_is_valid = np.concatenate([gt_future_is_valid[:, time_offset:], gt_future_is_valid[:, :time_offset]], axis=1)
        mapping_eval['time_offset'] = time_offset

    ################## Loading files #################

    if args.direct_relation_path is not None:
        # this is the new direct type of relation, a list of [influencer_id, reactor_id]
        # sample:  [[25.0, 1.0], [1.0, 2.0], [1.0, 4.0], [2.0, 8.0], [1.0, 9.0]]
        # new sample:  [[25.0, 1.0, 1], [1.0, 2.0, 2], [1.0, 4.0, 0], [2.0, 8.0, 2], [1.0, 9.0, 0]]
        if globals.direct_relation is None:
            print("loading direct relation from: ", args.direct_relation_path)
            globals.direct_relation = structs.load(args.direct_relation_path)
            print("loading direct relation finished")

    if args.influencer_pred_file_path is not None:
        if globals.influencer_pred is None:
            print("loading trajectory prediction from: ", args.influencer_pred_file_path)
            globals.influencer_pred = structs.load(args.influencer_pred_file_path)
            print("pd trajectory loaded")
        # replace gt with marginal predictions
        influencer_pred = globals.influencer_pred
        loaded_inf = load_scenario_from_dictionary(influencer_pred, scenario_id)
        if loaded_inf is not None:
            prediction_result = loaded_inf['rst']
            agents_ids_in_prediction = loaded_inf['ids']
            prediction_scores = loaded_inf['score']
        else:
            # print(f'{scenario_id} not found in path prediction file, {tracks_type[0]}')
            return None
        loading_summary['scenarios_in_traj_pred'] += 1

    ################## End of files loading #################

    # preparing trajectories for encoders of different tasks
    if 'detect_all_inf' in args.other_params:
        if 'inf_pred_k' in args.other_params:
            if np.sum(sample_is_valid.numpy().copy()) < 4:
                # skip the scenario which has valid agents fewer than 4 to predict
                return None
        # load the influencers of the current agent
        selected_agent_id = int(objects_id[select])

        # swap current agent to index 0
        def swap(tensor):
            tensor[select], tensor[0] = tensor[0].copy(), tensor[select].copy()
        for each in [gt_trajectory, gt_future_is_valid, tracks_type, objects_id]:
            swap(each)

        if args.do_eval or args.do_test:
            inf_label = 0
            pass
        elif args.do_train:
            # the relation label is binary and is a list of 0 and 1, 1 means is an influencer, 0 is not
            # get current_agent's influencer, if there is any
            assert globals.direct_relation is not None, f'pass direct relation file path to use'
            direct_relation = load_scenario_from_dictionary(globals.direct_relation, scenario_id)
            if direct_relation is None:
                # skip not found scenarios
                return None
            direct_relation = np.array(direct_relation)
            if len(direct_relation.shape) == 1:
                direct_relation = direct_relation[np.newaxis, :]

            influencer_indices = []
            non_influencer_indices = []
            influencer_ids = []
            has_found = False
            # find the influencers of current agent
            for influencer_id, reactor_id, relation_label in direct_relation:
                if reactor_id == selected_agent_id and relation_label != 2:
                    has_found = True
                    for idx, agent_id in enumerate(objects_id):
                        # following "PA" selection rule
                        if int(agent_id) == influencer_id:
                            assert idx != 0, direct_relation
                            if 'pair_vv' in args.other_params:
                                if tracks_type[idx] != AgentType['vehicle']:
                                    break
                            if 'filter_all_valid_inf' in args.other_params:
                                if min(gt_future_is_valid[idx][11:].astype(int)) == 0:
                                # if min(gt_future_is_valid[idx][11:]) == 0:
                                    break

                            influencer_indices.append(idx)
                            influencer_ids.append(int(influencer_id))
                        else:
                            non_influencer_indices.append(idx)

            if 'no_direct_edge' in args.other_params:
                # find the reactors of current agent
                for influencer_id, reactor_id, relation_label in direct_relation:
                    if influencer_id == selected_agent_id and relation_label != 2:
                        has_found = True
                        for idx, agent_id in enumerate(objects_id):
                            if int(agent_id) == reactor_id:
                                assert idx != 0, direct_relation
                                if 'pair_vv' in args.other_params:
                                    if tracks_type[idx] != AgentType['vehicle']:
                                        break
                                if 'filter_all_valid_inf' in args.other_params:
                                    if min(gt_future_is_valid[idx][11:].astype(int)) == 0:
                                        # if min(gt_future_is_valid[idx][11:]) == 0:
                                        break
                                # squash them all in the influencer list
                                influencer_indices.append(idx)
                                influencer_ids.append(int(influencer_id))
                            else:
                                non_influencer_indices.append(idx)

            if not has_found:
                return None
            if len(influencer_indices) == 0:
                target_inf_index = random.randint(0, objects_id.shape[0] - 1)
                inf_label = 0
                globals.loading_summary['label_0_loaded'] += 1
            elif random.random() < 0.5:
                target_inf_index = non_influencer_indices[random.randint(0, len(non_influencer_indices) - 1)]
                inf_label = 0
                globals.loading_summary['label_0_loaded'] += 1
            else:
                target_inf_index = influencer_indices[random.randint(0, len(influencer_indices) - 1)]
                inf_label = 1
                if tracks_to_predict[target_inf_index]:
                    globals.loading_summary['label_1_to_predict'] += 1
                globals.loading_summary['label_1_loaded'] += 1

            # swap current agent to index 0
            def swap_1(tensor):
                tensor[target_inf_index], tensor[1] = tensor[1].copy(), tensor[target_inf_index].copy()

            for each in [gt_trajectory, gt_future_is_valid, tracks_type, objects_id]:
                swap_1(each)

        if 'no_reactor_intention' in args.other_params:
            gt_reactor = np.zeros_like(gt_trajectory[0])
        else:
            # get future trajectory of the reactor
            gt_reactor = gt_trajectory[0].copy()
            # to keep input the same, only include yaw infos before frame 11
            gt_reactor[11:, 4:] = 0
            if args.eval_rst_saving_number is None:
                # use gt_traj for training: when training, do not give eval_rst_saving_number, leave gt_traj as it is
                pass
            else:
                assert args.marginal_pred_file_path is not None
                # load prediction result for the reactor
                assert len(prediction_result.shape) in [3, 4], prediction_result.shape
                if len(prediction_result.shape) == 3:
                    assert prediction_result.shape == (6, 80, 2), prediction_result.shape
                    print("Your marginal prediction file has only one agent in it, shape: (6, 80, 2)")
                num_of_agents_in_prediction, _, _, _ = prediction_result.shape
                prediction_result_reactor = None
                for i in range(num_of_agents_in_prediction):
                    if agents_ids_in_prediction[i] == selected_agent_id:
                        prediction_result_reactor = prediction_result[i]
                        prediction_scores_reactor = prediction_scores[i]
                assert prediction_result_reactor is not None, f'{selected_agent_id} not found in {agents_ids_in_prediction} at {scenario_id}'
                # assert prediction_result_reactor.shape == (6, 80, 2), prediction_result_reactor.shape
                assert prediction_result_reactor.shape[0] == 6 and prediction_result_reactor.shape[
                    2] == 2, prediction_result_reactor.shape
                assert prediction_scores_reactor.shape == (6,), prediction_scores.shape

                # use one of the prediction for evaluation, send eval_rst_saving_number
                target_idx = int(args.eval_rst_saving_number)
                gt_reactor[:, :2] = np.concatenate(
                    [gt_trajectory[0, :11, :2], prediction_result_reactor[target_idx]])
                gt_reactor[:11, :] = gt_trajectory[0, :11, :].copy()

    select = None

    if not gt_future_is_valid[0, history_frame_num - 1]:
        return None
    if args.do_train:
        # also using the goal point as a guild for inf prediction
        if not gt_future_is_valid[0, - 1]:
            return None

    last_valid_index = history_frame_num - 1

    speed = utils.get_dis_point2point((gt_trajectory[0, history_frame_num - 1, 5], gt_trajectory[0, history_frame_num - 1, 6]))
    waymo_yaw = gt_trajectory[0, last_valid_index, 4]
    track_type_int = tracks_type[0]
    trajectory_type = utils_cython.classify_track(gt_future_is_valid[0], gt_trajectory[0])
    headings = gt_trajectory[0, history_frame_num:, 4].copy()

    angle = -waymo_yaw + math.radians(90)

    normalizer = utils.Normalizer(gt_trajectory[0, last_valid_index, 0], gt_trajectory[0, last_valid_index, 1], angle)

    if 'detect_all_inf' in args.other_params:
        gt_trajectory[:, :, :] = utils_cython.get_normalized(gt_trajectory[:, :, :], normalizer)
        gt_reactor[:, :] = utils_cython.get_normalized(gt_reactor[:, :][np.newaxis, :], normalizer)[0]

    labels = gt_trajectory[0, history_frame_num:history_frame_num + args.future_frame_num, :2].copy() * \
             gt_future_is_valid[0, history_frame_num:history_frame_num + args.future_frame_num, np.newaxis]

    # More about the yaw angel from WOD: The yaw angle in radians of the forward direction of the bounding box
    # (the vector from the center of the box to the middle of the front box segment) counter clockwise
    # from the X-axis (right hand system about the Z axis).
    yaw_labels = gt_trajectory[0, history_frame_num:history_frame_num + args.future_frame_num, 4].copy() * \
                 gt_future_is_valid[0, history_frame_num:history_frame_num + args.future_frame_num]


    labels_is_valid = gt_future_is_valid[0, history_frame_num:history_frame_num + args.future_frame_num].copy()

    if 'raster' in args.other_params:
        if 'detect_all_inf' in args.other_params:
            image = np.zeros([224, 224, 60 + 90], dtype=np.int8)
        else:
            assert False
        args.image = image

    if args.do_train:
        vectors, polyline_spans, trajs = utils_cython.get_agents(gt_trajectory, gt_future_is_valid, tracks_type,
                                                                 args.visualize, args, gt_reactor)
        map_start_polyline_idx = len(polyline_spans)
        vectors_, polyline_spans_, goals_2D, lanes = utils_cython.get_roads(decoded_example, normalizer, args)
        polyline_spans_ = polyline_spans_ + len(vectors)
        vectors = np.concatenate([vectors, vectors_])
        polyline_spans = np.concatenate([polyline_spans, polyline_spans_])
        polyline_spans = [slice(each[0], each[1]) for each in polyline_spans]
        if len(lanes) == 0:
            if args.do_eval:
                pass
            else:
                assert False

        stage_one_label = np.argmin([utils.get_dis(lane, gt_trajectory[0, -1, :2]).min() for lane in lanes]) if len(lanes) > 0 else 0
        mapping = {
            'matrix': vectors,
            'polyline_spans': polyline_spans,
            'map_start_polyline_idx': map_start_polyline_idx,
            'labels': labels,
            'labels_is_valid': labels_is_valid,
            # 'predict_agent_num': predict_agent_num,
            'normalizer': normalizer,
            'goals_2D': goals_2D,
            'polygons': lanes,
            'stage_one_label': stage_one_label,
            'waymo_yaw': waymo_yaw,
            'speed': speed,
            'headings': headings,
            'track_type_int': track_type_int,
            'track_type_string': AgentType.to_string(track_type_int),
            'trajectory_type': trajectory_type,
            'tracks_type': tracks_type,
            'file_name': file_name,
            'instance_id': (scenario_id, objects_id[0]),
            'eval_time': eval_time,

            'yaw_labels': yaw_labels,

            'scenario_id': scenario_id,
            'object_id': tf.convert_to_tensor(objects_id)[0],

            'all_agent_ids': objects_id,
        }
        if 'detect_all_inf' in args.other_params:
            # mapping['influencer_idx'] = inf_labels
            mapping['inf_label'] = inf_label

        if eval_time < 80:
            mapping['final_idx'] = eval_time - 1

        if args.visualize:
            mapping.update({
                'trajs': trajs,
                'vis_lanes': lanes,
            })

        if 'raster' in args.other_params:
            mapping['image'] = args.image

        final_idx = mapping.get('final_idx', -1)
        mapping['goals_2D_labels'] = np.argmin(utils.get_dis(goals_2D, labels[final_idx]))

        if args.do_eval:
            mapping.update(mapping_eval)

        return mapping

    elif args.do_eval or args.do_test:
        # swap current agent to index 0
        def swap_1(tensor, index):
            tensor[index], tensor[1] = tensor[1].copy(), tensor[index].copy()

        mappings_to_return = []
        for i in range(1, objects_id.shape[0]):
            target_inf_id = objects_id[i]
            for each in [gt_trajectory, gt_future_is_valid, tracks_type, objects_id]:
                swap_1(each, i)
            assert objects_id[1] == target_inf_id
            trajectory_type = utils_cython.classify_track(gt_future_is_valid[i], gt_trajectory[i])
            if trajectory_type == 'STATIONARY':
                continue
            vectors, polyline_spans, trajs = utils_cython.get_agents(gt_trajectory, gt_future_is_valid, tracks_type,
                                                                     args.visualize, args, gt_reactor)
            map_start_polyline_idx = len(polyline_spans)
            vectors_, polyline_spans_, goals_2D, lanes = utils_cython.get_roads(decoded_example, normalizer, args)
            polyline_spans_ = polyline_spans_ + len(vectors)
            vectors = np.concatenate([vectors, vectors_])
            polyline_spans = np.concatenate([polyline_spans, polyline_spans_])
            polyline_spans = [slice(each[0], each[1]) for each in polyline_spans]
            if len(lanes) == 0:
                if args.do_eval:
                    pass
                else:
                    assert False

            stage_one_label = np.argmin([utils.get_dis(lane, gt_trajectory[0, -1, :2]).min() for lane in lanes]) if len(
                lanes) > 0 else 0

            mapping = {
                'matrix': vectors,
                'polyline_spans': polyline_spans,
                'map_start_polyline_idx': map_start_polyline_idx,
                'labels': labels,
                'labels_is_valid': labels_is_valid,
                # 'predict_agent_num': predict_agent_num,
                'normalizer': normalizer,
                'goals_2D': goals_2D,
                'polygons': lanes,
                'stage_one_label': stage_one_label,
                'waymo_yaw': waymo_yaw,
                'speed': speed,
                'headings': headings,
                'track_type_int': track_type_int,
                'track_type_string': AgentType.to_string(track_type_int),
                'trajectory_type': trajectory_type,
                'tracks_type': tracks_type,
                'file_name': file_name,
                'instance_id': (scenario_id, objects_id[0]),
                'eval_time': eval_time,

                'yaw_labels': yaw_labels,

                'scenario_id': scenario_id,
                'object_id': tf.convert_to_tensor(objects_id)[0],
                'inf_id': objects_id[1],
                'all_agent_ids': objects_id.copy(),
            }

            if 'detect_all_inf' in args.other_params:
                # mapping['influencer_idx'] = inf_labels
                mapping['inf_label'] = inf_label

            if eval_time < 80:
                mapping['final_idx'] = eval_time - 1

            if args.visualize:
                mapping.update({
                    'trajs': trajs,
                    'vis_lanes': lanes,
                })

            if 'raster' in args.other_params:
                mapping['image'] = args.image

            final_idx = mapping.get('final_idx', -1)
            mapping['goals_2D_labels'] = np.argmin(utils.get_dis(goals_2D, labels[final_idx]))

            if args.do_eval:
                mapping.update(mapping_eval)

            mappings_to_return.append(mapping)

        return mappings_to_return


speed_data_ids = {}


def type_is_ok(type, args):
    return args.agent_type is None or type == AgentType[args.agent_type]


def types_are_ok(types, args):
    return args.inter_agent_types is None or \
           types[0] == AgentType[args.inter_agent_types[0]] and types[1] == AgentType[args.inter_agent_types[1]]


def get_ex_list_from_file(file_name, args: utils.Args, trajectory_type_2_ex_list=None, balance_queue=None):

    if 'detect_all_inf' in args.other_params:
        ex_list = []
        dataset = tf.data.TFRecordDataset(file_name)
        for step, data in enumerate(dataset):
            inputs, decoded_example = _parse(data)

            # get predict_agent_num
            sample_is_valid = inputs['sample_is_valid']
            tracks_to_predict = tf.boolean_mask(inputs['tracks_to_predict'], sample_is_valid)
            # Set predicting tracks to interactive tracks if train_pair_interest flag is specified.
            if args.agent_selection[0] == 'I':
                interactive_tracks_to_predict = tf.boolean_mask(inputs['interactive_tracks_to_predict'],
                                                                sample_is_valid)
                predict_agent_num = interactive_tracks_to_predict.numpy().sum()
            elif args.agent_selection[0] == 'P':
                predict_agent_num = tracks_to_predict.numpy().sum()
            elif args.agent_selection[0] == 'A':
                predict_agent_num = sample_is_valid.numpy().sum()
            elif args.agent_selection[0] == 'S':
                predict_agent_num = 1
            else:
                assert False, f'unknown agent selection type {args.agent_selection}'
            # tracks_type = tf.boolean_mask(decoded_example['state/type'], sample_is_valid)
            # tracks_type = tracks_type.numpy().copy().reshape(-1)

            # load instance, eval and training share the same loading process
            # if 'train_pair_interest' in args.other_params:
            #     assert False, 'Deperacated, use agent selection mode to control agents loaded'
            if args.do_eval or args.do_test:
                for select in range(predict_agent_num):
                    # several possibilities, return one none, return a list of mappings
                    instance = get_instance(args, inputs, decoded_example,
                                            f'{os.path.split(file_name)[1]}.{str(step)}', select=select,
                                            time_offset=args.frame_offset)
                    if instance is not None and len(instance) > 0:
                        assert None not in instance
                        assert isinstance(instance[0], type({})), instance[0]
                        max_size_per_batch = 8
                        current_len = len(instance) - max_size_per_batch
                        starting_index = max_size_per_batch
                        ex_list.append(instance[:max_size_per_batch])
                        while current_len > 0:
                            ex_list.append(instance[starting_index:starting_index+max_size_per_batch])
                            starting_index += max_size_per_batch
                            current_len -= max_size_per_batch

                # instance = [get_instance(args, inputs, decoded_example,
                #                          f'{os.path.split(file_name)[1]}.{str(step)}',
                #                          select=select) for select in range(predict_agent_num)]
                # instance = [per_ins for per_ins in instance if per_ins is not None]
                # if instance is not None and len(instance) > 0:
                #     ex_list.append(instance)
            else:
                for select in range(predict_agent_num):
                    instance = get_instance(args, inputs, decoded_example,
                                            f'{os.path.split(file_name)[1]}.{str(step)}', select=select)

                    if instance is not None and len(instance) > 0 and None not in instance:
                        ex_list.append(instance)

        return ex_list

    else:
        assert False



class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, batch_size, rank=0, to_screen=True):
        # self.loader = WaymoDL(args.data_dir[0])
        self.args = args
        self.rank = rank
        self.world_size = args.distributed_training if args.distributed_training else 1
        self.batch_size = batch_size

        tf_example_dir = args.data_dir[0]
        self.file_names = [os.path.join(tf_example_dir, f) for f in os.listdir(tf_example_dir) if
                           os.path.isfile(os.path.join(tf_example_dir, f))]
        self.file_names = sorted(self.file_names)
        if args.starting_file_num is not None and args.ending_file_num is not None:
            self.file_names = self.file_names[args.starting_file_num: args.ending_file_num]

        if to_screen:
            print("valid file_names is", len(self.file_names))

        if args.do_eval:
            self.load_queue = np.arange(len(self.file_names))
            self.load_queue = self.load_queue[self.rank::self.world_size]
            self.load_queue = iter(self.load_queue)
            self.waymo_generate(expected_len=200)
        else:
            self.set_epoch(0)

        self.batch_size = batch_size

    def __len__(self):
        args = self.args
        if self.args.do_eval:
            return int(500_000 * 0.15 * 2 / self.world_size)
        else:
            return int(500_000 * 0.7 / self.batch_size)

    def __getitem__(self, idx):
        # print('__getitem__', idx)
        return self.__next__()

    def __next__(self):
        if isinstance(self.ex_list, list):
            self.ex_list = iter(self.ex_list)

        if self.args.do_eval:
            try:
                return next(self.ex_list)
            except StopIteration:
                return None
        else:
            mapping = []
            for i in range(self.batch_size // self.world_size):
                try:
                    mapping.append(next(self.ex_list))
                except StopIteration:
                    return None

            return mapping

    def set_epoch(self, i_epoch):
        if i_epoch == 0:
            if hasattr(self, 'load_queue'):
                return
        self.load_queue = np.arange(len(self.file_names))
        np.random.seed(i_epoch)
        np.random.shuffle(self.load_queue)
        utils.logging('set_train_epoch', self.load_queue[:20])
        self.load_queue = self.load_queue[self.rank::self.world_size]
        self.load_queue = iter(self.load_queue)
        self.waymo_generate(expected_len=200)
        self.set_ex_list_length(200)

    def set_ex_list_length(self, length):
        if self.args.do_train:
            random.shuffle(self.ex_list)
            if not self.args.debug_mode:
                assert len(self.ex_list) >= length, str(len(self.ex_list)) + '/' + str(length)
            self.ex_list = self.ex_list[:length]

    def waymo_generate(self, expected_len=500 * 20):
        self.ex_list = []

        args = self.args

        if args.do_eval:
            expected_len = 200

        if 'raster' in args.other_params and args.do_train:
            if expected_len > 500 * 10:
                expected_len = expected_len // 2

        assert expected_len >= self.batch_size

        while len(self.ex_list) < expected_len:
            try:
                file_name = self.file_names[next(self.load_queue)]
            except StopIteration:
                return False, len(self.ex_list)

            self.ex_list.extend(get_ex_list_from_file(file_name, args))

        random.shuffle(self.ex_list)

        return True, len(self.ex_list)


def generate_protobuf(output_dir, file_name, other_params, *lists):
    submission = motion_submission_pb2.MotionChallengeSubmission()
    submission.account_name = 'gujunru123@gmail.com'
    submission.unique_method_name = 'Anonymous610'
    MOTION_PREDICTION = True

    waymo_pred = structs.WaymoPred()

    for prediction_trajectory, prediction_score, \
        ground_truth_trajectory, ground_truth_is_valid, object_type, scenario_id, object_id in zip(*lists):
        ground_truth_trajectory = None
        ground_truth_is_valid = None
        object_type = None

        if isinstance(prediction_trajectory, np.ndarray):
            prediction_trajectory = tf.convert_to_tensor(prediction_trajectory)
        if isinstance(prediction_score, np.ndarray):
            prediction_score = tf.convert_to_tensor(prediction_score)

        predict_num = len(prediction_trajectory)
        # prediction_set = motion_submission_pb2.PredictionSet()

        scenario_prediction = submission.scenario_predictions.add()
        scenario_prediction.scenario_id = scenario_id

        if len(prediction_trajectory.shape) == 5:
            MOTION_PREDICTION = False
            joint_prediction = scenario_prediction.joint_prediction
            assert predict_num == 1
            for i in range(predict_num):

                for k in range(6):
                    ScoredJointTrajectory = joint_prediction.joint_trajectories.add()
                    ScoredJointTrajectory.confidence = prediction_score[i, k]
                    assert prediction_trajectory.shape[2] == 2
                    for c in range(2):
                        ObjectTrajectory = ScoredJointTrajectory.trajectories.add()
                        ObjectTrajectory.object_id = object_id[c]
                        Trajectory = ObjectTrajectory.trajectory

                        interval = 5
                        traj = prediction_trajectory[i, k, c, (interval - 1)::interval, :]
                        Trajectory.center_x[:] = traj[:, 0].numpy().tolist()
                        Trajectory.center_y[:] = traj[:, 1].numpy().tolist()
            pass
        else:
            prediction_set = scenario_prediction.single_predictions
            for i in range(predict_num):
                # SingleObjectPrediction
                prediction = prediction_set.predictions.add()
                prediction.object_id = object_id[i]
                obj = structs.MultiScoredTrajectory(prediction_score[i, :].numpy(), prediction_trajectory[i, :, :, :].numpy())
                waymo_pred[(scenario_id, object_id.numpy()[i])] = obj

                for k in range(6):
                    # ScoredTrajectory
                    scored_trajectory = prediction.trajectories.add()
                    scored_trajectory.confidence = prediction_score[i, k]
                    trajectory = scored_trajectory.trajectory

                    if prediction_trajectory.shape[2] == 16:
                        traj = prediction_trajectory[i, k, :, :]
                    else:
                        assert prediction_trajectory.shape[2] == 80, prediction_trajectory.shape
                        interval = 5
                        traj = prediction_trajectory[i, k, (interval - 1)::interval, :]

                    trajectory.center_x[:] = traj[:, 0].numpy().tolist()
                    trajectory.center_y[:] = traj[:, 1].numpy().tolist()

    if 'std_on_inter' in other_params:
        structs.save(waymo_pred, output_dir, utils.get_eval_identifier(), prefix='std_on_inter')

    if 'out_pred' in other_params:
        structs.save(waymo_pred, output_dir, utils.get_eval_identifier())

    if MOTION_PREDICTION:
        submission.submission_type = motion_submission_pb2.MotionChallengeSubmission.SubmissionType.MOTION_PREDICTION
    else:
        submission.submission_type = motion_submission_pb2.MotionChallengeSubmission.SubmissionType.INTERACTION_PREDICTION

    if 'out_submit' in other_params:
        path = os.path.join(output_dir, file_name)
        with open(path, "wb") as f:
            f.write(submission.SerializeToString())

        os.system(f'tar -zcvf {path}.tar.gz {path}')
        os.system(f'rm {path}')


def filter_other_agent(args, batch, pack):
    (gt_trajectory, gt_is_valid, object_type, scenario_id, object_id) = pack
    predict_agent_num = batch[0]['predict_agent_num']
    new_batch = []
    assert predict_agent_num == len(batch)
    tracks_to_predict = np.ones(predict_agent_num, dtype=np.bool)
    for i in range(predict_agent_num):
        if args.agent_type is None \
                or AgentType[args.agent_type] == batch[i]['track_type_int']:
            new_batch.append(batch[i])
        else:
            tracks_to_predict[i] = False

    tracks_to_predict = tf.convert_to_tensor(tracks_to_predict)
    batch = new_batch
    gt_trajectory = tf.boolean_mask(gt_trajectory, tracks_to_predict)
    gt_is_valid = tf.boolean_mask(gt_is_valid, tracks_to_predict)
    object_type = tf.boolean_mask(object_type, tracks_to_predict)
    scenario_id = scenario_id
    object_id = tf.boolean_mask(object_id, tracks_to_predict)

    return batch, (gt_trajectory, gt_is_valid, object_type, scenario_id, object_id)
