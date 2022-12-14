import json
import os
from json import JSONEncoder

import pickle
import numpy as np

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def load_data_from_pickle(loading_file_name=None):
    if loading_file_name is None:
        with open('interactive_relations_non_relation_type.pickle', 'rb') as f:
            return pickle.load(f)
    else:
        with open(loading_file_name, 'rb') as f:
            return pickle.load(f)

def convert_type(obj):
    if isinstance(obj, dict):
        return convert_dic_with_np(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return convert_list_with_np(obj)
    elif isinstance(obj, bool) or isinstance(obj, int) or isinstance(obj, float) or isinstance(obj, str) or obj is None:
        return obj
    else:
        try:
            return float(obj)
        except:
            # print("unknown type: ", obj, type(obj))
            return None

def convert_list_with_np(a_list):
    list_to_return = []
    for each_obj in a_list:
        list_to_return.append(convert_type(each_obj))
    return list_to_return

def convert_dic_with_np(dic):
    dic_copy = {}
    for each_key in dic:
        if isinstance(each_key, bool) or isinstance(each_key, int) or isinstance(each_key, float) or isinstance(each_key, str):
            dic_copy[each_key] = convert_type(dic[each_key])
        elif isinstance(each_key, np.int32) or isinstance(each_key, np.int64):
            dic_copy[int(each_key)] = convert_type(dic[each_key])
        else:
            try:
                new_key = float(each_key)
                dic_copy[new_key] = convert_type(dic[each_key])
            except:
                print("unknown type dic key:", each_key, dic[each_key])
    return dic_copy

def convert_one_scenario(data_dic):
    for each_key in data_dic:
        if each_key == 'predicting':
            sub_dic = data_dic[each_key]
            # del sub_dic['trajectory_to_mark']
            data_dic[each_key] = convert_type(sub_dic)
        else:
            data_dic[each_key] = convert_type(data_dic[each_key])
    return data_dic


def numpy_to_list(dic, dataset='Waymo'):
    # two level
    print(f'Total Scenarios: {len(list(dic.keys()))}')
    counter = 0
    dic_to_return = {}
    for each_scenario_id in dic:
        counter += 1
        # if counter > 100:
        #     break
        data_dic = dic[each_scenario_id]
        # if 'road' in data_dic and dataset != 'Waymo':
        #     del data_dic['road']

        # data_dic = {'agent': data_dic['agent']}

        # ['agent', 'traffic_light', 'category', 'scenario', 'edges', 'skip', 'edge_type', 'route', 'type', 'goal',
        #  'predicting', 'planner_timer', 'predict_timer']
        keys = ['road', 'agent', 'traffic_light', 'predicting']
        print(f'Processsing: {each_scenario_id}')
        try:
            each_scenario_id = each_scenario_id.decode()
        except (UnicodeDecodeError, AttributeError):
            pass
        dic_to_return[each_scenario_id] = convert_one_scenario(data_dic)
    return dic_to_return



# simulation_path = 'training_data/08251233'
# playback_path = os.path.join(simulation_path, 'playback')
# all_pickles = [[each_path, os.path.join(playback_path, each_path)] for each_path in os.listdir(playback_path)]
# json_dir = os.path.join(simulation_path, 'json')
# if not os.path.isdir(json_dir):
#     os.makedirs(json_dir)
# for file_name, each_path in all_pickles:
#     rst = load_data_from_pickle(each_path)
#     rst = numpy_to_list(rst)
#     rst_path = os.path.join(json_dir, file_name.split('.playback')[0]+'.json')
#     print(f"Saving Json to {rst_path} with {len(list(rst.keys()))} scenarios ...")
#     with open(rst_path, 'w') as fp:
#         json.dump(rst, fp)
#     print("Saving Done")


def run_convert(path):
    if path is None:
        simulation_path = 'intersim_rst/ltp_results/09261653'
        playback_path = os.path.join(simulation_path, 'playback')
    else:
        playback_path = path
        simulation_path = os.path.join(path, '..')

    all_pickles = [[each_path, os.path.join(playback_path, each_path)] for each_path in os.listdir(playback_path)]
    json_dir = os.path.join(simulation_path, 'json')
    if not os.path.isdir(json_dir):
        os.makedirs(json_dir)
    for file_name, each_path in all_pickles:
        rst = load_data_from_pickle(each_path)
        rst = numpy_to_list(rst)
        rst_path = os.path.join(json_dir, file_name.split('.playback')[0] + '.json')
        print(f"Saving Json to {rst_path} with {len(list(rst.keys()))} scenarios ...")
        with open(rst_path, 'w') as fp:
            json.dump(rst, fp)
        print("Saving Done")

def run_convert_onefile(playback_path, file_name):
    json_dir = os.path.join(playback_path, '..', 'json')
    if not os.path.isdir(json_dir):
        os.makedirs(json_dir)
    each_path = os.path.join(playback_path, file_name+'.playback')
    rst = load_data_from_pickle(each_path)
    rst = numpy_to_list(rst)
    rst_path = os.path.join(json_dir, file_name + '.json')
    print(f"Saving Json to {rst_path} with {len(list(rst.keys()))} scenarios ...")
    with open(rst_path, 'w') as fp:
        json.dump(rst, fp)
    print("Saving Done")

def run_convert_dic(dic):
    return convert_one_scenario(dic)


