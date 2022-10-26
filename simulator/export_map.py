from dataloader.DataLoaderNuPlan import NuPlanDL

data_loader = NuPlanDL(scenario_to_start=0)
road_dic= data_loader.get_map()

import json
import os

import pickle
import numpy as np

def numpy_to_list(dic):
    # two level
    print(f'Total Scenarios: {len(list(dic.keys()))}')
    counter = 0
    for each_scenario_id in dic:
        counter += 1
        if counter > 100:
            break
        data_dic = dic[each_scenario_id]
        keys = ['road', 'agent', 'traffic_light', 'predicting']
        print(f'Processsing: {each_scenario_id}')
        for each_key in data_dic:
            # 'agent'
            if each_key not in keys:
                assert not isinstance(data_dic[each_key], dict), f'{each_key}'
            elif each_key == 'predicting':
                sub_dic = data_dic[each_key]
                del sub_dic['original_trajectory']
                del sub_dic['points_to_mark']
                traj_to_mark = sub_dic['trajectory_to_mark']  # a list of nparray
                converted_list = [each_np.tolist() for each_np in traj_to_mark]
                sub_dic['trajectory_to_mark'] = converted_list
            else:
                sub_dic = data_dic[each_key]
                for each_obj_id in sub_dic:
                    data_each_obj = sub_dic[each_obj_id]
                    if isinstance(data_each_obj, dict):
                        for each_obj_key in data_each_obj:
                            # 'dir', 'type', etc
                            obj = data_each_obj[each_obj_key]
                            if type(obj) == type(np.empty(1)):
                                data_each_obj[each_obj_key] = obj.tolist()
                            elif isinstance(obj, list):
                                target_obj = []
                                for each_obj in obj:
                                    if type(each_obj) == type(np.empty(1)):
                                        target_obj.append(each_obj.tolist())
                                    else:
                                        target_obj.append(each_obj)
                                data_each_obj[each_obj_key] = target_obj
                            elif isinstance(obj, dict):
                                print("unkown dict in dict", obj, each_obj_key)


        # if isinstance(obj, dict):
        #     for each_key2 in obj:
        #         obj2 = dic[each_key][each_key2]
        #         print("test: ", type(obj2), type(obj2)==type(np.ones((1))))
        #         if isinstance(obj2, type(np.ones((1)))):
        #             dic[each_key][each_key2] = obj2.tolist()
    return dic

rst = {'map':{'road': road_dic}}
rst = numpy_to_list(rst)
rst_path = 'boston_map.json'
with open(rst_path, 'w') as fp:
    json.dump(rst, fp)

print('saved with ', len(list(road_dic.keys())))