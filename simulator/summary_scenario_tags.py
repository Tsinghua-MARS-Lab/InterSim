import sys
import os
import shutil
import importlib.util
import logging
import argparse
import datetime
import copy

# import tensorflow as tf
# import torch
import pickle

# from plan.base_planner import BasePlanner

global_predictors = None


def load_data_from_pickle(loading_file_name=None):
    if loading_file_name is None:
        with open('interactive_relations_non_relation_type.pickle', 'rb') as f:
            return pickle.load(f)
    else:
        with open(loading_file_name, 'rb') as f:
            return pickle.load(f)

def save_data_to_pickle(data_to_save, saving_file_path):
    with open(saving_file_path, 'wb') as f:
        pickle.dump(data_to_save, f, pickle.HIGHEST_PROTOCOL)

# directory_root = '../relation_inter_val_keep/'
directory_root = '../relation_inter_val_noExcluded/'
# directory_root = '../relation_inter_training/'
move_files = True
if move_files:
    # move files into the root folder
    for each_folder in os.listdir(directory_root):
        if os.path.isfile(os.path.join(directory_root, each_folder)):
            continue
        for num in os.listdir(os.path.join(directory_root, each_folder)):
            final_folder = os.path.join(directory_root, each_folder, num, 'relation_tags')
            for each_file in os.listdir(final_folder):
                if '.relation_tags' not in each_file:
                    continue
                final_file_path = os.path.join(directory_root, each_folder, num, 'relation_tags', each_file)
                os.replace(final_file_path, os.path.join(directory_root, each_file))

scenarios_to_keep = []
scenarios_to_inspect = {}
scenarios_to_exclude = {}
special_scenarios = []
summary = {
    'All': 0,
    'In': [0, 0],  # number, rate
    'InNeg': [0, 0],  # number, rate
    'Out': [0, 0],  # number, rate
    'Inspect': [0, 0],  # number, rate
    'FCC': [0, 0],
    'Traffic Rules': [0, 0],
    'Different_Plan': [0, 0],
    'NoInteraction': [0, 0],

    'FCC_Correct': [0, 0],
    'FCC_Wrong': [0, 0],
    'FCC_Others': [0, 0],

    'Conflict_Detect_Correct': [0, 0],
    'Conflict_Missing': [0, 0],
    'Conflict_toYield_Missing': [0, 0],
    'Relation_Pred_Correct': [0, 0],
    'Relation_Pred_False': [0, 0],

    'target_yield': 0,
    'target_pass': 0

}

for each_file in os.listdir(directory_root):
    path = os.path.join(directory_root, each_file)
    if not os.path.isfile(path):
        continue
    if '.relation_tags' not in each_file:
        continue
    dic = load_data_from_pickle(path)
    for each_scenario in dic:
        relation_dic = dic[each_scenario]
        edges_by_planner = relation_dic['edges_by_planner']
        edges_gt = relation_dic['edges_gt']
        ego_index, ego_id = relation_dic['ego']
        # classify
        summary['All'] += 1
        fcc = False
        wrong_pred = False
        correct_pred = False
        ego_influencer = []
        for each_edge_by_planner in edges_by_planner:
            inf, reactor, reason = each_edge_by_planner
            if inf != ego_id and reactor != ego_id:
                continue
            # note no relation included for ego as inf
            if reason == 'FCC':
                fcc = True
                for each_gt_edge in edges_gt:
                    gt_inf, gt_reactor, _, _ = each_gt_edge
                    if gt_inf == reactor and gt_reactor == inf:
                        # wrong prediction
                        wrong_pred = True
                    if gt_inf == inf and gt_reactor == reactor:
                        # correct prediction
                        correct_pred = True
                break
            for each_gt_edge in edges_gt:
                gt_inf, gt_reactor, _, _ = each_gt_edge
                if gt_inf == reactor and gt_reactor == inf:
                    # wrong prediction
                    wrong_pred = True
                    break
                if gt_inf == inf and gt_reactor == reactor:
                    # correct prediction
                    correct_pred = True
                    break

        if fcc:
            summary['FCC'][0] += 1
            summary['Out'][0] += 1
            scenarios_to_exclude[each_scenario] = 'FCC'

            if correct_pred:
                summary['FCC_Correct'][0] += 1
            elif wrong_pred:
                summary['FCC_Wrong'][0] += 1
                special_scenarios.append(each_scenario)
            else:
                summary['FCC_Others'][0] += 1

        elif wrong_pred:
            summary['Inspect'][0] += 1
            summary['Relation_Pred_False'][0] += 1
            summary['Conflict_Detect_Correct'][0] += 1
            scenarios_to_inspect[each_scenario] = [each_edge_by_planner, each_gt_edge]
        elif correct_pred:
            summary['In'][0] += 1
            summary['Relation_Pred_Correct'][0] += 1
            summary['Conflict_Detect_Correct'][0] += 1
            scenarios_to_keep.append(each_scenario)
        elif len(edges_by_planner) > 0:
            # has interaction in planning but no interaction in gt
            summary['Inspect'][0] += 1
            scenarios_to_inspect[each_scenario] = [edges_by_planner, []]
        else:
            # no interactions in planning
            # check relation in gt
            traffic_rule = False
            for each_gt_edge in edges_gt:
                gt_inf, gt_reactor, _, _ = each_gt_edge
                if gt_reactor == ego_id:
                    summary['Traffic Rules'][0] += 1
                    summary['Conflict_Missing'][0] += 1
                    summary['Conflict_toYield_Missing'][0] += 1
                    traffic_rule = True
                    scenarios_to_exclude[each_scenario] = 'TR'
                    break
                elif gt_inf == ego_id:
                    # no interaction on planning and ego is inf in gt
                    # not clear whether correct or no conflicts with new plans
                    traffic_rule = True
                    summary['Different_Plan'][0] += 1
                    summary['Conflict_Missing'][0] += 1
                    scenarios_to_exclude[each_scenario] = 'DiffPlan'
                    # special_scenarios.append(each_scenario)
                    break
            if not traffic_rule:
                # in as negative labels
                summary['NoInteraction'][0] += 1
                summary['InNeg'][0] += 1
                summary['Conflict_Detect_Correct'][0] += 1
                # scenarios_to_exclude[each_scenario] = 'NoInteraction'
                scenarios_to_keep.append(each_scenario)
            else:
                summary['Out'][0] += 1
                scenarios_to_exclude.append(each_scenario)


        # calculate rate

# minus different plans (which cannot be simulated now)
summary['Different_Plan'][1] = summary['Different_Plan'][0] / summary['All']

summary['In'][1] = summary['In'][0] / summary['All']
summary['Out'][1] = summary['Out'][0] / summary['All']
summary['Inspect'][1] = summary['Inspect'][0] / summary['All']
summary['FCC'][1] = summary['FCC'][0] / summary['All']
summary['Traffic Rules'][1] = summary['Traffic Rules'][0] / summary['All']
summary['NoInteraction'][1] = summary['NoInteraction'][0] / summary['All']
summary['InNeg'][1] = summary['InNeg'][0] / summary['All']

summary['All'] -= summary['Different_Plan'][0]


summary['FCC_Correct'][1] = summary['FCC_Correct'][0] / summary['FCC'][0]
summary['FCC_Wrong'][1] = summary['FCC_Wrong'][0] / summary['FCC'][0]
summary['FCC_Others'][1] = summary['FCC_Others'][0] / summary['FCC'][0]

summary['Conflict_Missing'][1] = summary['Conflict_Missing'][0] / (summary['Conflict_Missing'][0] + summary['Conflict_Detect_Correct'][0])
summary['Conflict_Detect_Correct'][1] = summary['Conflict_Detect_Correct'][0] / (summary['Conflict_Missing'][0] + summary['Conflict_Detect_Correct'][0])
summary['Conflict_toYield_Missing'][1] = summary['Conflict_toYield_Missing'][0] / (summary['Conflict_Missing'][0] + summary['Conflict_Detect_Correct'][0])
summary['Relation_Pred_Correct'][1] = summary['Relation_Pred_Correct'][0] / (summary['Relation_Pred_False'][0] + summary['Relation_Pred_Correct'][0])
summary['Relation_Pred_False'][1] = summary['Relation_Pred_False'][0] / (summary['Relation_Pred_False'][0] + summary['Relation_Pred_Correct'][0])


print("Summary Finished: ", summary)
print(f"total scenarios to save: {len(scenarios_to_keep)+len(list(scenarios_to_inspect.keys()))+len(list(scenarios_to_exclude.keys()))}")

save_data_to_pickle(data_to_save={
    'scenarios_to_keep': scenarios_to_keep,
    'scenarios_to_inspect': scenarios_to_inspect,
    'scenarios_to_exclude': scenarios_to_exclude,
    'special_scenarios': special_scenarios,
}, saving_file_path='relation_inspected_scenarios_interVal.pickle')
# }, saving_file_path='relation_inspected_scenarios_interTrain.pickle')

# for waymo interactive validation dataset
# Summary Finished:
#     {'All': 42763, 'In': [2345, 0.054837125552463575], 'Out': [38363, 0.8971073123962304],
#      'Inspect': [2055, 0.04805556205130604], 'FCC': [16421, 0.38400018707761385],
#      'Traffic Rules': [11808, 0.27612655800575264], 'NoInteraction': [10134, 0.23698056731286393]}
