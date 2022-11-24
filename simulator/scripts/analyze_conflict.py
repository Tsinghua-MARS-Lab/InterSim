import os
import pickle
import numpy as np

def load_data_from_pickle(loading_file_name):
    with open(loading_file_name, 'rb') as f:
        return pickle.load(f)

# method = 'relation_type'
method = 'relation_type_NuPlan-Boston_11210934'
sim_paths = os.listdir('sim_result')
summary_dic_to_save = {}
total_scenarios = 0
non_trivial_conflicts = 0
for each_path in sim_paths:
    if method not in each_path:
        continue
    playback_folder = os.path.join('sim_result', each_path, 'playback')
    for each_playback_name in os.listdir(playback_folder):
        if '.playback' not in each_playback_name:
            continue
        loaded_dic = load_data_from_pickle(os.path.join(playback_folder, each_playback_name))
        for each_scenario in loaded_dic:
            loaded_edges = loaded_dic[each_scenario]['predicting']['all_relations_last_step']
            print(f"Adding {each_scenario} with {loaded_edges}")
            summary_dic_to_save[each_scenario] = loaded_edges
            total_scenarios += 1
            for each_edge in loaded_edges:
                inf, reactor, type = each_edge
                if reactor != 'ego' or type == 'FCC':
                    continue
                else:
                    non_trivial_conflicts += 1
                    break
rst_path = 'boston_nuplan_relation_analyze.pickle'
# with open(path, 'wb') as f:
#     pickle.dump(obj, f)

print(f"Summary: \ntotal scenarios: {total_scenarios}\nnon-trivial rate: {non_trivial_conflicts / (total_scenarios + 0.001)}")