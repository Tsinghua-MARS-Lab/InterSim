import pickle5 as pickle
import os
import numpy as np
import random

def save(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

# change this number into 10x if you only want to consider crossings within x seconds
TIME_TO_CONSIDER = 999999999

# file_path = '../Relation_Matrix/interactive_relations_traingset_matrix_0-100.pickle'
# path ='../Relation_Matrix/'
path = None  # your path to process
list_of_files = []

for root, dirs, files in os.walk(path):
    for file in files:
        print("test: ", file)
        if 'traingInter' in file:
            list_of_files.append(os.path.join(root, file))
# for name in list_of_files:
#     print(name)
data_to_save = {}
label_counter = [0, 0, 0]
last_scenario_id = None
all_agent_ids = {}
for each_file_name in list_of_files:
    obj = load(each_file_name)
    print(f'file {each_file_name} loaded with {len(list(obj.keys()))}')
    for i, scenario_id in enumerate(obj):
        assert scenario_id not in data_to_save
        relations = []
        relations_pairs = []
        agent_ids = []
        # sample of the matrix:
        # [array([[0, 1, 0, ..., 0, 0, 0],
        #        [0, 0, 0, ..., 0, 0, 0],
        #        [0, 0, 0, ..., 0, 0, 0],
        #        ...,
        #        [0, 0, 0, ..., 0, 0, 0],
        #        [0, 0, 0, ..., 0, 0, 0],
        #        [0, 0, 0, ..., 0, 0, 0]]),
        #        [2671, 2678, 2689, 3259, 2692, 2672, 2673, 2683, 2687, 2691, 2677, 2674, 2684, 2680, 2676, 2681, 2690, 2694, 2697, 2699, 2700, 2701, 2706, 2703, 2707, 2712, 2716, 2695, 2714, 2715, 2717, 2721, 2709, 2718, 2719, 2713, 2720, 2696, 2722, 2723, 2724, 2704, 2727, 2730, 2725, 2705, 2726, 3258, 2686, 2732, 2738, 2675],
        #        [[2671.0, 2678.0, 88.0], [2671.0, 2678.0, 88.0], [2671.0, 2678.0, 88.0], [2671.0, 2701.0, 86.0]]]
        # shape of index 1, index 2: (52, 52) (52,)
        for relation_pair in obj[scenario_id][2]:
            if int(relation_pair[0]) < int(relation_pair[1]):
                label_int = 0
            elif int(relation_pair[0]) > int(relation_pair[1]):
                label_int = 1
            else:
                assert False, f'two same agent ids in one pair, {scenario_id, relation_pair}'
            this_pair = [int(relation_pair[0]), int(relation_pair[1]), label_int]
            collidion_start_at_index = relation_pair[2]
            if collidion_start_at_index > TIME_TO_CONSIDER:
                continue
            if this_pair in relations:
                continue
            relations.append(this_pair)
            relations_pairs.append([int(relation_pair[0]), int(relation_pair[1])])
            label_counter[label_int] += 1
            agent_ids.append(this_pair[0])
            agent_ids.append(this_pair[1])

        # add equal number of label 2 pairs to relation list
        # num_of_label_two_to_add = label_counter[0] + label_counter[1] - label_counter[2]
        # num_of_all_agents = len(obj[scenario_id][1])
        # for j in range(num_of_all_agents):
        #     for k in range(num_of_all_agents):
        #         if label_counter[0] + label_counter[1] - label_counter[2] <= 0:
        #             continue
        #         if j == k:
        #             continue
        #         agent1_id = obj[scenario_id][1][j]
        #         agent2_id = obj[scenario_id][1][k]
        #         if [agent1_id, agent2_id] not in relations_pairs and [agent2_id, agent1_id] not in relations_pairs:
        #             relations.append([agent1_id, agent2_id, 2])
        #             label_counter[2] += 1
        relations = np.array(relations)
        if len(relations.shape) == 1:
            relations = relations[np.newaxis, :]
        data_to_save[scenario_id] = relations
        all_agent_ids[scenario_id] = agent_ids
        last_scenario_id = scenario_id
file_to_save = 'gt_direct_relation_1223.pickle'
save(file_to_save, data_to_save)
path_to_save = 'all_agents_ids_1223.pickle'
save(path_to_save, all_agent_ids)
print(f'data saved with {len(list(data_to_save.keys()))} scenarios')
print(f'samples number of each label: {label_counter}')
print(f'one data sample: {last_scenario_id}, {data_to_save[last_scenario_id]}, {all_agent_ids[last_scenario_id]}')




