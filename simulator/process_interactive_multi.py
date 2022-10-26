from interactive_sim.envs.DataLoader import WaymoDL
import numpy as np
import pickle


def load_data_from_pickle(loading_file_name=None):
    if loading_file_name is None:
        with open('interactive_relations_non_relation_type.pickle', 'rb') as f:
            return pickle.load(f)
    else:
        with open(loading_file_name, 'rb') as f:
            return pickle.load(f)

final_data_saved = False
def save_data_to_pickle(data_to_save, saving_file_path):
    if not final_data_saved:
        with open(saving_file_path, 'wb') as f:
            pickle.dump(data_to_save, f, pickle.HIGHEST_PROTOCOL)

def edges_to_matrix(edges, edge_types, agent_dic):
    total_agent_num = len(list(agent_dic.keys()))
    keys_list = list(agent_dic.keys())
    matrix_to_return = np.zeros((total_agent_num, total_agent_num), dtype=np.int)
    for i, edge in enumerate(edges):
        agent_id1, agent_id2, _, _ = edge
        idx_1 = keys_list.index(agent_id1)
        idx_2 = keys_list.index(agent_id2)
        matrix_to_return[idx_1, idx_2] = edge_types[i]
    return matrix_to_return, keys_list


# tf_example_dir = '../waymo_data/tf_validation_interact'
# tf_example_dir = '/media/aa/HDD/waymo_motion/waymo_open_dataset_motion_v_1_0_0/uncompressed/tf_example/training'
# training files number: 1000
# tf_example_dir = '/public/MARS/datasets/waymo_motion/waymo_open_dataset_motion_v_1_0_0/uncompressed/tf_example/training_interactive/'
# validation files number: 150
tf_example_dir = '/public/MARS/datasets/waymo_motion/waymo_open_dataset_motion_v_1_0_0/uncompressed/tf_example/validation_interactive/'


# calculate and save edges for one pair
# while more_data:
#     loaded_scenario = data_loader.get_next(process_intersection=False, agent_only=True, only_predict_interest_agents=True)
#
#     if loaded_scenario is None:
#         print("No more data to load in path - ", tf_example_dir)
#         more_data = False
#         save_data_to_pickle(data_to_save)
#         exit()
#
#     if loaded_scenario['skip']:
#         skip_scene += 1
#         print("WARNING: Skipping scenario: ", loaded_scenario['scenario'])
#         continue
#
#     loaded_edges = loaded_scenario['edges']
#     loaded_edge_types = loaded_scenario['edge_type']
#
#     if len(loaded_edges) > 0:
#         assert len(loaded_edges) == 1, loaded_edges
#         edge_zero = loaded_edges[0]
#         interact_scene += 1
#         # label: 0=small id first, 1=large id first, 2=no edge/relation
#         if int(edge_zero[0]) > int(edge_zero[1]):
#             label = 1
#         else:
#             label = 0
#     else:
#         non_interact_scene += 1
#         label = 2
#
#     data_to_save[loaded_scenario['id']] = np.array([edge_zero[0], edge_zero[1], label, loaded_edge_types[0]], dtype=int)
#
#     if (interact_scene + non_interact_scene) % 500 == 0 or (interact_scene + non_interact_scene) < 10:
#         print(f"summary: {interact_scene / (interact_scene + non_interact_scene) * 100:.03f}%", " in ",
#               f"{interact_scene + non_interact_scene} scenes", f" and skipping {skip_scene}")
#         print(f"inspect: {data_to_save[loaded_scenario['id']]}")
#
#     if (interact_scene + non_interact_scene) % 10000 == 0:
#         save_data_to_pickle(data_to_save)

FILE_INTERVAL = 1
def process(starting_file):
    print("loading file from:", starting_file)

    file_interval = FILE_INTERVAL
    # file_interval = 1
    ending_file = starting_file + file_interval
    if ending_file == 1000:
        ending_file = '1k'
    saving_file_path = f'validation_dataset/interactive_relations_traingInter_allAgents_{starting_file}_{ending_file}.pickle'
    # saving_file_path = f'training_dataset/interactive_relations_traingInter_allAgents_{starting_file}_{ending_file}.pickle'

    data_loader = WaymoDL(filepath=tf_example_dir, file_to_start=starting_file, max_file_number=file_interval)
    # uncomment code below to check statistics without graphs
    non_interact_scene = 0
    interact_scene = 0
    skip_scene = 0
    more_data = True
    data_to_save = {}

    # calculate and save edges for multiple pairs
    while more_data:
        loaded_scenario = data_loader.get_next(process_intersection=False, agent_only=True, only_predict_interest_agents=False)

        if loaded_scenario is None:
            print(f"No more data to load in path - {tf_example_dir}")
            print(f"Total scenarios to save: {len(list(data_to_save.keys()))}")
            more_data = False
            save_data_to_pickle(data_to_save, saving_file_path)
            break
        elif loaded_scenario['skip']:
            print("skip scenario with the skipping flag")
            skip_scene += 1
            continue
        else:
            # [agent_id_influencer, agent_id_reactor, frame_idx_reactor_passing_cross_point, abs(frame_diff)]
            loaded_edges = loaded_scenario['edges']
            loaded_edge_types = loaded_scenario['edge_type']
            agent_dic = loaded_scenario['agent']
            matrix, agent_ids = edges_to_matrix(edges=loaded_edges, edge_types=loaded_edge_types, agent_dic=agent_dic)
            if len(loaded_edges) < 1:
                print("skip scenario with no edge")
                non_interact_scene += 1
                continue
            else:
                data_to_save[loaded_scenario['id']] = [matrix, agent_ids, loaded_edges]
                interact_scene += 1

            if (interact_scene + non_interact_scene) % 500 == 10 or 0 < (interact_scene + non_interact_scene) < 10:
                print(f"summary: {interact_scene / (interact_scene + non_interact_scene) * 100:.03f}%", " in ",
                      f"{interact_scene + non_interact_scene} scenes", f" and skipping {skip_scene}")
                print(f"scenarios: {len(list(data_to_save.keys()))}")
                print(f"inspect: {data_to_save[loaded_scenario['id']]}")

            if (interact_scene + non_interact_scene) % 10000 == 0:
                save_data_to_pickle(data_to_save, saving_file_path)
    #
    # end of multiple pair

    save_data_to_pickle(data_to_save, saving_file_path)
    print(f"Total scenarios saved and loop ended: {len(list(data_to_save.keys()))} to {saving_file_path}")

from multiprocessing import Pool

if __name__ == '__main__':

    ########## merge all dictionary in one ###############
    import os
    counter = 0
    data_to_save = {}
    # path_to_save = 'training_directR_gt.pickle'
    path_to_save = 'validation_directR_gt.pickle'
    # all_files = [f for f in os.listdir(".") if os.path.isfile(f) and 'pickle' in f and 'traingInter' in f]
    # all_files = [f for f in os.listdir("./training_dataset") if os.path.isfile(os.path.join("./training_dataset", f)) and 'pickle' in f and 'traingInter' in f]
    all_files = [f for f in os.listdir("./validation_dataset") if
                 os.path.isfile(os.path.join("./validation_dataset", f)) and 'pickle' in f and 'traingInter' in f]
    for dic_file_name in all_files:
        # file_path = os.path.join("./training_dataset", dic_file_name)
        file_path = os.path.join("./validation_dataset", dic_file_name)
        print('loading: ', file_path)
        data = load_data_from_pickle(file_path)
        for i in data.keys():
            counter += 1
            data_to_save.update(data)
    print("total: ", counter)
    save_data_to_pickle(data_to_save, path_to_save)
    exit()
    ########## end of dictionary merging ##############


    # with Pool(100) as p:
        # p.map(process, list(range(800, 1000, FILE_INTERVAL)))
        # p.map(process, list(range(0, 100, 1)))
    with Pool(50) as p:
        p.map(process, list(range(100, 150, 1)))

# only add relation type
# data_to_save = load_data_from_pickle()
# while more_data:
#     loaded_scenario = data_loader.get_next(process_intersection=False, relation=False)
#     if loaded_scenario['skip']:
#         continue
#     if loaded_scenario is None:
#         print("No more data to load in path - ", tf_example_dir)
#         more_data = False
#         save_data_to_pickle(data_to_save)
#         print("inspect:", data_to_save)
#         exit()
#
#     data_to_save[loaded_scenario['id']] = np.append(data_to_save[loaded_scenario['id']], loaded_scenario['edge_type'])
#
# print("unexpected ending")
# exit()


# data_to_return = {
#     "road": road_dic,
#     "agent": agent_dic,
#     "traffic_light": traffic_dic
# }
#
# category = classify_scenario(data_to_return)
# data_to_return["category"] = category
# data_to_return['id'] = scenario_id
#
# data_to_return['edges'] = edges
