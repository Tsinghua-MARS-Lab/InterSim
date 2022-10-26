import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

from dataloader.DataLoaderNuPlan import NuPlanDL
import numpy as np
import pickle

from nuplan.database.nuplan_db.nuplandb_wrapper import NuPlanDBWrapper

# for SH server
os.environ['NUPLAN_DATA_ROOT'] = "/public/MARS/datasets/nuPlan/data"
os.environ['NUPLAN_MAPS_ROOT'] = "/public/MARS/datasets/nuPlan/nuplan-maps-v1.0"
# os.environ['NUPLAN_DB_FILES'] = "/public/MARS/datasets/nuPlan/data/nuplan-v1.0/mini"
os.environ['NUPLAN_DB_FILES'] = "/public/MARS/datasets/nuPlan/data/nuplan-v1.0/data/cache/public_set_boston_train"

# os.environ['NUPLAN_DATA_ROOT'] = "/Users/qiaosun/nuplan/dataset"
# os.environ['NUPLAN_MAPS_ROOT'] = "/Users/qiaosun/nuplan/dataset/maps"
# os.environ['NUPLAN_DB_FILES'] = "/Users/qiaosun/nuplan/dataset/nuplan-v1.0/mini/"
# os.environ['NUPLAN_DB_FILES'] = "/Users/qiaosun/Downloads/data/cache/public_set_pitts_train"

NUPLAN_DATA_ROOT = os.getenv('NUPLAN_DATA_ROOT', '/data/sets/nuplan')
NUPLAN_MAPS_ROOT = os.getenv('NUPLAN_MAPS_ROOT', '/data/sets/nuplan/maps')
NUPLAN_DB_FILES = os.getenv('NUPLAN_DB_FILES', '/data/sets/nuplan/nuplan-v1.0/mini')
NUPLAN_MAP_VERSION = os.getenv('NUPLAN_MAP_VERSION', 'nuplan-maps-v1.0')

cpus = 200

# node 29 (process2): 0-300
# node 29 (process): 300-600
# node 11: 600-900
# node 10: 900-1200
# gpu 56: 1200-1534


files_names = [os.path.join(NUPLAN_DB_FILES, each_path) for each_path in os.listdir(NUPLAN_DB_FILES)]
print("all file names: ", files_names[:2], len(files_names))

files_names = files_names[900:1200]

# can only track the memory but not free them
# import gc
# import os, psutil
# def usage():
#     process = psutil.Process(os.getpid())
#     return process.memory_info()[0] / float(2 ** 20)
# test_files = True
# if test_files:
#     for count, each_path in enumerate(files_names):
#         print("loading: ", each_path, count, "/", len(files_names))
#         db = NuPlanDBWrapper(
#             data_root=NUPLAN_DATA_ROOT,
#             map_root=NUPLAN_MAPS_ROOT,
#             db_files=each_path,
#             map_version=NUPLAN_MAP_VERSION,
#             max_workers=cpus
#         )
#         print("log: ", len(db.log_dbs))
#         del db
#         gc.collect(generation=2)
#         gc.collect(generation=1)
#         gc.collect(generation=0)
#         print(usage())
#     exit()

db = NuPlanDBWrapper(
    data_root=NUPLAN_DATA_ROOT,
    map_root=NUPLAN_MAPS_ROOT,
    db_files=files_names,
    map_version=NUPLAN_MAP_VERSION,
    # max_workers=cpus
)

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

FILE_INTERVAL = 1000000
def process(starting_file):

    print(f"loading scenarios from {starting_file}")

    # scenario_interval = FILE_INTERVAL
    scenario_interval = 3
    ending_file = starting_file + scenario_interval
    # saving_file_path = f'nu_validation_dataset/interactive_relations_traingInter_allAgents_{starting_file}_{ending_scenario}.pickle'
    saving_file_path = f'nu_training_dataset/boston_relations_allAgents_900-1200_{starting_file}_{ending_file}.pickle'

    dataset_name = 'v1.0-trainval'
    # nusc = NuScenes('v1.0-trainval', dataroot=Nuscene_DATAROOT)

    data_loader = NuPlanDL(file_to_start=starting_file, cpus=200, db=db)

    print("total scenarios: ", data_loader.total_file_num)
    # uncomment code below to check statistics without graphs
    non_interact_scene = 0
    interact_scene = 0
    skip_scene = 0
    more_data = True
    data_to_save = {}

    # calculate and save edges for multiple pairs
    while more_data:
        if data_loader.current_file_index >= ending_file:
            break
        loaded_scenario, _ = data_loader.get_next(process_intersection=False, agent_only=True,
                                                  only_predict_interest_agents=False,
                                                  detect_gt_relation=True)
        if isinstance(loaded_scenario, type([])):
            for each_scenario in loaded_scenario:
                if each_scenario is None:
                    print(f"No more data to load")
                    print(f"Total scenarios to save: {len(list(data_to_save.keys()))}")
                    more_data = False
                    save_data_to_pickle(data_to_save, saving_file_path)
                    break
                elif each_scenario['skip']:
                    print("skip scenario with the skipping flag")
                    skip_scene += 1
                    continue
                else:
                    # [agent_id_influencer, agent_id_reactor, frame_idx_reactor_passing_cross_point, abs(frame_diff)]
                    loaded_edges = each_scenario['edges']
                    # loaded_edge_types = each_scenario['edge_type']
                    agent_dic = each_scenario['agent']
                    # matrix, agent_ids = edges_to_matrix(edges=loaded_edges, edge_types=loaded_edge_types,
                    #                                     agent_dic=agent_dic)
                    if len(loaded_edges) < 1:
                        print("skip scenario with no edge")
                        non_interact_scene += 1
                        # continue
                    else:
                        data_to_save[each_scenario['scenario']] = loaded_edges
                        interact_scene += 1

                    if (interact_scene + non_interact_scene) % 500 == 10 or 0 < (
                            interact_scene + non_interact_scene) < 10:
                        print(f"summary: {interact_scene / (interact_scene + non_interact_scene) * 100:.03f}%", " in ",
                              f"{interact_scene + non_interact_scene} scenes", f" and skipping {skip_scene}")
                        print(f"scenarios: {len(list(data_to_save.keys()))} and current: {each_scenario['scenario']}")
                        print(f"inspect: {data_to_save[each_scenario['scenario']]}")

                    if (interact_scene + non_interact_scene) % 10000 == 0:
                        save_data_to_pickle(data_to_save, saving_file_path)
        else:
            if loaded_scenario is None:
                print(f"No more data to load in path - {NUPLAN_DATA_ROOT}")
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
                # loaded_edge_types = loaded_scenario['edge_type']
                agent_dic = loaded_scenario['agent']
                # matrix, agent_ids = edges_to_matrix(edges=loaded_edges, edge_types=loaded_edge_types, agent_dic=agent_dic)
                if len(loaded_edges) < 1:
                    print("skip scenario with no edge")
                    non_interact_scene += 1
                    continue
                else:
                    # data_to_save[loaded_scenario['id']] = [matrix, agent_ids, loaded_edges]
                    data_to_save[loaded_scenario['scenario']] = [_, _, loaded_edges]
                    interact_scene += 1

                if (interact_scene + non_interact_scene) % 500 == 10 or 0 < (interact_scene + non_interact_scene) < 10:
                    each_scenario_key = list(data_to_save.keys())[0]
                    # Example: [False, False, [['ego', '6f12c1981627584f', 84, 79], ['ego', '6f12c1981627584f', 84, 79], ['8ea67908f66a5104', 'e6fb9c3018865978', 82, 77], ['5e227712537156dc', 'bbbb7c46f8c155ef', 47, 42], ['5e227712537156dc', 'bbbb7c46f8c155ef', 47, 42], ['8ea67908f66a5104', 'e6fb9c3018865978', 82, 77]]]
                    print(f"summary: {interact_scene / (interact_scene + non_interact_scene) * 100:.03f}%", " in ",
                          f"{interact_scene + non_interact_scene} scenes", f" and skipping {skip_scene}")
                    print(f"scenarios: {len(list(data_to_save.keys()))} and scenario id: {each_scenario_key}" )
                    print(f"inspect: {data_to_save[loaded_scenario['scenario']]}")

                print(f"sum per scenario: {data_loader.current_file_index} of {data_loader.total_file_num} and {data_loader.current_scenario_index} of {data_loader.current_file_total_scenario}")

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
    # counter = 0
    # data_to_save = {}
    # # path_to_save = 'training_directR_gt.pickle'
    # path_to_save = 'validation_directR_gt.pickle'
    # # all_files = [f for f in os.listdir(".") if os.path.isfile(f) and 'pickle' in f and 'traingInter' in f]
    # # all_files = [f for f in os.listdir("./training_dataset") if os.path.isfile(os.path.join("./training_dataset", f)) and 'pickle' in f and 'traingInter' in f]
    # all_files = [f for f in os.listdir("./validation_dataset") if
    #              os.path.isfile(os.path.join("./validation_dataset", f)) and 'pickle' in f and 'traingInter' in f]
    # for dic_file_name in all_files:
    #     # file_path = os.path.join("./training_dataset", dic_file_name)
    #     file_path = os.path.join("./validation_dataset", dic_file_name)
    #     print('loading: ', file_path)
    #     data = load_data_from_pickle(file_path)
    #     for i in data.keys():
    #         counter += 1
    #         data_to_save.update(data)
    # print("total: ", counter)
    # save_data_to_pickle(data_to_save, path_to_save)
    # exit()
    ########## end of dictionary merging ##############


    # with Pool(100) as p:
        # p.map(process, list(range(800, 1000, FILE_INTERVAL)))
        # p.map(process, list(range(0, 100, 1)))

    # 1523
    # with Pool(100) as p:
    #     p.map(process, list(range(0, 300, 3)))
    with Pool(110) as p:
        p.map(process, list(range(0, 330, 3)))

    # singapore 2400
    # with Pool(200) as p:
    #     p.map(process, list(range(0, 2400, 12)))

    # process(0)

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
