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
tf_example_dir = '../waymo_data/tf_validation_interact'
# tf_example_dir = '/media/aa/HDD/waymo_motion/waymo_open_dataset_motion_v_1_0_0/uncompressed/tf_example/training'
data_loader = WaymoDL(filepath=tf_example_dir, file_to_start=0, max_file_number=100)
# uncomment code below to check statistics without graphs
more_data = True
data_to_save = {}

loaded_dic = load_data_from_pickle('interactive_relations_traingset_200_300.pickle')
for s_id in loaded_dic:
    print(loaded_dic[s_id])
exit()


def save_data_to_pickle(data_to_save):
    if not final_data_saved:
        with open('waymo_prediction_data.pickle', 'wb') as f:
            pickle.dump(data_to_save, f, pickle.HIGHEST_PROTOCOL)

# load map and save
for i in range(10):
    loaded_scenario = data_loader.get_next(process_intersection=False)

    if loaded_scenario is None:
        print("No more data to load in path - ", tf_example_dir)
        more_data = False
        # save_data_to_pickle(data_to_save)
        exit()

    if loaded_scenario['skip']:
        skip_scene += 1
        continue

    data_to_save[loaded_scenario['id']] = loaded_scenario

save_data_to_pickle(data_to_save)

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