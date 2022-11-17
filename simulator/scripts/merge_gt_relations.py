import sys, os
import shutil
import pickle

paths = os.listdir('../')
file_paths = []
for each_path in paths:
    if 'relation_gt' in each_path:
        file_paths.append(each_path)
rst = {}
for each_path in file_paths:
    path_to_load = os.path.join('../', each_path)
    loaded_dic = None
    with open(path_to_load, 'rb') as f:
        loaded_dic = pickle.load(f)
    for each_key in loaded_dic:
        if each_key not in rst:
            rst[each_key] = loaded_dic[each_key]

def save_data_to_pickle(data_to_save, saving_file_path):
    with open(saving_file_path, 'wb') as f:
        pickle.dump(data_to_save, f, pickle.HIGHEST_PROTOCOL)

save_data_to_pickle(rst, 'relation_gt_merged.pickle')
print("merged relation with ", len(list(rst.keys())), " scenarios.")