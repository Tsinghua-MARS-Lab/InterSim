from structs import load
import os
import pickle

def save(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


# dir_prefix = '~/results/relation/'
# dir_prefix = '~/results/relation/'
# file_dir_list = ['r.densetnt.relation.v2c.wd_p3.VAL', 'r.densetnt.relation.v2p.wd_p3.VAL', 'r.densetnt.relation.v2v.wd_p3.VAL']
dir_prefix = '~/results/relation/with_threshold'
file_dir_list = ['thresholdP5.r.densetnt.relation.v2c.wd_p3.VAL', 'VthresholdP9.r.densetnt.relation.v2p.wd_p3.VAL']
# file_dir_list = ['thresholdP5.r.densetnt.relation.v2c.wd_p3.TEST', 'thresholdP5.r.densetnt.relation.v2p.wd_p3.TEST']
data_to_save = {}
for each_file in file_dir_list:
    obj = load(os.path.join(dir_prefix, each_file))
    print(f'{each_file} loaded with {len(list(obj.keys()))} scenarios')
    data_to_save.update(obj)
path_to_save = '~/results/relation/with_threshold/thresholdP5VP9.r.densetnt.relation.v2c.v2p.wd_p3.VAL'
# path_to_save = '~/results/relation/with_threshold/thresholdP5P5.r.densetnt.relation.v2c.v2p.wd_p3.TEST'
save(path_to_save, data_to_save)
print(f"file saved with {len(list(data_to_save.keys()))} scenarios")
