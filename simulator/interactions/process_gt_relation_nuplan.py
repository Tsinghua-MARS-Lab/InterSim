from multiprocessing import Pool
import os, argparse, importlib.util

import sys
import shutil
import gym
import interactive_sim
import logging
import argparse
import datetime
import copy

def save_data_to_pickle(data_to_save, saving_file_path):
    if not final_data_saved:
        with open(saving_file_path, 'wb') as f:
            pickle.dump(data_to_save, f, pickle.HIGHEST_PROTOCOL)

def main(args):
    spec = importlib.util.spec_from_file_location('config', args.config)
    if spec is None:
        parser.error('Config file not found.')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    output_dir = args.log_dir
    env = gym.make("Drive-v0")
    env_config = config.EnvConfig()
    env.configure(env_config, args=args)
    saving_file_path = f'relation_gt_{args.starting_file_num}_{args.ending_file_num}'

    non_interact_scene = 0
    interact_scene = 0
    data_to_save = {}

    assert env.running_mode == 1, env.running_mode
    for _ in range(args.max_scenarios):
        ended = env.reset(output_dir=output_dir, detect_gt_relation=True)
        if ended:
            break
        logging.info("running on new scenario: {}-{}".format(env.data_loader.current_file_index,
                                                             env.data_loader.current_scenario_index))
        loaded_edges = env.data_dic['edges']
        if len(loaded_edges) < 1:
            print("skip scenario with no edge")
            non_interact_scene += 1
        else:
            data_to_save[env.data_dic['scenario']] = loaded_edges
            interact_scene += 1
        if (interact_scene + non_interact_scene) % 500 == 10 or 0 < (interact_scene + non_interact_scene) < 10:
            print(f"summary: {interact_scene / (interact_scene + non_interact_scene) * 100:.03f}%", " in ",
                  f"{interact_scene + non_interact_scene} scenes")
            print(f"scenarios: {len(list(data_to_save.keys()))} and current: {each_scenario['scenario']}")
            print(f"inspect: {data_to_save[each_scenario['scenario']]}")
        if (interact_scene + non_interact_scene) % 10000 == 0:
            save_data_to_pickle(data_to_save, saving_file_path)

    save_data_to_pickle(data_to_save, saving_file_path)
    print(f"Total scenarios saved and loop ended: {len(list(data_to_save.keys()))} to {saving_file_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default='config.py')
    parser.add_argument('--log_dir', type=str, default='nu_training_dataset')
    parser.add_argument('--starting_file_num', type=int, default=-1)
    parser.add_argument('--ending_file_num', type=int, default=-1)
    parser.add_argument('--multi_process', default=False, action='store_true')
    parser.add_argument('--file_per_worker', type=int, default=1)
    parser.add_argument('--max_scenarios', type=int, default=100000)
    # parser.add_argument('--save_playback', default=True, action='store_true')
    args_p = parser.parse_args()
    spec = importlib.util.spec_from_file_location('config', args_p.config)
    if spec is None:
        parser.error('Config file not found.')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    env_config = config.EnvConfig()
    datatime = datetime.datetime.now()
    assert env_config.env.dataset == 'NuPlan', env_config.env.dataset
    dataset_with_map = f'NuPlan-{env_config.env.map_name}'
    output_dir = os.path.join(args_p.log_dir,
                              f'{dataset_with_map}')

    if args_p.multi_process:
        # multi-processer
        from multiprocessing import Pool

        starting_file_num = args_p.starting_file_num
        ending_file_num = args_p.ending_file_num
        total_files = ending_file_num - starting_file_num
        assert total_files > 0, total_files

        args_p.log_dir = output_dir

        file_per_worker = args_p.file_per_worker
        processor = int(total_files / file_per_worker)
        iter_list = []
        for i in range(processor):
            new_args = copy.deepcopy(args_p)
            new_args.starting_file_num = starting_file_num + i * file_per_worker
            new_args.ending_file_num = starting_file_num + (i + 1) * file_per_worker
            # if new_args.starting_file_num not in [113, 133]:
            #     continue
            iter_list.append(new_args)
        for args in iter_list:
            print("multiprocessing with start_file end_file: ", args.starting_file_num, args.ending_file_num)
        with Pool(total_files) as p:
            p.map(main, iter_list)
            p.close()
            p.join()

        print("All multiprocessing simulation finished")

    else:
        args_p.log_dir = output_dir
        main(args_p)
