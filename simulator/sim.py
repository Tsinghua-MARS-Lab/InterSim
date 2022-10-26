# from gym import spaces, logger
#
# action_space = spaces.Discrete(2)
# print(action_space.n)
# action = 0
# err_msg = "%r (%s) invalid" % (action, type(action))
# assert action_space.contains(action), err_msg
import sys
import os
import shutil
import gym
import interactive_sim
import importlib.util
import logging
import argparse
import datetime
import copy

import tensorflow as tf
import torch

# from plan.base_planner import BasePlanner

global_predictors = None
global_simulation_info = None

def main(args):
    global global_predictors

    spec = importlib.util.spec_from_file_location('config', args.config)
    if spec is None:
        parser.error('Config file not found.')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # datatime = datetime.datetime.now()
    # output_dir = os.path.join(args.log_dir,
    #                           f'{datatime.month:02d}{datatime.day:02d}{datatime.hour:02d}{datatime.minute:02d}')
    # handle_log_dir(output_dir, args=args)
    output_dir = args.log_dir

    # if args.multi_process:
    #     output_dir = os.path.join(output_dir, f'{args.starting_file_num}-{args.ending_file_num}')

    # config env after setting logging
    env = gym.make("Drive-v0")
    env_config = config.EnvConfig()
    # set up your planner
    # trajectory_chooser = BasePlanner(env_config)
    # env.configure(env_config, predictor=trajectory_chooser.online_predictor)
    env.configure(env_config, args=args)
    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    env.register_simulation(output_dir=output_dir, starting_time=start_time)

    if env.running_mode == 1:
        for _ in range(args.max_scenarios):  # 00):
            if global_predictors is None:
                ended = env.reset(output_dir=output_dir)
            else:
                ended = env.reset(output_dir=output_dir, predictor_list=global_predictors)

            if ended:
                break
            logging.info("running on new scenario: {}-{}".format(env.data_loader.current_file_index,
                                                                 env.data_loader.current_scenario_index))
            if env.env_planner is not None and env.env_planner.online_predictor is not None:
                if global_predictors is None:
                    global_predictors = [env.env_planner.online_predictor.goal_setter,
                                         env.env_planner.online_predictor.relation_predictor,
                                         env.env_planner.online_predictor.marginal_predictor]
            done = False
            while not done:
                # trajectory chooser:
                # action: 0-5 represents the index of the trajectory the planner chooses
                if False:
                    next_action = trajectory_chooser.get_action(env.scenario_frame_number, state_next)
                else:
                    next_action = 0
                state_next, reward, done, info = env.step(next_action)
            # env.update_data_to_save()
            # if i % 10 == 0:
            #     env.save_playback(output_dir, clear=False)
        # print("Saving final result to ", output_dir)
        # env.save_playback(output_dir, offset=-1)
    elif env.running_mode == 2:
        for _ in range(args.max_scenarios):
            ended = env.reset()
            if ended:
                break
            logging.info("running on new scenario: {}-{}".format(env.data_loader.current_file_index,
                                                                 env.data_loader.current_scenario_index))
            done = False
            while not done:
                if args.render:
                    env.render()
                # trajectory chooser:
                # action: 0-5 represents the index of the trajectory the planner chooses
                if False:
                    next_action = trajectory_chooser.get_action(env.scenario_frame_number, state_next)
                else:
                    next_action = 0
                state_next, reward, done, info = env.step(next_action)
    elif env.running_mode == 0:
        for _ in range(args.max_scenarios):
            ended = env.reset(output_dir=output_dir)
            if ended:
                break
            logging.info("running on new scenario: {}-{}".format(env.data_loader.current_file_index,
                                                                 env.data_loader.current_scenario_index))
            done = False
            while not done:
                if args.render:
                    env.render()
                # trajectory chooser:
                # action: 0-5 represents the index of the trajectory the planner chooses
                if False:
                    next_action = trajectory_chooser.get_action(env.scenario_frame_number, state_next)
                else:
                    next_action = 0
                state_next, reward, done, info = env.step(next_action)

                # if info['frame_num'] == 10:
                #     print("infos: ", info)
                # if info['frame_num'] == 0:
                #     key = input('Press any key to load the next scene')
                # elif info['frame_num'] == 11:
                #     key = input('Make prediction and press any key')
                # if info['frame_num'] > 11:
                #     done = True
                #     print("LOADING NEXT SCENARIO")
    global_simulation_info = env.sim_infos
    ending_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    env.register_simulation(output_dir=output_dir, starting_time=start_time, status='Done', ending_time=ending_time)
    print("Simulation Finished!")


def handle_log_dir(output_dir, args):
    #configure logging
    level=logging.INFO if not args.debug else logging.DEBUG
    stdout_handler = logging.StreamHandler(sys.stdout)
    if args.save_log:
        if os.path.exists(output_dir):
            if args.overwrite and not args.multi_process:
                shutil.rmtree(output_dir)
            else:
                if not args.multi_process:
                    key = input('training data directory already exists! Overwrite the folder?(y/n)')
                    if key == 'y' and not args.resume:
                        shutil.rmtree(output_dir)
                    else:
                        make_new_dir = False
        else:
            os.makedirs(output_dir)
            shutil.copy(args.config, os.path.join(output_dir, 'config.py'))
        mode = 'a' if args.resume else 'w'
        log_file = os.path.join(output_dir, 'output.log')
        file_handler = logging.FileHandler(log_file, mode=mode)
        logging.basicConfig(level=level, handlers=[stdout_handler,file_handler],
                            format="%(asctime)s,%(levelname)s:%(message)s",datefmt="%Y-%m-%d %H:%M:%S")
    else:
        logging.basicConfig(level=level, handlers=[stdout_handler,],
                            format="%(asctime)s,%(levelname)s:%(message)s",datefmt="%Y-%m-%d %H:%M:%S")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    # parser.add_argument('--config', type=str, default='config_mode1.py')#'config.py')
    # parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--config', type=str, default='config.py')
    parser.add_argument('--render', default=True, action='store_false')
    parser.add_argument('--method', type=str, default='unknown')
    parser.add_argument('--log_dir',type=str,default='sim_result')
    parser.add_argument('--overwrite',default=True,action='store_true')
    parser.add_argument('--resume',default=False,action='store_true')
    parser.add_argument('--debug',default=False,action='store_true')
    parser.add_argument('--save_log', default=False, action='store_true')
    parser.add_argument('--starting_file_num', type=int, default=-1)
    parser.add_argument('--ending_file_num', type=int, default=-1)
    parser.add_argument('--multi_process', default=False, action='store_true')
    parser.add_argument('--file_per_worker', type=int, default=1)
    parser.add_argument('--save_playback_data', default=True, action='store_true')
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
    if env_config.env.dataset == 'Waymo':
        dataset_with_map = f'Waymo-{env_config.env.map_name}'
    elif env_config.env.dataset == 'NuPlan':
        dataset_with_map = f'NuPlan-{env_config.env.map_name}'
    output_dir = os.path.join(args_p.log_dir,
                              f'{args_p.method}_{dataset_with_map}_{datatime.month:02d}{datatime.day:02d}{datatime.hour:02d}{datatime.minute:02d}')

    if args_p.multi_process:
        # multi-processer
        from multiprocessing import Pool
        starting_file_num = args_p.starting_file_num
        ending_file_num = args_p.ending_file_num
        total_files = ending_file_num - starting_file_num
        assert total_files > 0, total_files

        handle_log_dir(output_dir, args=args_p)
        args_p.log_dir = output_dir

        file_per_worker = args_p.file_per_worker
        processor = int(total_files/file_per_worker)
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
        from sim_result.summary import SummaryAPI
        s = SummaryAPI()
        s.summary_and_save(output_dir, global_simulation_info)
        # iter_list = []
        # for i in range(total_files):
        #     new_args = copy.deepcopy(args_p)
        #     new_args.starting_file_num = starting_file_num+i
        #     new_args.ending_file_num = starting_file_num+i+1
        #     iter_list.append(new_args)
        # for args in iter_list:
        #     print("test numbers: ", args.starting_file_num, args.ending_file_num)
        # with Pool(total_files) as p:
        #     p.map(main, iter_list)
    else:
        handle_log_dir(output_dir, args=args_p)
        args_p.log_dir = output_dir
        main(args_p)



# def cartpole():
#     env = gym.make("CartPole-v1")
#     observation_space = env.observation_space.shape[0]
#     action_space = env.action_space.n
#     dqn_solver = DQNSolver(observation_space, action_space)
#     while True:
#         state = env.reset()
#         state = np.reshape(state, [1, observation_space])
#         while True:
#             env.render()
#             action = dqn_solver.act(state)
#             state_next, reward, terminal, info = env.step(action)
#             reward = reward if not terminal else -reward
#             state_next = np.reshape(state_next, [1, observation_space])
#             dqn_solver.remember(state, action, reward, state_next, terminal)
#             dqn_solver.experience_replay()
#             state = state_next
#             if terminal:
#                 break
