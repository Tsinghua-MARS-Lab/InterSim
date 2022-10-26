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

# from plan.base_planner import BasePlanner

def main(args):
    spec = importlib.util.spec_from_file_location('config', args.config)
    if spec is None:
        parser.error('Config file not found.')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    datatime = datetime.datetime.now()
    output_dir = os.path.join(args.log_dir,
                              f'{datatime.month:02d}{datatime.day:02d}{datatime.hour:02d}{datatime.minute:02d}')
    handle_log_dir(output_dir, args=args)

    # config env after setting logging
    env = gym.make("Drive-v0")
    env_config = config.Env_config()
    # set up your planner
    # trajectory_chooser = BasePlanner(env_config)
    # env.configure(env_config, predictor=trajectory_chooser.online_predictor)
    env.configure(env_config, args=args)

    data_multiplier = 100
    counter = 0
    assert env.running_mode == 1

    for i in range(10000000):  # 00):
        logging.info("running on new scenario: {}-{}".format(env.data_loader.current_file_index,
                                                             env.data_loader.current_scenario_index))
        env.reset(output_dir=output_dir)
        done = False
        while not done:
            # trajectory chooser:
            # action: 0-5 represents the index of the trajectory the planner chooses
            if False:
                next_action = trajectory_chooser.get_action(env.scenario_frame_number, state_next)
            else:
                next_action = 0
            state_next, reward, done, info = env.step(next_action)
        env.update_data_to_save()
        env.save_playback(output_dir, clear=False)
        if counter < data_multiplier:
            # add the counter and simulate current scenario again
            counter += 1
            env.data_loader.current_scenario_index -= 1
        else:
            # reset counter and sim next scenario
            counter = 0
    env.save_playback(output_dir)



def handle_log_dir(output_dir, args):
    #configure logging
    level=logging.INFO if not args.debug else logging.DEBUG
    stdout_handler = logging.StreamHandler(sys.stdout)
    if args.save_log:
        make_new_dir = True
        if os.path.exists(output_dir):
            if args.overwrite:
                shutil.rmtree(output_dir)
            else:
                key = input('training data directory already exists! Overwrite the folder?(y/n)')
                if key == 'y' and not args.resume:
                    shutil.rmtree(output_dir)
                else:
                    make_new_dir = False
        if make_new_dir:
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
    parser.add_argument('--render', default=True, action='store_true')
    parser.add_argument('--log_dir',type=str,default='training_data')
    parser.add_argument('--overwrite',default=True,action='store_true')
    parser.add_argument('--resume',default=False,action='store_true')
    parser.add_argument('--debug',default=False,action='store_true')
    parser.add_argument('--save_log', default=False, action='store_true')
    # parser.add_argument('--save_playback', default=True, action='store_true')
    args_p = parser.parse_args()
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
