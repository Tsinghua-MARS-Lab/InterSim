import gym
try:
    from interactive_sim.envs.graphics import *
    GRAPHICS_AVAILABLE = True
except Exception as e:
    GRAPHICS_AVAILABLE = False
    print("Import graphics with errors, Do not render: " + str(e))

import numpy as np
import time, os, logging, pickle, math, copy

from dataloader.DataLoaderNuPlan import NuPlanDL
try:
    from dataloader.DataLoaderNuPlan import NuPlanDL
except:
    print("Import NuPlanDL fail")

try:
    from dataloader.DataLoaderWaymo import WaymoDL
except:
    print("Import WaymoDL fail")

import interactive_sim.envs.util as utils
from interactive_sim.envs.metrics import EnvMetrics

from prediction.predictor import Predictor
from prediction.predictor import PredictionLoader
from prediction.predictor import PredictionLoaderMultiTime

from prediction.M2I.predictor import M2IPredictor
from plan.base_planner import BasePlanner
from plan.env_planner import EnvPlanner
from plan.trajectory_prediction_planner import TrajPredPlanner
from plan.dummy_planner import DummyPlanner
from plan.e2e_planner import E2EPlanner
# from plan.chainpnp_planner import EnvPlanner

import interactive_sim.envs.util as utils
from interactive_sim.envs.metrics import EnvMetrics

DRAW_AGENT_ARROW = False
DRAW_AGENT_ID = True
DRAW_LANE_ID = True
DRAW_RELATION_EDGE = True
DRAW_DASHED_LINES = False
TEXT_SIZE = 8  # good for snap shot
TEXT_SIZE = 15  # good for debug

# Inspect Options
INSPECT_ON_NO_EDGES_SCENES = False

if GRAPHICS_AVAILABLE:
    PREDICTION_COLORS = [color_rgb(255, 204, 13), color_rgb(255, 115, 38), color_rgb(255, 25, 77),
                         color_rgb(191, 38, 105), color_rgb(112, 42, 140), color_rgb(136, 247, 226),
                         color_rgb(68, 212, 146)]
else:
    PREDICTION_COLORS = ['black']


def init_your_planner_here(planner=None, env_config=None, predictor=None, dataset=None):
    from plan.base_planner import BasePlanner
    return BasePlanner(env_config=env_config, predictor=M2IPredictor(), dataset=dataset)

def init_your_predictor_here(predictor=None):
    from prediction.M2I.predictor import M2IPredictor
    return M2IPredictor()


class DriveEnv(gym.Env):
    def __init__(self):
        # meta_data
        self.running_mode = None
        self.window_w = 0
        self.window_h = 0
        self.scale = 0
        self.show_trajectory = None
        self.frame_rate = 0
        self.data_path = None

        self.data_loader = None
        self.data_dic = {}
        self.drawer = None

        self.scenario_frame_number = None
        self.state = None

        self.last_time = 0
        self.prediction_data = {}

        self.collided_pairs = []
        self.total_collisions = 0
        self.draw_prediction = True
        # self.use_online_predictor = None
        self.loaded_prediction_path = None
        self.loaded_prediction = None
        self.loaded_prediction_with_offset = False
        self.draw_collision_pt = False
        self.env_planner = None

        self.last_file = None
        self.data_to_save = {}

        self.ego_planner = None

        self.filter_config = {}
        self.metrics = EnvMetrics()

        self.last_tic = None
        self.agents_packed = []

        self.dataset = 'Waymo'
        self.config = None

        self.save_playback_data = False
        self.sim_infos = None

    # def set_prediction_dic(self, prediction_dic):
    #     self.prediction_data = prediction_dic

    def configure(self, config, args, predictor=None):
        self.dataset = config.env.dataset
        self.running_mode = config.env.running_mode

        self.window_w = config.env.window_w
        self.window_h = config.env.window_h
        self.scale = config.env.scale
        self.show_trajectory = config.env.show_trajectory
        self.frame_rate = config.env.frame_rate
        if args.render:
            if self.running_mode == 0:
                assert GRAPHICS_AVAILABLE, 'Cannot render without a monitor (Graphics not imported)'
                self.draw_prediction = config.env.draw_prediction
                self.draw_collision_pt = config.env.draw_collision_pt
                if self.draw_prediction:
                    self.loaded_prediction_path = config.env.loaded_prediction_path
                    self.loaded_prediction_with_offset = config.env.load_prediction_with_offset
                if self.loaded_prediction_path is not None:
                    with open(self.loaded_prediction_path, 'rb') as f:
                        self.loaded_prediction = pickle.load(f)
        self.drawer = GraphicDrawer(self.scale, self.frame_rate, self.window_w, self.window_h, self.show_trajectory,
                                    dataset=self.dataset)
        # check starting or ending number
        if args.starting_file_num != -1:
            starting_file_num = args.starting_file_num
        else:
            starting_file_num = None

        if args.ending_file_num != -1 and starting_file_num is not None:
            max_file_num = args.ending_file_num - starting_file_num
            assert max_file_num > 0, max_file_num
        else:
            max_file_num = None

        if self.dataset == 'Waymo':
            self.data_path = config.env.tf_example_dir
            self.data_loader = WaymoDL(filepath=self.data_path, gt_relation_path=config.env.relation_gt_path,
                                       file_to_start=starting_file_num,
                                       max_file_number=max_file_num)
        elif self.dataset == 'NuPlan':
            self.data_path = config.env.data_path
            self.data_loader = NuPlanDL(scenario_to_start=0,
                                        file_to_start=starting_file_num,
                                        max_file_number=max_file_num,
                                        data_path=self.data_path)

        self.data_loader.total_frames = config.env.planning_to

        self.config = config
        self.scenario_frame_number = 0

        if config.env.predictor == 'M2I':
            predictor = M2IPredictor()
        elif config.env.predictor is None:
            predictor = None
        else:
            predictor = init_your_predictor_here(predictor=config.env.predictor)

        if config.env.dynamic_env_planner == 'env':
            # not playback, dynamic environment
            self.env_planner = EnvPlanner(env_config=config, predictor=predictor, dataset=self.dataset)
        if config.env.ego_planner == 'base':
            self.ego_planner = BasePlanner(env_config=config, predictor=predictor, dataset=self.dataset)
        elif config.env.ego_planner == 'trajpred':
            self.ego_planner = TrajPredPlanner(env_config=config, predictor=predictor)
        elif config.env.ego_planner == 'e2e':
            self.ego_planner = E2EPlanner(env_config=config, predictor=predictor)
        elif config.env.ego_planner == 'dummy':
            self.ego_planner = DummyPlanner(env_config=config, predictor=predictor)
        elif config.env.ego_planner == 'playback':
            self.ego_planner = PlaybackPlanner(env_config=config, predictor=predictor)
        elif config.env.ego_planner is None:
            self.ego_planner = None
        else:
            self.ego_planner = init_your_planner_here(planner=config.env.ego_planner,
                                                      env_config=config, predictor=predictor,
                                                      dataset=self.dataset)

        logging.info('The hyperparameter of gym environment is list as blow')
        logging.info('frame rate is :{},'.format(self.frame_rate))

        if config.env.filter_static:
            self.filter_config['filter_static'] = True
        if config.env.filter_non_vehicle:
            self.filter_config['filter_non_vehicle'] = True

        self.save_playback_data = args.save_playback_data

        self.sim_infos = {
            'name': args.method,
            'task': config.env.planning_task,
            'dataset': self.dataset,
            'planner': config.env.ego_planner,
            'predictor': config.env.predictor,
            'max_frames': config.env.planning_to,
            'map_info': config.env.map_name  # None if self.dataset == 'Waymo' else config.env.map_name
        }

        print("DriveEnv Environment initialized")

    def set_offset(self):
        data_dic = self.data_dic
        agents = data_dic["agent"]

        # compute offsets and scales
        minimal_x, minimal_y = np.inf, np.inf
        max_x, max_y = -np.inf, -np.inf

        for agent_id in agents.keys():
            random_agent_id = agent_id
            break
        total_frame = agents[random_agent_id]['pose'].shape[0]
        # assert total_frame == 91

        for frame_idx in range(total_frame):
            for agent_id in agents.keys():
                agent = agents[agent_id]
                # print(agent["pose"][frame_idx, :].flatten().tolist())
                x, y, _, yaw = agent["pose"][frame_idx, :].flatten().tolist()
                if x == -1.0 or y == -1.0:
                    continue
                if abs(x) < 100 or abs(y) < 100:
                    continue
                if x < minimal_x:
                    minimal_x = x
                if x > max_x:
                    max_x = x
                if y < minimal_y:
                    minimal_y = y
                if y > max_y:
                    max_y = y

        self.drawer.offsets = - abs(max_x - minimal_x) / 2 - minimal_x, - abs(max_y - minimal_y) / 2 - minimal_y

    def step(self, action):
        done = False
        reward = 0
        info = {'frame_num': self.scenario_frame_number}

        if self.data_dic['skip']:
        # if self.data_dic['skip'] or ('predicting' in self.data_dic and self.data_dic['predicting']['goal_fit'] is not None and not self.data_dic['predicting']['goal_fit']):
            done = True
            return self.state, reward, done, info

        self.scenario_frame_number += 1
        self.state = dictionary_to_state(self.data_dic, self.scenario_frame_number)

        # commit environment step by predicting results
        if self.ego_planner is not None and self.running_mode != 2:
            self.ego_planner.clear_markers_per_step(current_frame_idx=self.scenario_frame_number,
                                                    current_state=self.data_dic)

            # frame_diff = self.scenario_frame_number - self.ego_planner.planning_from
            # if frame_diff >= 0 and frame_diff % self.ego_planner.planning_interval == 0:
            if self.ego_planner.is_planning(self.scenario_frame_number):
                ego_id = self.data_dic['predicting']['ego_id'][1]
                self.data_dic['agent'][ego_id]['pose'][self.scenario_frame_number:, :] = -1
            planned_dic = self.ego_planner.plan_ego(current_state=self.data_dic,
                                                    current_frame_idx=self.scenario_frame_number)
            if planned_dic is not None:
                self.data_dic = planned_dic
                # self.env_planner.online_predictor.update_state(self.data_dic)

        if self.env_planner is not None and self.running_mode != 2:
            # assert 'predicting' in self.data_dic, list(self.data_dic.keys())
            # plan and update trajectory to commit for relevant environment agents
            if self.ego_planner is None:
                self.env_planner.clear_markers_per_step(current_frame_idx=self.scenario_frame_number,
                                                        current_state=self.data_dic)
            self.env_planner.get_prediction_trajectories(current_frame_idx=self.scenario_frame_number,
                                                         current_state=self.data_dic,
                                                         time_horizon=self.config.env.planning_horizon)
            # if 'predicting' in self.data_dic:
            #     # choose one trajectory to draw
            #     # self.prediction_data = self.data_dic['predicting']['marginal_trajectory']
            #     self.prediction_data = self.data_dic['predicting']['guilded_trajectory']
            # assert self.env_planner.online_predictor.prediction_data is not None
            # planned_dic = self.env_planner.update_env_trajectory(current_frame_idx=self.scenario_frame_number,
            #                                                      current_state=self.data_dic,
            #                                                      drawer=self.drawer)
            # if self.ego_planner is None:
            #     planned_dic = self.env_planner.update_env_trajectory_for_sudo_base_planner(
            #         current_frame_idx=self.scenario_frame_number,
            #         current_state=self.data_dic)
            #
            # if planned_dic is not None:
            #     self.data_dic = planned_dic
            #     self.env_planner.online_predictor.update_state(self.data_dic)
            planned_dic = self.env_planner.update_env_trajectory_reguild(current_frame_idx=self.scenario_frame_number,
                                                                         current_state=self.data_dic,
                                                                         plan_for_ego=self.ego_planner is None)
            if planned_dic is not None:
                self.data_dic = planned_dic
                # self.env_planner.online_predictor.update_state(self.data_dic)

        # check if scenario is ended
        if (self.running_mode in [0, 1] and self.env_planner is not None) or self.running_mode == 2:
            # doing planning or replay planning
            if self.scenario_frame_number >= self.config.env.planning_to - 1:
                done = True
        else:
            if self.scenario_frame_number >= 90:
                done = True

        # get current mode
        self.data_dic['agent'] = utils.mark_agents_mode(self.data_dic['agent'], self.scenario_frame_number)

        if not done:
            # for agent in self.drawer.agents:
            #     for poly in agent.agent_polys:
            #         poly.undraw()
            # self.drawer.agents = []

            self.agents_packed = []

            for idx, agent_id in enumerate(self.data_dic['agent'].keys()):
                # pack each agent and draw
                agent = self.data_dic['agent'][agent_id]
                x, y, _, yaw = agent["pose"][self.scenario_frame_number]
                if x == -1.0 or y == -1.0 or yaw == -1.0:
                    continue
                yaw = utils.normalize_angle(yaw + math.pi / 2)
                width, length, _ = agent["shape"][0]
                width, length = max(width, 1), max(length, 1)
                if 'speed' in agent:
                    v, v_yaw = agent["speed"][self.scenario_frame_number]
                else:
                    v, v_yaw = 0, 0
                offsets = self.drawer.offsets
                recentered_xy = self.drawer.recenter((x, y), offsets)
                new_agent = Agent(x=recentered_xy[0],
                                  y=recentered_xy[1], yaw=-yaw,
                                  vx=v, length=length * self.drawer.scale, width=width * self.drawer.scale,
                                  agent_id=agent_id)
                if agent['to_predict']:
                    if 'to_predict' not in info:
                        info['to_predict'] = []
                    if agent_id not in info['to_predict']:
                        info['to_predict'].append(agent_id)

                # important for collision detection even for mode 1
                self.agents_packed.append(new_agent)

                # check collision
                for agent in self.agents_packed:
                    # if new_agent.vx < 0.05 and agent.vx < 0.05:
                    #     # do not detect two parking cars
                    #     continue
                    # if check_collision_for_two_agents(checking_agent=new_agent,
                    #                                   target_agent=agent):
                    if new_agent.x == -1.0 or new_agent.y == -1.0 or new_agent.yaw == -1.0 or agent.x == -1.0 or agent.yaw == -1.0:
                        continue
                    if new_agent.agent_id == agent.agent_id:
                        continue
                    if utils.check_collision_for_two_agents_rotate_and_dist_check(checking_agent=new_agent,
                                                                                  target_agent=agent):
                        # if utils.check_collision_for_two_agents_dense(checking_agent=new_agent, target_agent=agent):
                        new_agent.crashed = True
                        agent.crashed = True
                        crashed_pair = [new_agent.agent_id, agent.agent_id]
                        crashed_pair.sort()
                        if crashed_pair not in self.collided_pairs and (
                                'predicting' not in self.data_dic or self.data_dic['predicting']['ego_id'][
                            1] in crashed_pair):
                            # we only calculate collidings with the ego
                            self.collided_pairs.append(crashed_pair)
                            # log collision types to metrics
                            yaw_diff = utils.normalize_angle(new_agent.yaw - agent.yaw)
                            if -math.pi / 180 * 30 < yaw_diff < math.pi / 180 * 30:
                                self.metrics.rear_collisions += 1
                            elif -math.pi / 180 * 150 < yaw_diff or yaw_diff > math.pi / 180 * 150:
                                self.metrics.side_collisions += 1
                            else:
                                self.metrics.front_collisions += 1
                            if self.dataset == 'Waymo':
                                if self.data_dic['scenario_str'] not in self.metrics.colliding_scenarios:
                                    self.metrics.colliding_scenarios.append(self.data_dic['scenario_str'])
                            elif self.dataset == 'NuPlan':
                                if self.data_dic['scenario'] not in self.metrics.colliding_scenarios:
                                    self.metrics.colliding_scenarios.append(self.data_dic['scenario'])

                            if 'predicting' in self.data_dic:
                                if self.data_dic['predicting']['emergency_stopping']:
                                    self.metrics.emergency_stops += 1

                                # terminate current scenario if collision with ego
                                logging.info('Terminate on collision')
                                self.data_dic['predicting']['terminate_log'] = [self.scenario_frame_number, 'collision']
                                done = True

        # if self.ego_planner is not None and (not self.ego_planner.current_on_road):
        #     # terminate off road (>30m) scenarios
        #     self.metrics.offroad_scenarios += 1
        #     logging.info('Terminate at offroad')
        #     self.data_dic['predicting']['terminate_log'] = [self.scenario_frame_number, 'offroad']  # 'collision', 'offroad'
        #     done = True

        if self.running_mode in [0, 1] and self.env_planner is not None and done:
            # doing planning and done, then calculate metrics
            logging.info(f" scenario ended: collisions: {len(self.collided_pairs)} / {self.total_collisions}")
            self.total_collisions += len(self.collided_pairs)
            metrics_cover_ego = False
            if self.env_planner.test_task in [1, 2]:
                metrics_cover_ego = True
            metric_per_scenario = self.metrics.update_per_scenario(data_dic=self.data_dic,
                                                                   collisions=self.collided_pairs,
                                                                   include_ego=metrics_cover_ego,
                                                                   current_frame=self.scenario_frame_number)
            if 'metrics' not in self.data_dic:
                self.data_dic['metrics'] = {}
            metric_per_scenario['collided_pairs'] = self.collided_pairs
            # if len(flipping_relations) > 0:
            #     self.data_dic['metrics']['flipping_relations'] = flipping_relations
            # if len(engage_with_false_r) > 0:
            #     self.data_dic['metrics']['engage_with_false_r'] = engage_with_false_r
            # if progress is not None:
            #     self.data_dic['metrics']['progress'] = progress
            # else:
            #     self.data_dic['metrics']['progress'] = -1
            self.data_dic['metrics'] = metric_per_scenario

            logging.info(self.metrics.summary())
            if 'planner_timer' in self.data_dic:
                logging.info(sum(self.data_dic['planner_timer']) / (len(self.data_dic['planner_timer']) + 0.001))
                logging.info(sum(self.data_dic['predict_timer']) / (len(self.data_dic['predict_timer']) + 0.001))
                for each_time in self.data_dic['predict_timer']:
                    if each_time > 0.1:
                        self.metrics.gpu_relation_predictions += 1

        if 'edges' in self.data_dic and isinstance(self.data_dic['edges'], list):
            # new version for prediction, has different prediction in each frame
            # if self.scenario_frame_number in self.data_dic['edges']:
            #     info['edges'] = self.data_dic['edges'][self.scenario_frame_number]
            # old version for gt, has only one label
            if 'to_predict' in info and len(info['to_predict']) > 0:
                edges_info = []
                for each_edge in self.data_dic['edges']:
                    if each_edge[1] in info['to_predict']:
                        edges_info.append(each_edge)
                info['edges'] = edges_info
            else:
                info['edges'] = self.data_dic['edges']

        if 'predicting' in self.data_dic and self.scenario_frame_number > self.env_planner.planning_from and self.running_mode != 1:
            if self.ego_planner is not None and self.ego_planner.is_first_planning(self.scenario_frame_number):
                self.drawer.draw_route(data_dic=self.data_dic, agent_id=self.data_dic['predicting']['ego_id'][1])

        if done and self.save_playback_data:
            # add current scenario simulation result into current file playback to save
            self.update_data_to_save()

        # print("DriveEnv Stepped")
        return self.state, reward, done, info

    def reset(self, output_dir=None, predictor_list=None):
        if self.config.env.planning_task == 'LTP':
            assert self.dataset == 'NuPlan', f'LTP currently only support NuPlan but got dataset of {self.dataset}'
            map_dic = self.data_loader.get_map()
            # Todo

        self.prediction_data = {}
        if self.running_mode in [0, 1] and self.config.env.relation_gt_path is not None:
            relation = True
        else:
            relation = False

        if self.config.env.playback_dir is None:
            loaded_scenario, new_files_loaded = self.data_loader.get_next(filter_config=self.filter_config,
                                                                          relation=relation,
                                                                          seconds_in_future=int(self.config.env.planning_to / self.frame_rate))
        else:
            loaded_scenario, new_files_loaded = self.data_loader.get_next_from_playback(
                filter_config=self.filter_config,
                playback_dir=self.config.env.playback_dir)

        if loaded_scenario is None:
            print("ERROR: no more data to load in path - ", self.data_path)
            return True

        assert isinstance(loaded_scenario, dict), loaded_scenario
        assert isinstance(new_files_loaded, bool)

        def set_planner(loaded_scenario):
            if self.dataset == 'NuPlan':
                if self.env_planner is not None:
                    self.env_planner.map_api = self.data_loader.map_api
                if self.ego_planner is not None:
                    self.ego_planner.map_api = self.data_loader.map_api

            if loaded_scenario['skip']:
                print("Skip planner setting with skip flag")
                return loaded_scenario
            if self.ego_planner is not None:
                self.ego_planner.reset(new_data=loaded_scenario, model_path=self.config.env.model_path,
                                       time_horizon=self.config.env.planning_horizon, predict_device=self.config.env.predict_device,
                                       predictor_list=predictor_list, ego_planner=True)
            if self.env_planner is not None:
                self.env_planner.reset(new_data=loaded_scenario, model_path=self.config.env.model_path,
                                       time_horizon=self.config.env.planning_horizon, predict_device=self.config.env.predict_device,
                                       predictor_list=predictor_list, ego_planner=False)
                assert 'predicting' in loaded_scenario, list(loaded_scenario.keys())
            return loaded_scenario

        loaded_scenario = set_planner(loaded_scenario)

        # check skip flag
        to_skip = loaded_scenario['skip']
        while to_skip:
            print("Skipping with skip flag loaded")
            if self.config.env.playback_dir is None:
                loaded_scenario, new_files_loaded_p = self.data_loader.get_next(filter_config=self.filter_config,
                                                                                relation=relation,
                                                                                seconds_in_future=int(self.config.env.planning_to / self.frame_rate))
            else:
                loaded_scenario, new_files_loaded_p = self.data_loader.get_next_from_playback(
                    filter_config=self.filter_config,
                    playback_dir=self.config.env.playback_dir)
            if loaded_scenario is None:
                # end current sim
                if output_dir is not None and self.save_playback_data:
                    print(f"[Ending Saving] Saving playback to {output_dir} with file index of {self.data_loader.current_file_index - 1}")
                    # save playback and clear cache
                    self.save_playback(output_dir, offset=-1, clear=True)
                    # save simulation result and refresh metric for a new file
                    self.save_metrics(output_dir=output_dir, refresh_metric=True)
                return True
            loaded_scenario = set_planner(loaded_scenario)
            new_files_loaded = new_files_loaded | new_files_loaded_p
            to_skip = loaded_scenario['skip']
            if not to_skip and len(list(loaded_scenario['agent'].keys())) < 1:
                to_skip = True

        if (self.config.env.save_log_every_scenario or new_files_loaded) and output_dir is not None and self.save_playback_data:
            print(f"Saving playback to {output_dir} with file index of {self.data_loader.current_file_index - 1}")
            # save playback and clear cache
            if new_files_loaded:
                self.save_playback(output_dir, offset=-1, clear=True)
            else:
                self.save_playback(output_dir, offset=-1, clear=False)
            # save simulation result and refresh metric for a new file
            self.save_metrics(output_dir=output_dir, refresh_metric=True)

        if INSPECT_ON_NO_EDGES_SCENES:
            # only show non-edge scenes
            empty_edges = False

            while not empty_edges:
                if len(loaded_scenario["edges"]) > 0:
                    loaded_scenario, new_files_loaded = self.data_loader.get_next()
                    print("Has edges, skipping")
                else:
                    empty_edges = True

        interested_scenario = False
        while not interested_scenario:
            if "category" in loaded_scenario.keys():
                # change list into [2, 3] to run only on intersections
                if loaded_scenario["category"] not in [1, 2, 3]:
                    loaded_scenario, new_files_loaded = self.data_loader.get_next()
                    if loaded_scenario is None:
                        print("No more data to load in path - ", self.data_path)
                        return True
                else:
                    interested_scenario = True
            else:
                interested_scenario = True

        if self.running_mode == 0:
            if self.draw_prediction and self.env_planner is None:
                # using loaded prediction with predictor data loader
                # self.prediction_data = self.data_loader.current_scene_pred
                prediction_found = False
                if self.loaded_prediction_with_offset:
                    data_loader = PredictionLoaderMultiTime(all_agents=False)
                else:
                    data_loader = PredictionLoader(all_agents=False)

                self.prediction_data = data_loader(loaded_scenario, prediction_loaded=self.loaded_prediction)
                while not prediction_found:
                    if self.prediction_data is None:
                        loaded_scenario, new_files_loaded = self.data_loader.get_next()
                        self.prediction_data = data_loader(loaded_scenario, prediction_loaded=self.loaded_prediction)
                    else:
                        prediction_found = True

        self.data_dic = loaded_scenario
        self.scenario_frame_number = 0
        self.state = dictionary_to_state(loaded_scenario)

        agents = self.data_dic["agent"]
        for agent_id in agents.keys():
            random_agent_id = agent_id
            break
        self.total_frame = agents[random_agent_id]['pose'].shape[0]
        # assert self.total_frame == 91, self.total_frame

        self.drawer.reset(self.scale, self.frame_rate, self.window_w, self.window_h, self.show_trajectory)
        self.set_offset()
        # print("DriveEnv Reset")

        self.collided_pairs = []

        if self.draw_collision_pt:
            # temporal drawing to check collision point predictions
            # path = 'collisionPt.vis.off0.model15.pickle'
            path = '0117.vis.collisionPt.pickle'
            path = 'waymo.densetnt.raster.tfR.predModes.noStopMode.off0-20.outNoChange.outFromStop.0505.mdl17.vis_val.0506.pickle'
            with open(path, 'rb') as f:
                loaded = pickle.load(f)

            scenario_id = loaded_scenario['scenario']
            if scenario_id in loaded:
                self.loaded_current = loaded[scenario_id]
            else:
                self.loaded_current = {}
        print("resetting scenario: ", loaded_scenario['scenario'])
        return False

    def render(self, mode='human'):
        road = self.data_dic["road"]
        if self.dataset == 'Waymo':
            scenario_id = self.data_dic["scenario_str"]
        elif self.dataset == 'NuPlan':
            scenario_id = self.data_dic["scenario"]
        if self.drawer.win is None:
            self.drawer.init_window()
        else:
            current_time = time.time()
            time_interval = current_time - self.last_time + 0.001
            self.last_time = current_time
            fps = round(1 / time_interval, 2)
            self.drawer.draw_fps(fps, frame_idx=self.scenario_frame_number,
                                 message='scenario: #' + str(self.data_loader.current_scenario_index))

        if not self.drawer.map_inited:
            if self.dataset == 'Waymo':
                for road_seg_id in road.keys():
                    road_seg = road[road_seg_id]
                    # self.drawer.init_map_drawing(road_dic=road_seg, scenario_type=self.data_dic["category"])
                    # DEBUG: uncomment to show road_id on screen
                    self.drawer.init_map_drawing(road_dic=road_seg, scenario_type=self.data_dic["category"],
                                                 road_id=road_seg_id)
                self.drawer.map_inited = True
            elif self.dataset == 'NuPlan':
                for road_seg_id in road.keys():
                    # draw others
                    road_seg = road[road_seg_id]
                    if road_seg['type'] not in [17, 18]:
                        continue
                    self.drawer.init_map_drawing(road_dic=road_seg, scenario_type=self.data_dic["category"],
                                                 road_id=road_seg_id)

                # draw route
                for each_road_id in self.data_dic['route']:
                    if each_road_id not in road:
                        continue
                    road_seg = road[each_road_id]
                    self.drawer.init_route_drawing(road_seg)

                # draw lanes
                for road_seg_id in road.keys():
                    road_seg = road[road_seg_id]
                    if road_seg['type'] in [17, 18]:
                        continue
                    # self.drawer.init_map_drawing(road_dic=road_seg, scenario_type=self.data_dic["category"])
                    # DEBUG: uncomment to show road_id on screen
                    self.drawer.init_map_drawing(road_dic=road_seg, scenario_type=self.data_dic["category"],
                                                 road_id=road_seg_id)
                if 'ego_goal' in self.data_dic:
                    goal = self.data_dic['ego_goal']
                    self.drawer.init_goal_drawing(goal=copy.deepcopy(goal))

                self.drawer.map_inited = True

        for agent in self.drawer.agents:
            for poly in agent.agent_polys:
                poly.undraw()

        self.drawer.agents = []
        for idx, agent_id in enumerate(self.data_dic['agent'].keys()):
            # pack each agent and draw
            agent = self.data_dic['agent'][agent_id]
            agent_type = agent['type']
            if self.dataset == 'NuPlan' and agent_type not in [0, 7]:
                # only draw vehicles to render faster
                continue
            x, y, _, yaw = agent["pose"][self.scenario_frame_number]
            if x == -1.0 or y == -1.0 or yaw == -1.0:
                continue
            yaw = utils.normalize_angle(yaw + math.pi / 2)
            width, length, _ = agent["shape"][0]
            if 'speed' in agent:
                v, v_yaw = agent["speed"][self.scenario_frame_number]
            else:
                v, v_yaw = 0, 0
            offsets = self.drawer.offsets
            recentered_xy = self.drawer.recenter((x, y), offsets)
            new_agent = Agent(x=recentered_xy[0],
                              y=recentered_xy[1], yaw=-yaw,
                              vx=v, length=length * self.drawer.scale, width=width * self.drawer.scale,
                              agent_id=agent_id)
            self.drawer.agents.append(new_agent)

        if 'predicting' in self.data_dic:
            self.drawer.draw_agent_on_window_by_frame(self.data_dic["agent"], self.scenario_frame_number,
                                                      self.running_mode,
                                                      ego_id=self.data_dic['predicting']['ego_id'][1])
        else:
            self.drawer.draw_agent_on_window_by_frame(self.data_dic["agent"], self.scenario_frame_number,
                                                      self.running_mode)

        if self.running_mode == 0:
            if self.scenario_frame_number > 11 and self.draw_prediction:
                if 'predicting' in self.data_dic:
                    pass
                    # self.drawer.draw_prediction_on_window_for_pred(self.prediction_data)
                else:
                    if self.loaded_prediction_with_offset:
                        self.drawer.draw_prediction_on_window_by_timeoffset(self.prediction_data,
                                                                            self.data_dic["agent"],
                                                                            self.scenario_frame_number - 11)
                    else:
                        self.drawer.draw_prediction_on_window_by_frame(self.prediction_data, self.data_dic["agent"],
                                                                       self.scenario_frame_number - 11)
        elif self.running_mode == 1:
            pass

        # draw marks from prediction
        if 'predicting' in self.data_dic:
            for each_point in self.data_dic['predicting']['points_to_mark']:
                self.drawer.draw_one_prediction_with_pts(agent_pose=np.array(each_point)[np.newaxis, :])
            self.drawer.draw_mark_trajectory_on_window_for_pred(self.data_dic['predicting']['trajectory_to_mark'])

            if 'mark_pts' in self.data_dic['predicting']:
                for each_point in self.data_dic['predicting']['mark_pts']:
                    self.drawer.draw_one_pt_single_frame(pt=each_point)

            draw_org_path = True
            if draw_org_path:
                ego_id = self.data_dic['predicting']['ego_id'][1]
                org_ego_pose = self.data_dic['predicting']['original_trajectory'][ego_id]["pose"][
                    self.scenario_frame_number]
                width, length, _ = self.data_dic['predicting']['original_trajectory'][ego_id]["shape"][0]
                recentered_ego_org = self.drawer.recenter((org_ego_pose[0], org_ego_pose[1]), self.drawer.offsets)
                v, v_yaw = 0, 0
                org_ego_agent = Agent(x=recentered_ego_org[0],
                                      y=recentered_ego_org[1], yaw=- org_ego_pose[3] - math.pi / 2,
                                      vx=v, length=length * self.drawer.scale, width=width * self.drawer.scale,
                                      agent_id=ego_id)

                agent_w = org_ego_agent.width
                agent_l = org_ego_agent.length
                self.drawer.draw_rectangle(
                    pts=utils.generate_contour_pts_with_direction((org_ego_agent.x, org_ego_agent.y),
                                                                  agent_w, agent_l,
                                                                  org_ego_agent.yaw), color='white',
                    agent=org_ego_agent,
                    mode='outline')
                self.drawer.agents.append(org_ego_agent)

        if self.draw_collision_pt:
            if 'predicting' in self.data_dic:
                xpt = self.data_dic['predicting']['XPt']
                for inf_id, reactor_id in xpt:
                    cpt_pose = xpt[(inf_id, reactor_id)]['pred_collision_pt']
                    self.drawer.draw_one_prediction_with_pts(agent_pose=cpt_pose[0][np.newaxis, :])

                # gpt = self.data_dic['predicting']['goal_pts']
                # for agent_id in gpt:
                #     if agent_id not in [731, 932, 752, 750]:
                #         continue
                #     pose = gpt[agent_id][0]
                #     if pose is None:
                #         continue
                #     else:
                #         pose = np.array(pose)
                #     self.drawer.draw_one_prediction_with_pts(agent_pose=pose[np.newaxis, :])
            else:
                # for mode changing pt
                pt_pose = self.loaded_current['pred_goals']
                # print("test: ", pt_pose[0], self.loaded_current['ids'], self.loaded_current['next_mode'], self.loaded_current['score'])
                self.drawer.draw_one_prediction_with_pts(agent_pose=pt_pose[0][np.newaxis, :])

                # for idx, agent_pair in enumerate(self.loaded_current.keys()):
                #     # print("test0: ",  self.loaded_current[agent_pair].keys())
                #     if self.scenario_frame_number - 11 in self.loaded_current[agent_pair]:
                #         # print("test1: ", self.loaded_current[agent_pair][self.scenario_frame_number - 11].keys())
                #         cpt_pose = self.loaded_current[agent_pair][self.scenario_frame_number - 11]["rst"]
                #         # print("test: ", self.scenario_frame_number, cpt_pose)
                #         # agent_yaw = loaded_current[agent_pair][0]["pred_yaw"]
                #         self.drawer.draw_one_prediction_with_pts(agent_pose=cpt_pose[0][np.newaxis, :])

        if DRAW_RELATION_EDGE:
            if 'predicting' in self.data_dic:
                if 'relations_per_frame_env' in self.data_dic['predicting']:
                    if self.scenario_frame_number in self.data_dic["predicting"]['relations_per_frame_env']:
                        self.drawer.draw_relation_from_prediction(
                            self.data_dic["predicting"]['relations_per_frame_env'][self.scenario_frame_number],
                            color='red')
                    elif len(list(self.data_dic["predicting"]['relations_per_frame_env'].keys())) > 0:
                        last_key = list(self.data_dic["predicting"]['relations_per_frame_env'].keys())[-1]
                        if 0 <= self.scenario_frame_number - last_key < 5:
                            self.drawer.draw_relation_from_prediction(
                                self.data_dic["predicting"]['relations_per_frame_env'][last_key],
                                color='red')
                if 'relations_per_frame_ego' in self.data_dic['predicting']:
                    if self.scenario_frame_number in self.data_dic["predicting"]['relations_per_frame_ego']:
                        self.drawer.draw_relation_from_prediction(
                            self.data_dic["predicting"]['relations_per_frame_ego'][self.scenario_frame_number],
                            color='blue')
                    elif len(list(self.data_dic["predicting"]['relations_per_frame_ego'].keys())) > 0:
                        last_key = list(self.data_dic["predicting"]['relations_per_frame_ego'].keys())[-1]
                        if 0 <= self.scenario_frame_number - last_key < 5:
                            self.drawer.draw_relation_from_prediction(
                                self.data_dic["predicting"]['relations_per_frame_ego'][last_key],
                                color='blue')
                # if 'relations_per_frame_ego' in self.data_dic['predicting'] and self.scenario_frame_number in self.data_dic["predicting"]['relations_per_frame_ego']:
                #     self.drawer.draw_relation_from_prediction(self.data_dic["predicting"]['relations_per_frame_ego'][self.scenario_frame_number], color='blue')
                # else:
                #     self.drawer.draw_relation_from_prediction(self.data_dic["predicting"]['relation'])
            else:
                self.drawer.draw_relation(self.data_dic["agent"], self.data_dic["edges"], self.scenario_frame_number)

        self.drawer.draw_traffic_light(tl_dics=self.data_dic["traffic_light"], road_dics=road,
                                       frame_idx=self.scenario_frame_number)
        update(self.drawer.frame_rate)

    def update_data_to_save(self):
        self.data_dic['info'] = self.sim_infos
        if self.dataset == 'Waymo':
            save_all = True
            if save_all:
                scenarios_id = self.data_dic['scenario_str']
                self.data_to_save[scenarios_id] = self.data_dic
            else:
                scenarios_id = self.data_dic['scenario_str']
                data_to_save = {
                    'agent': self.data_dic['agent'],
                    # 'road': self.data_dic['road'],
                    # 'traffic_light': self.data_dic['traffic_light']
                    # 'edges': self.data_dic['edges'],
                    # 'predicting': self.data_dic['predicting']
                }
                if 'edges' in self.data_dic:
                    # ground truth relations
                    data_to_save['edges'] = self.data_dic['edges']
                if 'predicting' in self.data_dic:
                    data_to_save['predicting'] = self.data_dic['predicting']
                if 'metrics' in self.data_dic:
                    data_to_save['metrics'] = self.data_dic['metrics']
                # data_to_save['result'] = self.metrics.output()
                if scenarios_id not in self.data_to_save:
                    self.data_to_save[scenarios_id] = {}
                data_to_save['info'] = self.sim_infos
                # self.data_to_save[scenarios_id].append(data_to_save)
                self.data_to_save[scenarios_id] = data_to_save
        elif self.dataset == 'NuPlan':
            # save all data for nuplan as playback
            scenarios_id = self.data_dic['scenario']
            self.data_to_save[scenarios_id] = self.data_dic

    def register_simulation(self, output_dir, status='Running', starting_time=None, ending_time=None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir, 'sim.info')
        dic_to_save = self.sim_infos.copy()
        if self.dataset == 'NuPlan':
            dic_to_save['dataset'] += f'-{self.config.env.map_name}'
        else:
            dic_to_save['dataset'] += f'-{self.config.env.map_name}'
        dic_to_save['status'] = status
        dic_to_save['starting_time'] = starting_time
        dic_to_save['ending_time'] = ending_time
        save(file_path, dic_to_save)

    def save_playback(self, output_dir, clear=True, offset=0):
        if len(list(self.data_to_save.keys())) < 1:
            return
        # inspect is valid
        one_key = list(self.data_to_save.keys())[0]
        one_scenario = self.data_to_save[one_key]

        output_dir = os.path.join(output_dir, 'playback')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_name = self.data_loader.file_names[self.data_loader.current_file_index + offset].split('/')[-1]
        file_path_to_save = os.path.join(output_dir, f'{file_name}.playback')
        print(f"Saving to {file_path_to_save} with {len(list(self.data_to_save.keys()))} scenarios")
        save(file_path_to_save, self.data_to_save)
        print(f"Saving completed")

        import scripts.playback_to_json as convert
        convert.run_convert_onefile(playback_path=output_dir, file_name=file_name)

        if clear:
            self.data_to_save = {}

    def save_metrics(self, output_dir, refresh_metric=True, offset=-1):
        result_dictionary = self.metrics.output()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_name = self.data_loader.file_names[self.data_loader.current_file_index + offset].split('/')[-1]
        assert os.path.exists(output_dir)
        path_to_save = os.path.join(output_dir, f'{file_name}.scene.metric')
        # path_to_save = os.path.join(output_dir, f'sim_rst.log')
        print(f"Saving to {path_to_save}")
        print(f"Result of metrics: {result_dictionary}")
        save(path_to_save, result_dictionary)
        if refresh_metric:
            self.metrics = EnvMetrics()
        print(f"Metrics saving complete")

    def close(self):
        self.drawer.close_window()


class GraphicDrawer():
    def __init__(self, scale=10, frame_rate=2, window_w=1500, window_h=1500, show_trajectory=False, dataset='Waymo'):
        self.scale = scale
        self.win = None
        self.window_w = window_w
        self.window_h = window_h
        self.agents = []
        self.pred_agents = []
        self.show_trajectory = show_trajectory
        self.map_polys = []
        self.traffic_lights_polys = []
        self.offsets = 0, 0
        self.frame_rate = frame_rate
        self.map_inited = False
        self.fps = None
        self.edge_cache = None
        self.dataset = dataset

    def reset(self, scale=10, frame_rate=2, window_w=1500, window_h=1500, show_trajectory=False):
        self.scale = scale
        self.window_w = window_w
        self.window_h = window_h
        for agent in self.agents:
            for poly in agent.agent_polys:
                poly.undraw()
        self.agents = []
        for agent in self.pred_agents:
            if isinstance(agent, list):
                for poly in agent:
                    poly.undraw()
            else:
                for poly in agent.agent_polys:
                    poly.undraw()
        self.pred_agents = []
        self.show_trajectory = show_trajectory
        self.offsets = 0, 0
        self.frame_rate = frame_rate
        self.map_inited = False
        self.edge_cache = None

        for obj in self.map_polys:
            obj.undraw()
        self.map_polys = []
        for tl_polys in self.traffic_lights_polys:
            for poly in tl_polys:
                poly.undraw()
        self.traffic_lights_polys = []
        if self.fps is not None:
            for obj in self.fps:
                obj.undraw()
            # self.fps.undraw()
            self.fps = None

    def init_window(self):
        # print("window inited")
        self.win = GraphWin('DriveGym', self.window_h, self.window_w)
        self.win.setBackground("black")
        self.win.autoflush = False

    def close_window(self):
        self.win.autoflush = True
        # self.win.getMouse()
        self.win.close()

    def draw_fps(self, fps, message='', frame_idx=None):
        if self.fps is not None:
            for obj in self.fps:
                obj.undraw()
        self.fps = []
        message = Text(Point(self.window_w - 100, 100), "fps: " + str(fps) + '\n' + str(message))
        message.setTextColor("red")
        message.setSize(int(np.clip(self.scale * 1.5, 15, 35)))
        message.draw(self.win)
        self.fps.append(message)
        if frame_idx is not None:
            if frame_idx > 11:
                # info = 'Predicting'
                info = 'Planning'
                color = 'red'
            else:
                info = 'Given'
                color = 'blue'
            message2 = Text(Point(160, 100), info)
            message2.setTextColor(color)
            message2.setSize(int(np.clip(self.scale * 6, 5, 36)))
            message2.draw(self.win)
            self.fps.append(message2)

    def init_route_drawing(self, road_dic):
        road_type = int(road_dic["type"])
        points = road_dic["xyz"][:, :2]
        direction = road_dic["dir"]

        turning_lane = road_dic["turning"]
        outbound = road_dic["outbound"]
        marking = road_dic["marking"]

        self.draw_poly(points=points, color=color_rgb(129, 169, 140))

    def init_goal_drawing(self, goal):
        goal_xy = self.recenter((goal[0], goal[1]), self.offsets)
        goal[0] = goal_xy[0]
        goal[1] = goal_xy[1]
        color = 'red'
        pts = utils.generate_contour_pts_with_direction((goal[0], goal[1]),
                                                        2 * self.scale, 4 * self.scale,
                                                        -goal[3] - math.pi / 2)
        pt1, pt2, pt3, pt4, pt5 = pts
        agent_poly = Polygon(Point(pt1[0], pt1[1]),
                             Point(pt2[0], pt2[1]),
                             Point(pt3[0], pt3[1]),
                             Point(pt4[0], pt4[1]),
                             Point(pt5[0], pt5[1]))
        self.map_polys.append(agent_poly)
        mode = 'outline'
        if mode == 'fill':
            agent_poly.setFill(color)
        elif mode == 'outline':
            agent_poly.setOutline(color)
        agent_poly.draw(self.win)

    def init_map_drawing(self, road_dic, scenario_type=3, road_id=0):
        """
        LaneCenter-Freeway = 1,
        LaneCenter-SurfaceStreet = 2,
        LaneCenter-BikeLane = 3,
        RoadLine-BrokenSingleWhite = 6,
        RoadLine-SolidSingleWhite = 7,
        RoadLine-SolidDoubleWhite = 8,
        RoadLine-BrokenSingleYellow = 9,
        RoadLine-BrokenDoubleYellow = 10,
        Roadline-SolidSingleYellow = 11,
        Roadline-SolidDoubleYellow=12,
        RoadLine-PassingDoubleYellow = 13,
        RoadEdgeBoundary = 15,
        RoadEdgeMedian = 16, StopSign = 17, Crosswalk = 18, SpeedBump = 19,
        :return:
        """

        road_type = int(road_dic["type"])
        points = road_dic["xyz"][:, :2]
        direction = road_dic["dir"]

        turning_lane = road_dic["turning"]
        outbound = road_dic["outbound"]
        marking = road_dic["marking"]

        # if road_id == 90:
        #     marking = 1
        # self.draw_small_dashed(points=points, color="red")
        #     return

        if road_type in [1, 2]:
            if marking == 1:  # or road_id in [138, 92]:
                self.draw_small_dashed(points=points, color="red")
                return
            # if turning_lane == 2:
            #     self.draw_small_dashed(points=points, color="cyan")
            #     return
            # if turning_lane == 3:
            #     self.draw_small_dashed(points=points, color=color_rgb(245, 180, 35))  # orange
            #     return
            if marking == 4:
                self.draw_small_dashed(points=points, color=color_rgb(0, 73, 101))  # for testing marks only
                return
            if marking == 5:
                # outbound lane lines
                self.draw_small_dashed(points=points, color=color_rgb(179, 135, 4))  # for testing marks only
                return

        if self.dataset == 'Waymo':
            if road_id != 0 and DRAW_LANE_ID:
                self.draw_text(points=points[0], text=str(road_id), size=TEXT_SIZE)
            if 0 < road_type < 20:
                if road_type == 17:
                    self.draw_text(points[0], text='Stop', size=int(TEXT_SIZE * 1.7))
                elif road_type == 7:
                    self.draw_line(points)
                elif road_type == 1:
                    # brown for freeway
                    self.draw_small_dashed(points=points, color=color_rgb(232, 121, 51), directions=outbound)
                elif road_type == 2:
                    # lane lines center
                    self.draw_small_dashed(points=points, color=color_rgb(80, 0, 94),
                                           directions=outbound)  # color_rgb(220, 224, 230))
                elif road_type == 3:
                    self.draw_small_dashed(points=points, color="blue", directions=outbound)
                elif road_type == 6:
                    self.draw_dashed_line(points)
                elif road_type == 8:
                    self.draw_double_line(points=points, directions=direction)
                elif road_type == 9:
                    self.draw_dashed_line(points=points, color="yellow")
                elif road_type == 10:
                    # dark yellow for double dashed line
                    self.draw_dashed_line(points=points, color=color_rgb(235, 207, 111))
                elif road_type == 11:
                    self.draw_line(points=points, color="yellow")
                elif road_type == 12:
                    self.draw_double_line(points=points, directions=direction, color="yellow")
                elif road_type == 13:
                    # dark yellow for passing double yellow line
                    self.draw_double_line(points=points, directions=direction, color=color_rgb(235, 207, 111))
                elif road_type == 15:
                    # road curbs
                    self.draw_line(points=points, color="green")
                elif road_type == 16:
                    # dark green for median edge
                    self.draw_line(points=points, color=color_rgb(0, 235, 132))
                elif road_type == 17:
                    self.draw_text(points)
                elif road_type == 18:
                    self.draw_line(points=points, color="purple")
                elif road_type == 19:
                    self.draw_line(points=points, color="red")
                else:
                    print("Unknown line type: ", road_type)
        elif self.dataset == 'NuPlan':
            if road_id != 0 and DRAW_LANE_ID and road_type in [0, 11]:
                self.draw_text(points=points[0], text=str(road_id), size=TEXT_SIZE)
            if road_type == 17:
                self.draw_poly(points=points, color="gray")
            elif road_type == 18:
                self.draw_poly(points=points, color=color_rgb(132, 129, 169))  # light purple
            # if road_type == 17:
            #     self.draw_text(points[0], text='Stop', size=int(TEXT_SIZE*1.7))
            elif road_type == 0:
                # brown for freeway
                self.draw_small_dashed(points=points, color=color_rgb(232, 121, 51), directions=outbound)
            elif road_type == 11:
                # lane lines center
                self.draw_small_dashed(points=points, color=color_rgb(239, 206, 177),
                                       directions=outbound)  # color_rgb(220, 224, 230))
                # self.draw_small_dashed(points=points, color=color_rgb(80, 0, 94), directions=outbound)  # color_rgb(220, 224, 230))

    def draw_poly(self, points, color="white"):
        points_list = []
        for pt in points:
            points_list.append(self.list_to_points(pt))
        poly = Polygon(points_list)
        poly.setFill(color)
        poly.draw(self.win)
        self.map_polys.append(poly)

    def draw_line(self, points, color="white"):
        for idx, pt in enumerate(points):
            if idx == 0:
                continue
            if len(points.shape) < 2:
                continue
            line = Line(self.list_to_points(points[idx - 1]), self.list_to_points(points[idx]))
            line.setFill(color)
            line.draw(self.win)
            self.map_polys.append(line)

    def draw_dashed_line(self, points, color="white"):
        # dash 5 lines
        number = 2
        for idx, pt in enumerate(points):
            if idx == 0 or idx % 5 != 0:
                continue
            start_x = points[idx - 1][0]
            start_y = points[idx - 1][1]
            length_x = points[idx][0] - points[idx - 1][0]
            length_y = points[idx][1] - points[idx - 1][1]
            for i in range(number):
                pos_start = start_x + length_x / number * i, start_y + length_y / number * i
                pos_end = start_x + length_x / number * (i + 0.5), start_y + length_y / number * i
                line = Line(self.list_to_points(pos_start), self.list_to_points(pos_end))
                line.setFill(color)
                line.draw(self.win)
                self.map_polys.append(line)

    def draw_double_line(self, points, directions, color="white"):
        interval = 0.03 * self.scale
        if len(directions.shape) < 1:
            return
        for idx, pt in enumerate(points):
            if idx == 0:
                continue
            line = Line(self.list_to_points(points[idx - 1]), self.list_to_points(points[idx]))
            line.setFill(color)
            line.draw(self.win)
            direction = directions[idx]
            start_pt_b = [points[idx - 1][0] + interval, points[idx - 1][1] + interval]
            start_b = utils.rotate(origin=points[idx - 1], point=start_pt_b, angle=direction)
            end_pt_b = [points[idx][0] + interval, points[idx][1] + interval]
            end_b = utils.rotate(origin=points[idx], point=end_pt_b, angle=direction)
            line_b = Line(self.list_to_points(start_b), self.list_to_points(end_b))
            line_b.setFill(color)
            line_b.draw(self.win)
            self.map_polys.append(line)
            self.map_polys.append(line_b)

    def draw_crosswalk(self):
        # TODO: x
        pass

    def draw_small_dashed(self, points, color="white", size=0.03, directions=[]):
        if not DRAW_DASHED_LINES:
            self.draw_line(points, color)
            return
        # draw 2 arrows at 1/3, 2/3 position of each line
        # size 0.x of 1/3
        size = size * self.scale
        if len(points.shape) < 2:
            return
        for idx, pt in enumerate(points):
            if idx == 0:
                continue
            start_x = points[idx - 1][0]
            start_y = points[idx - 1][1]
            length_x = points[idx][0] - points[idx - 1][0]
            length_y = points[idx][1] - points[idx - 1][1]

            position_1_start = length_x / 3 * (1 - size) + start_x, length_y / 3 * (1 - size) + start_y
            position_1_end = length_x / 3 * (1 + size) + start_x, length_y / 3 * (1 + size) + start_y
            position_2_start = length_x / 3 * (2 - size) + start_x, length_y / 3 * (2 - size) + start_y
            position_2_end = length_x / 3 * (2 + size) + start_x, length_y / 3 * (2 + size) + start_y
            arrow_1 = Line(self.list_to_points(position_1_start), self.list_to_points(position_1_end))
            arrow_2 = Line(self.list_to_points(position_2_start), self.list_to_points(position_2_end))
            arrow_1.setFill(color)
            arrow_2.setFill(color)
            arrow_1.draw(self.win)
            arrow_2.draw(self.win)
            self.map_polys.append(arrow_1)
            self.map_polys.append(arrow_2)

        # if len(directions) > 0:
        #     for direction in directions:
        #         if direction == 1:
        #             # draw left arrow
        #             pt1 = points[-25]
        #             pt2 = points[-15]
        #             pt3 = [points[-15][0] - 1*self.scale, points[-15][1]]
        #             bar = Line(self.list_to_points(pt1), self.list_to_points(pt2))
        #             arrow = Line(self.list_to_points(pt2), self.list_to_points(pt3))
        #             bar.setFill("white")
        #             arrow.setFill("white")
        #             arrow.setArrow("first")
        #             bar.draw(self.win)
        #             arrow.draw(self.win)

    def draw_text(self, points, text="Stop", size=0):
        message = Text(self.list_to_points(points), text)
        message.setTextColor("red")
        message.draw(self.win)
        if size != 0:
            message.setSize(size)
        self.map_polys.append(message)

    def draw_relation_from_prediction(self, edges, color='red'):
        for edge in edges:
            agent_id1 = edge[0]
            agent_id2 = edge[1]
            weight = 1
            point_found = 0
            starting_pt = None
            ending_pt = None
            for agent in self.agents:
                if agent.agent_id == agent_id2:
                    starting_pt = (agent.x, agent.y)
                    point_found += 1
                if agent.agent_id == agent_id1:
                    ending_pt = (agent.x, agent.y)
                    point_found += 1
                if point_found > 1 and starting_pt is not None and ending_pt is not None:
                    # reactor to influencer
                    # swap: influencer to reactor
                    ending_pt, starting_pt = starting_pt, ending_pt
                    aLine = Line(Point(ending_pt[0], ending_pt[1]), Point(starting_pt[0], starting_pt[1]))
                    aLine.setArrow("first")
                    # aLine.setFill(color_rgb(int(255 * weight), int(35 * weight), int(35 * weight)))
                    aLine.setFill(color)
                    aLine.setWidth(0.3 * self.scale)
                    aLine.draw(self.win)
                    agent.agent_polys.append(aLine)
                    break

    def draw_one_arrow(self, starting_pt, ending_pt, weight, size=0.3, colorRGB=[255, 35, 35]):
        size = 0.5
        ending_pt, starting_pt = starting_pt, ending_pt
        aLine = Line(Point(ending_pt[0], ending_pt[1]), Point(starting_pt[0], starting_pt[1]))
        aLine.setArrow("first")
        aLine.setFill(color_rgb(int(colorRGB[0] * weight), int(colorRGB[1] * weight), int(colorRGB[2] * weight)))
        aLine.setWidth(size * self.scale)
        aLine.draw(self.win)
        return aLine

    def draw_relation(self, agents, edges, frame_idx):

        def draw_one_line(self_obj, edges):
            for edge in edges:
                agent_id1, agent_id2, _, weight = edge
                point_found = 0
                for agent in self_obj.agents:
                    if agent.agent_id == agent_id2:
                        starting_pt = (agent.x, agent.y)
                        point_found += 1
                    if agent.agent_id == agent_id1:
                        ending_pt = (agent.x, agent.y)
                        point_found += 1
                    if point_found > 1:
                        # reactor to influencer
                        # swap: influencer to reactor
                        aLine = self_obj.draw_one_arrow(starting_pt, ending_pt, weight)
                        # ending_pt, starting_pt = starting_pt, ending_pt
                        # aLine = Line(Point(ending_pt[0], ending_pt[1]), Point(starting_pt[0], starting_pt[1]))
                        # aLine.setArrow("first")
                        # aLine.setFill(color_rgb(int(255*weight), int(35*weight), int(35*weight)))
                        # aLine.setWidth(0.3*self.scale)
                        # aLine.draw(self.win)
                        agent.agent_polys.append(aLine)
                        break

        if isinstance(edges, list):
            draw_one_line(self, edges)
            # single framge edge
            # for edge in edges:
            #     agent_id1, agent_id2, _, weight = edge
            #     point_found = 0
            #     for agent in self.agents:
            #         if agent.agent_id == agent_id2:
            #             starting_pt = (agent.x, agent.y)
            #             point_found += 1
            #         if agent.agent_id == agent_id1:
            #             ending_pt = (agent.x, agent.y)
            #             point_found += 1
            #         if point_found > 1:
            #             aLine = Line(Point(ending_pt[0], ending_pt[1]), Point(starting_pt[0], starting_pt[1]))
            #             aLine.setArrow("first")
            #             aLine.setFill(color_rgb(int(255*weight), int(15*weight), int(15*weight)))
            #             aLine.setWidth(0.3*self.scale)
            #             aLine.draw(self.win)
            #             agent.agent_polys.append(aLine)
            #             break
        elif isinstance(edges, dict):
            if frame_idx in edges:
                draw_one_line(self, edges[frame_idx])
                self.edge_cache = edges[frame_idx]
            elif self.edge_cache is not None:
                draw_one_line(self, self.edge_cache)
        else:
            assert False, f'unknown type of edges loaded: {edges}'

    def draw_agent_on_window_by_frame(self, agents, frame_idx, running_mode=0, ego_id=None):
        for agent in self.agents:
            agent_id = agent.agent_id
            agent_dic = agents[agent_id]
            # if agent_id != 564:
            #     continue
            if "marking" in agent_dic:
                marking = agent_dic["marking"]
            else:
                marking = False

            if ego_id is None:
                if self.dataset == 'Waymo':
                    predict = agent_dic["to_predict"]
                    if predict == 0:
                        color = "green"
                    else:
                        sdc = agent_dic["is_sdc"]
                        if sdc:
                            color = "grey"
                        else:
                            color = "blue"
                elif self.dataset == "NuPlan":
                    sdc = agent_dic["is_sdc"]
                    if sdc:
                        predict = 1
                        color = "grey"
                    else:
                        predict = 0
                        color = "green"
            else:
                predict = 0
                if agent_id == ego_id:
                    # color = "pink"
                    color = color_rgb(10, 255, 255)
                else:
                    color = "green"
            if agent.crashed:
                color = "red"

            if 'action' in agent_dic and agent_id != ego_id:
                if agent_dic['action'] == 'stop':
                    color = 'yellow'
                elif agent_dic['action'] == 'yield':
                    color = 'purple'
                elif agent_dic['action'] == 'controlled':
                    color = color_rgb(204, 255, 51)

            # draw
            agent_w = agent.width
            agent_l = agent.length
            self.draw_rectangle(pts=utils.generate_contour_pts_with_direction((agent.x, agent.y),
                                                                              agent_w, agent_l, agent.yaw),
                                color=color, agent=agent)
            # pt1, pt2, pt3, pt4 = generate_contour_pts((new_agent.x, new_agent.y), agent_w, agent_l, new_agent.yaw)
            # pt1, pt2, pt3, pt4, pt5 = utils.generate_contour_pts_with_direction((agent.x, agent.y), agent_w, agent_l,
            #                                                                     agent.yaw)
            # agent_poly = Polygon(Point(pt1[0], pt1[1]),
            #                      Point(pt2[0], pt2[1]),
            #                      Point(pt3[0], pt3[1]),
            #                      Point(pt4[0], pt4[1]),
            #                      Point(pt5[0], pt5[1]))
            #
            # agent.agent_polys.append(agent_poly)
            # agent_poly.setFill(color)
            # agent_poly.draw(self.win)
            if DRAW_AGENT_ARROW:
                x, y = pt2
                arrow_end_x, arrow_end_y = utils.rotate((x, y), (x, y - 5 * self.scale),
                                                        utils.normalize_angle(agent.yaw))
                aLine = Line(Point(arrow_end_x, arrow_end_y), Point(x, y))
                aLine.setArrow("first")
                aLine.setFill("white")
                aLine.draw(self.win)
                agent.agent_polys.append(aLine)
            if DRAW_AGENT_ID:
                text_to_draw = str(agent_id)
                if running_mode == 0 and marking and predict == 0:
                    if marking == 'Parking':
                        text_to_draw = f'{text_to_draw}(P)'

                if running_mode == 0 and 'current_mode' in agent_dic and predict != 0:
                    if agent_dic['current_mode'] is None:
                        text_to_draw = f'{text_to_draw}'
                    else:
                        current_mode = agent_dic['current_mode']
                        if current_mode == 0:
                            text_to_draw = f'{text_to_draw}(F)'
                        elif current_mode == 1:
                            text_to_draw = f'{text_to_draw}(R)'
                        elif current_mode == 2:
                            text_to_draw = f'{text_to_draw}(L)'
                        elif current_mode == 3:
                            text_to_draw = f'{text_to_draw}(S)'

                if running_mode == 1 and frame_idx > 10:
                    text_to_draw = "(P)" + text_to_draw
                aText = Text(Point(agent.x, agent.y - 1 * self.scale), text_to_draw)
                aText.setFill("white")
                aText.draw(self.win)
                aText.setSize(TEXT_SIZE)
                agent.agent_polys.append(aText)

    def draw_rectangle(self, pts, color, agent, mode='fill'):
        pt1, pt2, pt3, pt4, pt5 = pts
        agent_poly = Polygon(Point(pt1[0], pt1[1]),
                             Point(pt2[0], pt2[1]),
                             Point(pt3[0], pt3[1]),
                             Point(pt4[0], pt4[1]),
                             Point(pt5[0], pt5[1]))
        agent.agent_polys.append(agent_poly)
        if mode == 'fill':
            agent_poly.setFill(color)
        elif mode == 'outline':
            agent_poly.setOutline(color)
        agent_poly.draw(self.win)

    def draw_prediction_on_window_by_frame(self, prediction_dic, original_agents, frame_idx):
        for agent in self.pred_agents:
            for poly in agent.agent_polys:
                poly.undraw()
        self.pred_agents = []
        for idx, agent_id in enumerate(prediction_dic.keys()):
            agent_pos = prediction_dic[agent_id]["pred_trajectory"]
            agent_yaw = prediction_dic[agent_id]["pred_yaw"]
            org_agent = original_agents[agent_id]
            if 'pred_scores' in prediction_dic[agent_id].keys():
                # multi predictions
                for pred_index in range(agent_pos.shape[0]):
                    single_pos = agent_pos[pred_index]
                    single_yaw = agent_yaw[pred_index]
                    single_score = prediction_dic[agent_id]['pred_scores'][pred_index]
                    single_score = np.clip(single_score * 4, 0.001, 1)
                    color = color_rgb(int(15 + single_score * (255 - 15)), int(15 + single_score * (255 - 15)), 0)
                    self.draw_one_prediction(agent_pose=single_pos,
                                             agent_yaw=single_yaw,
                                             org_agent=org_agent,
                                             frame_idx=frame_idx,
                                             agent_id=agent_id,
                                             color=color)
            else:
                # single predictions
                self.draw_one_prediction(agent_pose=agent_pos,
                                         agent_yaw=agent_yaw,
                                         org_agent=org_agent,
                                         frame_idx=frame_idx,
                                         agent_id=agent_id)

    def draw_mark_trajectory_on_window_for_pred(self, trajectory_list):
        if not self.show_trajectory:
            return
        # pred_agents_cleared = False
        for agent in self.pred_agents:
            for poly in agent:
                poly.undraw()
        self.pred_agents = []
        for idx, trajectory in enumerate(trajectory_list):
            self.draw_one_prediction_with_pts(trajectory[:, :2], color_seed=idx)

    def draw_prediction_on_window_for_pred(self, prediction_dic):
        pred_agents_cleared = False
        for idx, agent_id in enumerate(prediction_dic.keys()):
            if not pred_agents_cleared:
                for agent in self.pred_agents:
                    for poly in agent:
                        poly.undraw()
                self.pred_agents = []
                pred_agents_cleared = True

            agent_pos = prediction_dic[agent_id]["rst"]
            # agent_yaw = prediction_dic[agent_id]["pred_yaw"]
            if 'score' in prediction_dic[agent_id].keys():
                # multi predictions
                for pred_index in range(agent_pos.shape[0]):
                    if pred_index > 0:
                        # only draw one prediction
                        break
                    single_pos = agent_pos[pred_index]
                    # single_yaw = agent_yaw[pred_index]
                    single_score = np.exp(prediction_dic[agent_id]['score'][pred_index])
                    single_score = np.clip(single_score * 4, 0.001, 1)
                    # if single_score < 0.3:
                    #     continue

                    # if idx == 0:
                    #     color = color_rgb(int(30+single_score*(255-30)), int(30+single_score*(255-30)), 0)
                    # else:
                    #     color = color_rgb(0, int(30 + single_score * (255 - 30)), int(30 + single_score * (255 - 30)))
                    self.draw_one_prediction_with_pts(agent_pose=single_pos, color_seed=idx)
            else:
                # single predictions
                self.draw_one_prediction_with_pts(agent_pose=agent_pos)

    def draw_prediction_on_window_by_timeoffset(self, prediction_dic, original_agents, frame_idx):
        pred_agents_cleared = False
        for agent in self.pred_agents:
            for poly in agent:
                poly.undraw()
        self.pred_agents = []
        for idx, agent_id in enumerate(prediction_dic.keys()):
            if frame_idx not in prediction_dic[agent_id]:
                continue
            # if not pred_agents_cleared:
            #     for agent in self.pred_agents:
            #         for poly in agent:
            #             poly.undraw()
            #     self.pred_agents = []
            #     pred_agents_cleared = True

            agent_pos = prediction_dic[agent_id][frame_idx]["pred_trajectory"]
            agent_yaw = prediction_dic[agent_id][frame_idx]["pred_yaw"]
            if 'pred_scores' in prediction_dic[agent_id][frame_idx].keys():
                # multi predictions
                for pred_index in range(agent_pos.shape[0]):
                    if pred_index > 0:
                        # only draw one prediction
                        break
                    single_pos = agent_pos[pred_index]
                    single_yaw = agent_yaw[pred_index]
                    single_score = prediction_dic[agent_id][frame_idx]['pred_scores'][pred_index]
                    single_score = np.clip(single_score * 4, 0.001, 1)
                    if single_score < 0.3:
                        continue

                    # if idx == 0:
                    #     color = color_rgb(int(30+single_score*(255-30)), int(30+single_score*(255-30)), 0)
                    # else:
                    #     color = color_rgb(0, int(30 + single_score * (255 - 30)), int(30 + single_score * (255 - 30)))
                    self.draw_one_prediction_with_pts(agent_pose=single_pos, color_seed=idx)
            else:
                # single predictions
                self.draw_one_prediction_with_pts(agent_pose=agent_pos)

    def draw_one_prediction_with_pts(self, agent_pose, color_seed=-1, score=1):
        # if agent_id != 564:
        #     continue
        offsets = self.offsets
        pts_group = []
        interval = 5
        for frame_idx_in_pred, each_pose in enumerate(agent_pose):
            if frame_idx_in_pred % interval != 0:
                continue
            if isinstance(each_pose, list):
                if len(each_pose) != 2:
                    print("invalid point to draw: ", each_pose)
                    continue
            else:
                if each_pose.shape[0] != 2:
                    print("invalid np array to draw: ", each_pose)
                    continue
            x, y = each_pose
            if x == -1.0 or y == -1.0:
                continue

            recentered_xy = self.recenter((x, y), offsets)
            diameter = 0.5 * self.scale
            # p1 = Point(recentered_xy[0] - diameter, recentered_xy[1] - diameter)
            # p2 = Point(recentered_xy[0] + diameter, recentered_xy[1] + diameter)
            # aOval = Oval(p1, p2)
            if color_seed == -1:
                diameter *= 2
                # elif color_seed == 0:
                #     color = color_rgb(int((255 - frame_idx_in_pred*1.8)*score), int((255-frame_idx_in_pred*1.8)*score), 0)
                # elif color_seed >= 1:
                #     color = color_rgb(0, int(255 - frame_idx_in_pred * 1.8), int(255 - frame_idx_in_pred * 1.8))
                aCircle = Circle(Point(recentered_xy[0], recentered_xy[1]), diameter)
                aCircle.setFill('red')
                aCircle.draw(self.win)
                pts_group.append(aCircle)
            else:
                color_index = color_seed % len(PREDICTION_COLORS)
                color = PREDICTION_COLORS[color_index]
                # elif color_seed == 0:
                #     color = color_rgb(int((255 - frame_idx_in_pred*1.8)*score), int((255-frame_idx_in_pred*1.8)*score), 0)
                # elif color_seed >= 1:
                #     color = color_rgb(0, int(255 - frame_idx_in_pred * 1.8), int(255 - frame_idx_in_pred * 1.8))
                aCircle = Circle(Point(recentered_xy[0], recentered_xy[1]), diameter)
                aCircle.setFill(color)
                aCircle.draw(self.win)
                pts_group.append(aCircle)
        self.pred_agents.append(pts_group)

    def draw_one_pt_single_frame(self, pt, color_seed=-1):
        x, y = pt
        if x == -1.0 or y == -1.0:
            return
        pts_group = []
        recentered_xy = self.recenter((x, y), self.offsets)
        diameter = 0.5 * self.scale
        # p1 = Point(recentered_xy[0] - diameter, recentered_xy[1] - diameter)
        # p2 = Point(recentered_xy[0] + diameter, recentered_xy[1] + diameter)
        # aOval = Oval(p1, p2)
        if color_seed == -1:
            diameter *= 2
            # elif color_seed == 0:
            #     color = color_rgb(int((255 - frame_idx_in_pred*1.8)*score), int((255-frame_idx_in_pred*1.8)*score), 0)
            # elif color_seed >= 1:
            #     color = color_rgb(0, int(255 - frame_idx_in_pred * 1.8), int(255 - frame_idx_in_pred * 1.8))
            aCircle = Circle(Point(recentered_xy[0], recentered_xy[1]), diameter)
            aCircle.setFill('red')
            aCircle.draw(self.win)
            pts_group.append(aCircle)
        else:
            color_index = color_seed % len(PREDICTION_COLORS)
            color = PREDICTION_COLORS[color_index]
            # elif color_seed == 0:
            #     color = color_rgb(int((255 - frame_idx_in_pred*1.8)*score), int((255-frame_idx_in_pred*1.8)*score), 0)
            # elif color_seed >= 1:
            #     color = color_rgb(0, int(255 - frame_idx_in_pred * 1.8), int(255 - frame_idx_in_pred * 1.8))
            aCircle = Circle(Point(recentered_xy[0], recentered_xy[1]), diameter)
            aCircle.setFill(color)
            aCircle.draw(self.win)
            pts_group.append(aCircle)
        self.pred_agents.append(pts_group)

    def draw_one_prediction(self, agent_pose, agent_yaw, org_agent, frame_idx, agent_id, color='yellow'):
        # if agent_id != 564:
        #     continue
        x, y = agent_pose[frame_idx]
        yaw = agent_yaw[frame_idx]
        if x == -1.0 or y == -1.0:
            return

        yaw = utils.normalize_angle(yaw + math.pi / 2)
        width, length, _ = org_agent["shape"][0]

        offsets = self.offsets
        recentered_xy = self.recenter((x, y), offsets)

        new_agent = Agent(x=recentered_xy[0],
                          y=recentered_xy[1], yaw=-yaw,
                          vx=0, length=length * self.scale, width=width * self.scale)
        self.pred_agents.append(new_agent)
        # draw
        agent_w = new_agent.width
        agent_l = new_agent.length
        # pt1, pt2, pt3, pt4 = generate_contour_pts((new_agent.x, new_agent.y), agent_w, agent_l, new_agent.yaw)
        pt1, pt2, pt3, pt4, pt5 = utils.generate_contour_pts_with_direction((new_agent.x, new_agent.y), agent_w,
                                                                            agent_l, new_agent.yaw)
        contour_line_pts = [[Point(pt1[0], pt1[1]), Point(pt2[0], pt2[1])],
                            [Point(pt2[0], pt2[1]), Point(pt3[0], pt3[1])],
                            [Point(pt3[0], pt3[1]), Point(pt4[0], pt4[1])],
                            [Point(pt4[0], pt4[1]), Point(pt5[0], pt5[1])],
                            [Point(pt5[0], pt5[1]), Point(pt1[0], pt1[1])], ]
        for pts in contour_line_pts:
            aLine = Line(pts[0], pts[1])
            aLine.setFill(color)
            aLine.draw(self.win)
            new_agent.agent_polys.append(aLine)

        if DRAW_AGENT_ID:
            x, y = pt2
            # arrow_end_x, arrow_end_y = rotate((x, y), (x, y - 5 * self.scale),
            #                                   normalize_angle(new_agent.yaw))
            # aLine = Line(Point(arrow_end_x, arrow_end_y), Point(x, y))
            # aLine.setArrow("first")
            # aLine.setFill("white")
            # aLine.draw(self.win)
            # new_agent.agent_polys.append(aLine)

            aText = Text(Point(new_agent.x, new_agent.y - 1 * self.scale), "(P)" + str(agent_id))
            aText.setFill("white")
            aText.draw(self.win)
            aText.setSize(TEXT_SIZE)
            new_agent.agent_polys.append(aText)

    def recenter(self, pt, offsets):
        if (isinstance(pt, list) and len(pt) == 2) or (type(pt) is type(np.array([])) and pt.shape[0] == 2):
            x, y = pt
            pt = x, y
        if isinstance(pt, tuple):
            x, y = pt
            return - (x + offsets[0]) * self.scale + self.window_w / 2, (
                    y + offsets[1]) * self.scale + self.window_h / 2
        else:
            return 0, 0

    def list_to_points(self, pt_list):
        x, y = self.recenter(pt_list, self.offsets)
        return Point(x, y)

    def draw_traffic_light(self, tl_dics, road_dics, frame_idx):
        for tl_polys in self.traffic_lights_polys:
            for poly in tl_polys:
                poly.undraw()
        self.traffic_lights_polys = []
        for lane_id in tl_dics.keys():
            if lane_id == -1:
                continue
            tl = tl_dics[lane_id]
            # get the position of the end of this lane
            tl_state = tl["state"][frame_idx]

            found_lane_in_road_dic = False
            for seg_id in road_dics.keys():
                if lane_id == seg_id:
                    found_lane_in_road_dic = True
                    road_seg = road_dics[seg_id]
                    if self.dataset == 'Waymo':
                        if road_seg["type"] in [1, 2, 3]:
                            if len(road_seg["dir"].shape) < 1:
                                continue
                            end_point = road_seg["xyz"][0][:2]
                            direction = road_seg["dir"][0]
                            # draw traffic light
                            poly_list = self.draw_tl_on_window(position=end_point,
                                                               direction=direction,
                                                               status=tl_state,
                                                               size=2)
                            self.traffic_lights_polys.append(poly_list)

                        elif road_seg["type"] == 18:
                            print("traffic light for crosswalk, ignoring")
                        else:
                            print("WARNING: traffic light for unknown type lane - ", str(road_seg["type"]))
                        break
                    elif self.dataset == 'NuPlan':
                        end_point = road_seg["xyz"][0][:2]
                        direction = road_seg["dir"][0]
                        # draw traffic light
                        poly_list = self.draw_tl_on_window(position=end_point,
                                                           direction=direction,
                                                           status=tl_state,
                                                           size=2)
                        self.traffic_lights_polys.append(poly_list)

            if not found_lane_in_road_dic:
                print("WARNING: lane_id not found - ", str(lane_id))

    def draw_tl_on_window(self, position, direction, status, size=20):
        # init polygons to shape list
        # for each traffic light
        # Unknown = 0, Arrow_Stop = 1, Arrow_Caution = 2, Arrow_Go = 3, Stop = 4, Caution = 5, Go = 6, Flashing_Stop = 7, Flashing_Caution = 8
        size = size * self.scale
        pc_x, pc_y = position
        offsets = self.offsets
        pc_x, pc_y = self.recenter((pc_x, pc_y), offsets)
        direction = utils.normalize_angle(direction)
        if status in [7, 8]:
            # use circle to indicate flashing
            an_oval = Circle(Point(pc_x, pc_y), size)
        else:
            an_oval = Oval(Point(pc_x - size / 2, pc_y - size / 2),
                           Point(pc_x + size / 2, pc_y + size / 2))
        # facing to the left as 0
        left_upper = utils.rotate(origin=(pc_x, pc_y),
                                  point=(pc_x, pc_y - size / 2),
                                  angle=direction,
                                  tuple=True)
        right_upper = utils.rotate(origin=(pc_x, pc_y),
                                   point=(pc_x + size / 2, pc_y - size / 2),
                                   angle=direction,
                                   tuple=True)
        right_lower = utils.rotate(origin=(pc_x, pc_y),
                                   point=(pc_x + size / 2, pc_y + size / 2),
                                   angle=direction,
                                   tuple=True)
        left_lower = utils.rotate(origin=(pc_x, pc_y),
                                  point=(pc_x, pc_y + size / 2),
                                  angle=direction,
                                  tuple=True)
        a_polygon = Polygon(Point(left_upper[0], left_upper[1]),
                            Point(right_upper[0], right_upper[1]),
                            Point(right_lower[0], right_lower[1]),
                            Point(left_lower[0], left_lower[1]))
        left_upper_block = utils.rotate(origin=(pc_x, pc_y),
                                        point=(pc_x - size / 2, pc_y),
                                        angle=direction,
                                        tuple=True)
        right_upper_block = (pc_x, pc_y)
        right_lower_block = left_lower
        left_lower_block = utils.rotate(origin=(pc_x, pc_y),
                                        point=(pc_x - size / 2, pc_y + size / 2),
                                        angle=direction,
                                        tuple=True)
        a_right_block = Polygon(Point(left_upper_block[0], left_upper_block[1]),
                                Point(right_upper_block[0], right_upper_block[1]),
                                Point(right_lower_block[0], right_lower_block[1]),
                                Point(left_lower_block[0], left_lower_block[1]))

        a_right_block.setFill("black")

        if status in [1, 4, 7]:
            an_oval.setFill("red")
        elif status in [3, 6]:
            an_oval.setFill("green")
        elif status in [2, 5, 8]:
            an_oval.setFill("yellow")
        elif status in [0, -1]:
            an_oval.setFill("white")
        else:
            print("ERROR: unknown traffic light color: ", status)

        a_polygon.setFill(color_rgb(130, 130, 130))
        an_oval.draw(self.win)
        a_polygon.draw(self.win)
        poly_list = [an_oval, a_polygon]
        if status in [1, 2, 3]:
            a_right_block.draw(self.win)
            poly_list.append(a_right_block)

        return poly_list

    def draw_route(self, data_dic, agent_id):
        if 'route' not in data_dic['predicting']:
            return
        if agent_id not in data_dic['predicting']['route']:
            return
        route_lane_ids = data_dic['predicting']['route'][agent_id]
        print("Drawing route for ego: ", route_lane_ids)
        current_accu_dist = 0
        current_pt_xy = None
        arrow_length = 20
        for each_lane in route_lane_ids:
            lane_xy = data_dic['road'][each_lane]['xyz']
            for pt_idx, each_pt in enumerate(lane_xy):
                if current_pt_xy is None:
                    current_pt_xy = each_pt
                else:
                    if pt_idx == 0:
                        continue
                    dist = utils.euclidean_distance(current_pt_xy[:2], each_pt[:2])
                    current_accu_dist += dist
                    if current_accu_dist > arrow_length:
                        poly = self.draw_one_arrow(starting_pt=self.recenter(each_pt[:2], self.offsets),
                                                   ending_pt=self.recenter(current_pt_xy[:2], self.offsets),
                                                   weight=1,
                                                   size=0.3,
                                                   colorRGB=[156, 175, 183])
                        current_pt_xy = each_pt
                        current_accu_dist = 0
                        self.map_polys.append(poly)


class Agent:
    def __init__(self,
                 # init location, angle, velocity
                 x=0.0, y=0.0, yaw=0.0, vx=0.01, vy=0, length=4.726, width=1.842, agent_id=None, color=None):
        self.x = x  # px
        self.y = y
        self.yaw = yaw
        self.vx = vx  # px/frame
        self.vy = vy
        self.length = length  # px
        self.width = width  # px
        self.agent_polys = []
        self.crashed = False
        self.agent_id = agent_id
        self.color = color


def dictionary_to_state(dic, frame_index=0):
    return dic.copy()


def load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def save(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
