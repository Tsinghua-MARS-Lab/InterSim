import interactive_sim.envs.util as util
import numpy as np
import math

class EnvMetrics:
    def __init__(self):
        self.past = 0
        self.get_stuck = 0
        # self.passiveness = 0
        self.progress = 0  # in total
        self.progress_others = 0  # in total
        self.front_collisions = 0
        self.side_collisions = 0
        self.rear_collisions = 0
        self.colliding_scenarios = []
        self.running_redlights = 0
        self.jerk = 0  # in total
        self.jerk_others = 0  # in total
        self.jerk_ego = 0
        self.total_agents_controlled = 0
        self.total_agents_others = 0
        self.total_scenarios = 0
        self.fde = 0  # in total
        self.ade = 0  # in total
        self.goal_fit = 0.0001

        self.flip_relation = 0
        self.engage_with_false_r = 0
        self.ego_progress = 0

        self.emergency_stops = 0

        self.gpu_relation_predictions = 0

        self.max_jerk_distribution = {'0-0.01': 0,
                                      '0.01-0.1': 0,
                                      '0.1-0.2': 0,
                                      '0.2-0.5': 0,
                                      '0.5-1': 0,
                                      '1-2': 0,
                                      '2-5': 0,
                                      '>5': 0}

        self.ego_progress_distribution = {'0-5': [0, 0],
                                          '5-10':  [0, 0],
                                          '10-50':  [0, 0],
                                          '50-100':  [0, 0],
                                          '100-200':  [0, 0],
                                          '200-500':  [0, 0],
                                          '>500':  [0, 0]}

        self.offroad_scenarios = 0


    def update_per_scenario(self, data_dic, collisions, include_ego=False, current_frame=-1):
        flipping_relations = {}  # return a list of all flipping relations
        engage_with_false_r = {}  # return a list of all false relations

        metrics_dic = {
            'collided_pairs': [],
            'flipping_relations': [],
            'engage_with_false_r': [],
            'progress': []
        }

        progress = None
        if 'predicting' in data_dic:
            relevant_agents = data_dic['predicting']['relevant_agents']
            goal_fit = data_dic['predicting']['goal_fit']

            controlled_agents_per_scenario = 0
            ego_agent = data_dic['predicting']['ego_id'][1]
            if include_ego and ego_agent not in relevant_agents:
                relevant_agents.append(ego_agent)

            for each_agent in data_dic['agent']:
                if not include_ego and each_agent == ego_agent:
                    continue
                pose = data_dic['agent'][each_agent]['pose']

                v = []
                ade_per_agent = []
                for i in range(90):
                    if i <= 11 or i >= 80:
                        continue
                    if i >= current_frame:
                        # terminate early
                        continue
                    if pose[i, 0] != -1 and pose[i+1, 0] != -1:
                        dist = util.euclidean_distance(pose[i, :2], pose[i + 1, :2])
                        if dist < 20:
                            # filter impossible outliers
                            v.append(dist)
                        else:
                            continue
                    else:
                        break

                    # if each_agent in relevant_agents or each_agent == ego_agent:
                    if each_agent == ego_agent:
                        # calculate ade
                        origial_pose = data_dic['predicting']['original_trajectory'][each_agent]['pose']
                        if pose[i, 0] != -1 and pose[i + 1, 0] != -1 and origial_pose[i, 0] != -1 and origial_pose[i, 1] != -1:
                            diff = util.euclidean_distance(origial_pose[i, :2], pose[i, :2])
                            if diff < 400:
                                ade_per_agent.append(diff)
                if len(v) < 5:
                    progress = sum(abs(np.array(v)))
                    max_jerk = None
                    if each_agent in relevant_agents or each_agent == ego_agent:
                        self.total_agents_controlled += 1
                        controlled_agents_per_scenario += 1
                        self.progress += progress
                        if each_agent == ego_agent:
                            if goal_fit is not None and goal_fit:
                                self.goal_fit += 1
                                self.ego_progress += progress
                                metrics_dic['progress'] = progress
                    else:
                        self.total_agents_others += 1
                        self.progress_others += progress
                else:
                    total_v = len(v)
                    v_dot = np.array(v[1:]) - np.array(v[:total_v-1])
                    v_dot_dot = abs(v_dot[1:] - v_dot[:total_v-2])
                    progress = sum(abs(np.array(v)))
                    max_jerk = np.max(v_dot_dot) * 10

                    if each_agent in relevant_agents or each_agent == ego_agent:
                        self.total_agents_controlled += 1
                        controlled_agents_per_scenario += 1
                        if max_jerk > 0.01:
                            # filter stopping scenarios
                            self.jerk += np.average(v_dot_dot) * 10
                        self.progress += progress
                        metrics_dic['progress'] = progress
                    else:
                        if np.max(v_dot_dot) > 0.01:
                            self.jerk_others += np.average(v_dot_dot) * 10
                        self.total_agents_others += 1
                        self.progress_others += sum(abs(np.array(v)))

                    if each_agent == ego_agent:
                        # calculate ego jerk
                        if max_jerk > 0.01:
                            # filter stopping scenarios
                            self.jerk_ego += np.average(v_dot_dot) * 10

                        if max_jerk < 0.01:
                            self.max_jerk_distribution['0-0.01'] += 1
                        elif 0.01 <= max_jerk < 0.1:
                            self.max_jerk_distribution['0.01-0.1'] += 1
                        elif 0.1 <= max_jerk < 0.2:
                            self.max_jerk_distribution['0.1-0.2'] += 1
                        elif 0.2 <= max_jerk < 0.5:
                            self.max_jerk_distribution['0.2-0.5'] += 1
                        elif 0.5 <= max_jerk < 1:
                            self.max_jerk_distribution['0.5-1'] += 1
                        elif 1 <= max_jerk < 2:
                            self.max_jerk_distribution['1-2'] += 1
                        elif 2 <= max_jerk < 5:
                            self.max_jerk_distribution['2-5'] += 1
                        elif 5 <= max_jerk:
                            self.max_jerk_distribution['>5'] += 1

                        # calculate progress
                        if goal_fit is not None and goal_fit:
                            self.goal_fit += 1
                            print("Fitted, Calculating ADE/FDE")
                            if len(ade_per_agent) > 0:
                                self.ade += np.average(ade_per_agent)
                            # calculate goal point reaching
                            current_pose = data_dic['agent'][each_agent]['pose'][current_frame, :2]
                            origial_goal = data_dic['predicting']['original_trajectory'][each_agent]['pose'][80, :2]
                            # calculate fde
                            fde = util.euclidean_distance(origial_goal, current_pose)
                            if fde < 400:
                                self.fde += fde
                            # check if past the goal point
                            angle_to_goal = util.get_angle_of_a_line(current_pose, origial_goal)
                            goal_yaw = data_dic['predicting']['original_trajectory'][each_agent]['pose'][-6, 3]
                            normalized_angle = util.normalize_angle(angle_to_goal - goal_yaw)
                            if normalized_angle > math.pi/2 or normalized_angle < -math.pi/2 or fde < 3:
                                # past goal point
                                self.past += 1
                            else:
                                # not past goal point
                                org_last_v = util.euclidean_distance(data_dic['predicting']['original_trajectory'][each_agent]['pose'][-6, :2],
                                                                     data_dic['predicting']['original_trajectory'][each_agent]['pose'][-6-15, :2])/5
                                if sum(v[-15:]) < 0.05 and org_last_v > 0.05:
                                    self.get_stuck += 1
                                # if util.euclidean_distance(origial_goal, current_pose) > 5:
                                #     self.passiveness += 1

                            self.ego_progress += progress
                            metrics_dic['progress'] = progress

                            if progress < 5:
                                self.ego_progress_distribution['0-5'] = [self.ego_progress_distribution['0-5'][0]+progress, self.ego_progress_distribution['0-5'][1]+1]
                            elif 5 <= progress < 10:
                                self.ego_progress_distribution['5-10'] = [self.ego_progress_distribution['5-10'][0]+progress, self.ego_progress_distribution['5-10'][1]+1]
                            elif 10 <= progress < 50:
                                self.ego_progress_distribution['10-50'] = [self.ego_progress_distribution['10-50'][0]+progress, self.ego_progress_distribution['10-50'][1]+1]
                            elif 50 <= progress < 100:
                                self.ego_progress_distribution['50-100'] = [self.ego_progress_distribution['50-100'][0]+progress, self.ego_progress_distribution['50-100'][1]+1]
                            elif 100 <= progress < 200:
                                self.ego_progress_distribution['100-200'] = [self.ego_progress_distribution['100-200'][0]+progress, self.ego_progress_distribution['100-200'][1]+1]
                            elif 200 <= progress < 500:
                                self.ego_progress_distribution['200-500'] = [self.ego_progress_distribution['200-500'][0]+progress, self.ego_progress_distribution['200-500'][1]+1]
                            elif progress >= 500:
                                self.ego_progress_distribution['>500'] = [self.ego_progress_distribution['>500'][0]+progress, self.ego_progress_distribution['>500'][1]+1]

            # detect flipping relations
            gt_relations_raw = data_dic['edges']  # [[1871, 1885, 0, 1], ...]
            gt_relations = []
            for each_r in data_dic['edges']:
                gt_relations.append([each_r[0], each_r[1]])
            if 'relations_per_frame_ego' in data_dic['predicting']:
                if 'scenario_str' in data_dic:
                    scenario_str = data_dic['scenario_str']
                else:
                    scenario_str = data_dic['scenario']
                relations_pred_all = data_dic['predicting']['relations_per_frame_ego']
                all_r_pred = []
                for each_frame in relations_pred_all:
                    relations_pred = relations_pred_all[each_frame]
                    for each_pair_r in relations_pred:
                        inf = each_pair_r[0]
                        reactor = each_pair_r[1]
                        if [reactor, inf] in all_r_pred and [inf, reactor] not in all_r_pred and [reactor,
                                                                                                  inf] not in relations_pred:
                            # reverse relation in past prediction but not in current prediction
                            # bi-directional relation prediction does not count as a flip
                            if scenario_str not in flipping_relations:
                                # flipping_relations[scenario_str] = []
                                self.flip_relation += 1
                            if [inf, reactor] not in flipping_relations[scenario_str]:
                                metrics_dic['flipping_relations'].append([inf, reactor])
                                # flipping_relations[scenario_str].append([inf, reactor])
                        if each_pair_r not in all_r_pred:
                            all_r_pred.append(each_pair_r)

                        # detect engage due to wrong prediction
                        # if reactor in relevant and [reactor, inf] in gt relations -> engage due to wrong prediction
                        if inf == ego_agent and reactor in relevant_agents and [reactor, inf] in gt_relations:
                            # engage_with_false_r: {scenario_id_str: [[inf, reactor], ...]}
                            if scenario_str not in engage_with_false_r:
                                # engage_with_false_r[scenario_str] = []
                                self.engage_with_false_r += 1
                            if [inf, reactor] not in engage_with_false_r[scenario_str]:
                                metrics_dic['engage_with_false_r'].append([inf, reactor])
                                # engage_with_false_r[scenario_str].append([inf, reactor])
        else:
            print("[Metrics]: No Predicting Found, Update Failed!!")

        # if len(jerk_per_scenario) > 0:
        #     self.jerk += jerk_per_scenario
            # self.jerk = (self.jerk + np.average(jerk_per_scenario)) / 2
        # if len(jerk_others_per_scenario) > 0:
        #     self.jerk_others += jerk_others_per_scenario
            # self.jerk_others = (self.jerk_others + np.average(jerk_others_per_scenario)) / 2
        self.total_scenarios += 1
        return metrics_dic

    def summary(self):
        return f"Metrics summary: \n" \
               f"total scenarios: {self.total_scenarios}\n" \
               f"goal fit rate: {self.goal_fit / self.total_scenarios}\n" \
               f"total agents controlled: {self.total_agents_controlled} / {self.total_agents_controlled/(self.total_agents_controlled+self.total_agents_others+0.00001)}\n" \
               f"jerk (ego): {self.jerk_ego/self.total_scenarios}\n" \
               f"jerk (controlled): {self.jerk/(self.total_agents_controlled+0.001)}\n" \
               f"jerk (others): {self.jerk_others/(self.total_agents_others+0.001)}\n" \
               f"total front collisions: {self.front_collisions}\n" \
               f"total side collisions: {self.side_collisions}\n" \
               f"total rear collisions: {self.rear_collisions}\n" \
               f"collision rate (per scenario, colliding scenarios/all): {(self.front_collisions+self.side_collisions+self.rear_collisions)/self.total_scenarios} / {len(self.colliding_scenarios)/self.total_scenarios}\n" \
               f"total past agents: {self.past}\n" \
               f"total stuck agents: {self.get_stuck}\n" \
               f"total progress per agent (ego-fit): {self.ego_progress/self.goal_fit}\n" \
               f"total progress per agent (controlled): {self.progress/(self.total_agents_controlled+0.001)}\n" \
               f"total progress per agent (others): {self.progress_others/(self.total_agents_others+0.001)}\n" \
               f"ade per agent (ego-fit): {self.ade / self.goal_fit}\n" \
               f"fde per agent (ego-fit): {self.fde / self.goal_fit}\n" \
               f"offroad rate: {self.offroad_scenarios / self.total_scenarios}\n" \
               f"flip rate: {self.flip_relation} / {self.flip_relation / self.total_scenarios}\n" \
               f"emergency stops per scenario: {self.emergency_stops / self.total_scenarios}\n" \
               f"engage due to wrong relations: {self.engage_with_false_r} / {self.engage_with_false_r / self.total_scenarios}\n" \
               f"gpu predictions per scenarios: {self.gpu_relation_predictions / self.total_scenarios}\n" \
               f"max jerk of the ego vehicle distribution: {self.print_max_jerk()}\n" \
               f"ego progress distribution: {self.print_ego_progress()}"




    # f"fde per agent: {self.fde / (self.total_agents_controlled + 0.001)}\n" \

    def print_max_jerk(self):
        str = f"\n   max_jerk   scenario_num   num/total_scenario_num\n"
        for title in self.max_jerk_distribution:
            str += f"   {title}    {self.max_jerk_distribution[title]}     {self.max_jerk_distribution[title]/self.total_scenarios}\n"
        return str

    def print_ego_progress(self):
        str = f"\n   ego_progress   average_progress(m)    scenario_num\n"
        for title in self.ego_progress_distribution:
            progress, num = self.ego_progress_distribution[title]
            if num > 0:
                str += f"   {title}    {progress / num}     {num}\n"
            else:
                str += f"   {title}    {0}    {0}\n"
        return str

    def output(self):
        # dic_to_return = {}
        # attributes = self.__dict__
        # print("debug 1: ", attributes)
        # attributes = [a for a in attributes if not(a.startswith('__') and a.endswith('__'))]
        # print("debug 2: ", attributes)
        # for each in attributes:
        #     dic_to_return[each] = self.__getattribute__(each)
        return self.__dict__