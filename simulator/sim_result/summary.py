import pickle
import os
import numpy as np
import json


def load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def save(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

class SummaryAPI():
    def summary_and_save(self, simulation_path, info):
        m = Metrics()
        m.addup_logs_beta(simulation_path)
        print(f"SUMMARY for {simulation_path}:")
        print(m.summary())
        dict_to_save = m.__dict__
        dict_to_save['simulation_info'] = info
        save(os.path.join(simulation_path, 'summary.simulation.metric'), dict_to_save)
        with open(os.path.join(simulation_path, 'summary.simulation.json'), 'w') as fp:
            json.dump(dict_to_save, fp)


class Metrics():
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
        self.ego_progress_log = 0

        self.emergency_stops = 0

        self.max_jerk_distribution = {'0-0.01': 0,
                                      '0.01-0.1': 0,
                                      '0.1-0.2': 0,
                                      '0.2-0.5': 0,
                                      '0.5-1': 0,
                                      '1-2': 0,
                                      '2-5': 0,
                                      '>5': 0}

        self.ego_progress_distribution = {'0-5': [0, 0],
                                          '5-10': [0, 0],
                                          '10-50': [0, 0],
                                          '50-100': [0, 0],
                                          '100-200': [0, 0],
                                          '200-500': [0, 0],
                                          '>500': [0, 0]}

        self.offroad_scenarios = 0
        self.total_collisions = 0

    def addup(self, path):
        print("Adding up for path: ", path)
        folders = [os.path.join(path, f) for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))]
        total_scenario_index = list(range(150))
        print("Num of folders: ", len(folders))
        for each_folder in folders:
            scenario_folders = [os.path.join(each_folder, f) for f in os.listdir(each_folder) if not os.path.isfile(os.path.join(each_folder, f))]
            for each_scenario_folder in scenario_folders:

                playback_path = os.path.join(each_scenario_folder, 'playback')
                scenario_file_paths = [os.path.join(playback_path, f) for f in os.listdir(playback_path) if os.path.isfile(os.path.join(playback_path, f))]
                for each_scenario_file_path in scenario_file_paths:
                    current_scenario_index = each_scenario_file_path.split('.tfrecord-')[1][:5]
                    if int(current_scenario_index) in total_scenario_index:
                        total_scenario_index.remove(int(current_scenario_index))
                    else:
                        print("not found, ", int(current_scenario_index),each_scenario_file_path, each_scenario_folder)
                        continue

                rst_paths = [os.path.join(each_scenario_folder, f) for f in os.listdir(each_scenario_folder) if os.path.isfile(os.path.join(each_scenario_folder, f))]
                for each_file in rst_paths:
                    if '.log' not in each_file:
                        continue
                    rst_dic = load(each_file)
                    print("Inspect scneario num:", each_file, each_scenario_folder, rst_dic['total_scenarios'])
                    self.total_scenarios += rst_dic['total_scenarios']
                    self.past += rst_dic['past']
                    self.get_stuck += rst_dic['get_stuck']
                    self.progress += rst_dic['progress']
                    self.progress_others += rst_dic['progress_others']
                    self.front_collisions += rst_dic['front_collisions']
                    self.side_collisions += rst_dic['side_collisions']
                    self.rear_collisions += rst_dic['rear_collisions']
                    self.total_collisions += (rst_dic['front_collisions'] + rst_dic['side_collisions'] + rst_dic['rear_collisions'])
                    self.colliding_scenarios += rst_dic['colliding_scenarios']
                    self.running_redlights += rst_dic['running_redlights']
                    self.jerk += rst_dic['jerk']
                    self.jerk_others += rst_dic['jerk_others']
                    self.jerk_ego += rst_dic['jerk_ego']
                    self.total_agents_controlled += rst_dic['total_agents_controlled']
                    self.total_agents_others += rst_dic['total_agents_others']
                    self.total_scenarios += rst_dic['total_scenarios']
                    self.fde += rst_dic['fde']
                    self.ade += rst_dic['ade']
                    self.goal_fit += rst_dic['goal_fit']
                    self.flip_relation += rst_dic['flip_relation']
                    self.engage_with_false_r += rst_dic['engage_with_false_r']   # definitely not working
                    self.ego_progress += rst_dic['ego_progress']
                    self.emergency_stops += rst_dic['emergency_stops']
                    self.offroad_scenarios += rst_dic['offroad_scenarios']

                    for each_key in self.max_jerk_distribution:
                        self.max_jerk_distribution[each_key] += rst_dic['max_jerk_distribution'][each_key]

                    for idx, each_key in enumerate(self.ego_progress_distribution):
                        avg, num = rst_dic['ego_progress_distribution'][each_key]
                        current_avg, current_num = self.ego_progress_distribution[each_key]
                        new_avg = (avg * num + current_avg * current_num) / (current_num + num + 0.0001)
                        self.ego_progress_distribution[each_key] = [new_avg, num + current_num]

                        self.ego_progress_log += idx * num

                    self.colliding_scenarios = np.unique(self.colliding_scenarios).tolist()
                    if len(rst_dic['colliding_scenarios']) > 0:
                        print("collision list: ", playback_path, rst_dic['colliding_scenarios'])

        print("Unresolved scenario indices: ", len(total_scenario_index), total_scenario_index)

    def addup_logs_beta(self, path):
        print("Adding up for path: ", path)
        playback_folder = os.path.join(path, 'playback')
        if not os.path.isdir(playback_folder):
            assert False, 'playback does not exist for ' + path
        all_sim_files = []
        for each_file_name in os.listdir(path):
            print("tttttest: ", each_file_name)
            if '.scene.metric' in each_file_name:
                all_sim_files.append(each_file_name.split('.scene.metric')[0])
                rst_dic = load(os.path.join(path, each_file_name))
        # print("Test all sim files: ", all_sim_files)
        # for each_sim_file in all_sim_files:
        #     playback = os.path.join(playback_folder, each_sim_file + '.playback')
        #     rst_dic = load(playback)
        # for each_file in os.listdir(playback_folder):
        #     if '.scene.metric' in each_file:
        #         rst_dic = load(os.path.join(playback_folder, each_file))
                print("Inspect file:", os.path.join(path, each_file_name), rst_dic['total_scenarios'])
                self.total_scenarios += rst_dic['total_scenarios']
                self.past += rst_dic['past']
                self.get_stuck += rst_dic['get_stuck']
                self.progress += rst_dic['progress']
                self.progress_others += rst_dic['progress_others']
                self.front_collisions += rst_dic['front_collisions']
                self.side_collisions += rst_dic['side_collisions']
                self.rear_collisions += rst_dic['rear_collisions']
                self.total_collisions += (
                            rst_dic['front_collisions'] + rst_dic['side_collisions'] + rst_dic['rear_collisions'])
                self.colliding_scenarios += rst_dic['colliding_scenarios']
                self.running_redlights += rst_dic['running_redlights']
                self.jerk += rst_dic['jerk']
                self.jerk_others += rst_dic['jerk_others']
                self.jerk_ego += rst_dic['jerk_ego']
                self.total_agents_controlled += rst_dic['total_agents_controlled']
                self.total_agents_others += rst_dic['total_agents_others']
                self.total_scenarios += rst_dic['total_scenarios']
                self.fde += rst_dic['fde']
                self.ade += rst_dic['ade']
                self.goal_fit += rst_dic['goal_fit']
                self.flip_relation += rst_dic['flip_relation']
                self.engage_with_false_r += rst_dic['engage_with_false_r']  # definitely not working
                self.ego_progress += rst_dic['ego_progress']
                self.emergency_stops += rst_dic['emergency_stops']
                self.offroad_scenarios += rst_dic['offroad_scenarios']

                for each_key in self.max_jerk_distribution:
                    self.max_jerk_distribution[each_key] += rst_dic['max_jerk_distribution'][each_key]

                for idx, each_key in enumerate(self.ego_progress_distribution):
                    avg, num = rst_dic['ego_progress_distribution'][each_key]
                    current_avg, current_num = self.ego_progress_distribution[each_key]
                    new_avg = (avg * num + current_avg * current_num) / (current_num + num + 0.0001)
                    self.ego_progress_distribution[each_key] = [new_avg, num + current_num]

                    self.ego_progress_log += idx * num

                self.colliding_scenarios = np.unique(self.colliding_scenarios).tolist()
                if len(rst_dic['colliding_scenarios']) > 0:
                    print("collision list: ", playback_path, rst_dic['colliding_scenarios'])

        print(f"Merged simulation result for {self.total_scenarios} scenarios.")

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
               f"collision rate (per scenario, colliding scenarios/all): {self.total_collisions/self.total_scenarios} / {len(self.colliding_scenarios)/self.total_scenarios}\n" \
               f"total past agents (num in total/rate): {self.past} / {self.past / (self.goal_fit)}\n" \
               f"total stuck agents (num in total/rate): {self.get_stuck} / {self.get_stuck / self.total_scenarios}\n" \
               f"total progress per agent (ego-fit): {self.ego_progress/self.goal_fit}\n" \
               f"total log-progress per agent (ego-fit): {self.ego_progress_log / self.goal_fit}\n" \
               f"total progress per agent (controlled): {self.progress/(self.total_agents_controlled+0.001)}\n" \
               f"total progress per agent (others): {self.progress_others/(self.total_agents_others+0.001)}\n" \
               f"ade per agent (ego-fit): {self.ade / self.goal_fit}\n" \
               f"fde per agent (ego-fit): {self.fde / self.goal_fit}\n" \
               f"offroad rate: {self.offroad_scenarios / self.total_scenarios}\n" \
               f"flip rate: {self.flip_relation} / {self.flip_relation / self.total_scenarios}\n" \
               f"emergency stops per scenario: {self.emergency_stops / self.total_scenarios}\n" \
               f"engage due to wrong relations: {self.engage_with_false_r} / {self.engage_with_false_r / self.total_scenarios}\n" \
               f"max jerk of the ego vehicle distribution: {self.print_max_jerk()}\n" \
               f"ego progress distribution: {self.print_ego_progress()}"

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

if __name__ == '__main__':
    # path = './constantV_noExcluded_val_3/'
    # path = './baseline_noExcluded_predPed_val'
    # path = './constantV_noExcluded_val_Sep12Final'  # constantV
    # path = './baseline_noExcluded_predPed_k6_constantV_val_KeepNInspect'  # predR
    # path = './baseline_noExcluded_yieldAll_k6_constantV_val_KeepNInspect'  # baseline without relation
    # path = './prediction_noExcluded_k6_val'  # DenseTNT
    # path = './predictionTNT_noExcluded_k6_val'  # TNT
    # path = './onestep_baseline_noExcluded_predPed_k6_constantV_val_Sep14'
    # path = './baseline_noExcluded_predPed_k6_constantV_val_Sep14'
    # path = './constantV_keep_predPed_k6_constantV_val'
    # path = './baseline_keep_predPed_k6_constantV_val'

    default_path = './baseline_noExcluded_predPed_k6_constantV_val_Sep15'

    # path = './baseline_selected_val'
    # path = './constantV_selected_k6_val'

    m = Metrics()
    m.addup(default_path)

    print(m.summary())
#
# def save_data_to_pickle(data_to_save, saving_file_path):
#     with open(saving_file_path, 'wb') as f:
#         pickle.dump(data_to_save, f, pickle.HIGHEST_PROTOCOL)
#
# save_data_to_pickle(m.colliding_scenarios, 'colliding_scenarios.pickle')
#
# print(m.colliding_scenarios)

