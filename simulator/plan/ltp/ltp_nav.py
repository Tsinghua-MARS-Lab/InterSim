import numpy as np
import random
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusData, TrafficLightStatusType
from nuplan.common.actor_state.state_representation import Point2D

class goalAndNav:

    def __init__(self, road_dic):
        self.road_dic = road_dic
        self.goals = {}  # [goal_pt, each_park_id]
        self.max_goal_num = 5

    def set_goals(self, map_api, k=1):
        # get all park areas
        self.goals = {}
        if 'ego' in self.goals:
            goals = self.goals['ego']
        else:
            goals = []
        park_area_ids = []
        for each_obj_id in self.road_dic:
            if self.road_dic[each_obj_id]['type'] == 14:
                # only spawn goals for those having parking lots in them
                if len(self.road_dic[each_obj_id]['lower_level']) > 0:
                    park_area_ids.append(each_obj_id)
        total_goal_num = min(k, self.max_goal_num)
        total_goal_num = min(len(park_area_ids), total_goal_num)
        selected_park_area_ids = random.sample(park_area_ids, total_goal_num)
        for each_park_id in selected_park_area_ids:
            obj = self.road_dic[each_park_id]
            vertices = obj['xyz']
            total_pt_num = vertices.shape[0]
            selected_edge_idx = random.choice(list(range(total_pt_num-1)))
            pt1 = vertices[selected_edge_idx, :2]
            pt2 = vertices[selected_edge_idx+1, :2]
            goal_pt = [pt1[0] + random.random() * (pt2[0] - pt1[0]), pt1[1] + random.random() * (pt2[1] - pt1[1])]
            obj_id, dist_to_lane = map_api.get_distance_to_nearest_map_object(point=Point2D(goal_pt[0], goal_pt[1]),
                                                                              layer=SemanticMapLayer.LANE)
            goals.append([goal_pt, each_park_id, obj_id])
        self.goals['ego'] = goals

    def set_route(self, agent_id='ego', current_pose=None, closest_lane_id=None):
        # BFS
        if current_pose is None:
            current_pose, current_park_id, closest_lane_id = self.goals[agent_id][0]
            target_pose, target_park_id, target_lane_id = self.goals[agent_id][1]
            available_routes = []
            checking_pile = [[closest_lane_id]]
            lanes_visited = []
            while len(checking_pile) > 0 and len(available_routes) < 3:
                next_pile = []
                for each_route in checking_pile:
                    latest_lane = each_route[-1]
                    if latest_lane == target_lane_id:
                        available_routes.append(each_route+[target_lane_id])
                    else:
                        all_next_lanes = self.road_dic[latest_lane]['next_lanes']
                        if len(all_next_lanes) == 0 and len(each_route) == 1:
                            # starting from a dead end, turn around
                            all_next_lanes = self.road[latest_lane]['previous_lanes']
                        for each_next_lane in all_next_lanes:
                            if each_next_lane not in lanes_visited:
                                next_pile.append(each_route+[each_next_lane])
                checking_pile = next_pile
            return available_routes
        # TBD
        return None

