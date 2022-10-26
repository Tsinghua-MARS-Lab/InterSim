from typing import Optional

from .structs import *

interactive_single_traj_waymo_pred: Optional[WaymoPred] = None

autoreg_1_waymo_pred: Optional[WaymoPred] = None

huawei_use_nms_waymo_pred: Optional[WaymoPred] = None

interactive_relations = None

pred_relations = {}
pred_direct_relations = {}

relation_pred = None

influencer_pred = None

direct_relation = None

all_relevant_agent_ids = None

waymo_autoreg_pred = WaymoAutoregPred()

temp = {}

loading_summary = {
    'label_0_loaded': 0,
    'label_1_loaded': 0,
    'label_1_to_predict': 0,
    'skipped': 0
}

