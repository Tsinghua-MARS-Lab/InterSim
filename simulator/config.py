from pathlib import Path

class Config(object):
    def __init__(self):
        self.window_w = 1200
        self.window_h = 1200
        self.scale = 5
        self.show_trajectory = True

        self.dataset = 'Waymo'  # or 'NuPlan'
        # self.dataset = 'NuPlan'
        self.running_mode = 1

        self.save_log_every_scenario = True

        self.dynamic_env_planner = 'env'  # pass in False for an open-loop simulation
        # other default ego planners: None (None for playback), 'dummy', 'trajpred', 'e2e'
        self.ego_planner = 'base'
        if self.dynamic_env_planner == 'env' and self.ego_planner == 'base':
            # parameters for default planners
            self.predict_env_for_ego_collisions = None  # or 'M2I'
            self.predict_relations_for_ego = True
            # env.predict_relations_for_ego = False
            self.predict_with_rules = True
        else:
            # parameters for default planners
            self.predict_env_for_ego_collisions = 'M2I'
            self.predict_relations_for_ego = False
            self.predict_with_rules = True

        self.planning_task = '8s'
        if self.dataset == 'Waymo':
            # Waymo
            self.frame_rate = 10
            self.planning_from = 11
            self.planning_to = 90
            self.planning_horizon = 80
            self.total_frame = 91
            self.planning_interval = 10  # 5
            self.tf_example_dir = str(Path.home()) + '/waymo_data/tf_validation_interact'
            self.map_name = 'Inter.Val'  # used for log simulation info
        if self.dataset == 'NuPlan':
            # NuPlan
            self.frame_rate = 20
            self.planning_from = 21
            self.planning_to = 181
            self.planning_horizon = 80
            self.total_frame = 181  # 91
            self.planning_interval = 20  # 5
            self.data_path = {
                'NUPLAN_DATA_ROOT': str(Path.home()) + "/nuplan/dataset",
                'NUPLAN_MAPS_ROOT': str(Path.home()) + "/nuplan/dataset/maps",
                'NUPLAN_DB_FILES': str(Path.home()) + "/nuplan/dataset/nuplan-v1.0/public_set_boston_train/",
            }
            self.map_name = 'Boston'  # used for log simulation info

        self.predictor = 'M2I'

        # Irrelevant configs
        self.loaded_prediction_path = '0107.multiInf_last8_vectorInf_tfR_reactorPred_all.pickle'
        self.draw_prediction = False  # for debugging loaded prediction trajectories
        self.draw_collision_pt = False  # for debugging loaded collision pts
        self.load_prediction_with_offset = True

        self.model_path = {
            'guilded_m_pred': './prediction/M2I/guilded_m_pred/pretrained/LowSpeedGoalDn.model.30.bin',
            'marginal_pred': './prediction/M2I/marginal_prediction/pretrained/8S.raster.maskNonInfFilterSteadyP5.Loopx10.0410.model.9.bin',
            'relation_pred': './prediction/M2I/relation_predictor/pretrained/infPred.NoTail.timeOffset.loopx2.IA.v2x.NonStop20.noIntention.0424.model.60.bin',
            'variety_loss_prediction': './prediction/M2I/variety_loss_prediction/pretrained/model.26.bin',
            'marginal_pred_tnt': './prediction/M2I/marginal_prediction_tnt/pretrained/tnt.model.21.bin'
        }

        self.test_task = 2  # 0=collision solving, 1=trajectory quality, 2=baseline ego planner
        self.testing_method = 2  # 0=densetnt with dropout, 1=0+post-processing, 2=1+relation, -1=variety loss

        self.filter_static = False
        self.filter_non_vehicle = False
        self.all_relevant = False
        self.follow_loaded_relation = False
        self.follow_prediction = False
        self.follow_gt_first = False  # do not use, unstable switching back and forward

        # test for upper bound
        # self.follow_gt_relation = True
        # self.follow_gt_conflict = True

        self.playback_dir = None

        self.scenario_augment = None
        self.scenario_augment = {
            'mask_non_influencer': [1, 100],  # [min, max]
            'mask_influencer': [1, 10],
            'change_speed': True,
            'change_route': True,
            'change_curvature': False,
        }

        self.relation_gt_path = None  # 'pickles/gt_direct_relation_1223.pickle'
        self.predict_device = 'cpu'  # 'mps' # 'cuda'

class EnvConfig(object):
    """
    running mode: 0=debug, 1=test planning algorithm, 2=playback previous results
    overwrite config parameters to change defaults
    """
    env = Config()







