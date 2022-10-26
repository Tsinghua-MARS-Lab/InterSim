# Usage
# python scripts/generate_protobuf.py -i ~/results/validation_interactive_rdensetnt_full.pickle -o ~/results/submission/validation_interactive_submission.pb

import argparse
import numpy as np
import os
import tqdm

import pickle5 as pickle
from waymo_open_dataset.protos import motion_submission_pb2


# Load the prediction from reactor predictions
def load_reactor_data(reactor_data, prediction_traj, prediction_score, prediction_ids):
    reactor_id = reactor_data["ids"]
    # Skip replacement if reactor id not found.
    if reactor_id not in prediction_ids:
        print(f"reactor id not found {reactor_id} / {prediction_ids}")
        return None, None, False, False

    reactor_prediction = reactor_data["rst"]
    reactor_score = reactor_data["score"]
    reactor_first = True
    replace_success = True
    if reactor_id == prediction_ids[1]:
        reactor_first = False
        influencer_prediction = prediction_traj[0][:, np.newaxis].repeat(6, 1)
        influencer_score = prediction_score[0]
    else:
        influencer_prediction = prediction_traj[1][:, np.newaxis].repeat(6, 1)
        influencer_score = prediction_score[1]

    # Combine influencer and reactor trajectories and scores, where influencer goes first.
    prediction_traj_joint = np.stack([influencer_prediction, reactor_prediction], axis=-3)
    prediction_score_joint = influencer_score[:, np.newaxis] + reactor_score
    prediction_score_joint = prediction_score_joint.reshape(-1)

    return prediction_traj_joint, prediction_score_joint, reactor_first, replace_success


def generate_pb(args):
    submission = motion_submission_pb2.MotionChallengeSubmission()
    # submission.account_name = 'cyrushx@gmail.com'
    submission.account_name = 'larksq@gmail.com'
    submission.unique_method_name = '#10019'
    submission_count = 0

    with open(args.input_path, 'rb') as f:
        data = pickle.load(f)

    if args.v2v_pred is not None:
        with open(args.v2v_pred, 'rb') as f:
            v2v_pred_dic = pickle.load(f)
    if args.c2v_pred is not None:
        with open(args.v2v_pred, 'rb') as f:
            c2v_pred_dic = pickle.load(f)
    if args.p2v_pred is not None:
        with open(args.p2v_pred, 'rb') as f:
            p2v_pred_dic = pickle.load(f)
    if args.x2c_pred is not None:
        with open(args.x2c_pred, 'rb') as f:
            x2c_pred_dic = pickle.load(f)
    if args.x2p_pred is not None:
        with open(args.x2p_pred, 'rb') as f:
            x2p_pred_dic = pickle.load(f)
    if args.x2v_pred is not None:
        with open(args.x2v_pred, 'rb') as f:
            x2v_pred_dic = pickle.load(f)


    replace_counts = {"v2v": 0, "c2v": 0, "p2v": 0, "x2c": 0, "x2p": 0, "x2v": 0}

    for key in tqdm.tqdm(data):
        scenario_prediction = submission.scenario_predictions.add()
        scenario_prediction.scenario_id = key.decode("utf-8")

        joint_prediction = scenario_prediction.joint_prediction

        # Read data.
        prediction_data = data[key]
        prediction_traj = prediction_data["rst"]
        prediction_score = prediction_data["score"]
        prediction_ids = prediction_data["ids"]

        if len(prediction_ids) != 2:
            print("Missing predictions in scenario {}".format(key))
            continue

        submission_count += 1
        # Combine marginal predictions into joint prediction.
        prediction_traj_1 = prediction_traj[0][:, np.newaxis].repeat(6, 1)
        prediction_traj_2 = prediction_traj[1][np.newaxis].repeat(6, 0)
        prediction_traj_joint_raw = np.stack([prediction_traj_1, prediction_traj_2], axis=-3)

        prediction_score_joint = (prediction_score[0][:, np.newaxis] + prediction_score[1][np.newaxis]).reshape(-1)

        # Replace prediction results using reactor prediction.
        if args.use_prediction:
            reactor_first = False
            replace_success = False

            if args.x2p_pred is not None:
                if key in x2p_pred_dic:
                    x2p_data = x2p_pred_dic[key]
                    prediction_traj_joint_raw, prediction_score_joint, reactor_first, replace_success \
                        = load_reactor_data(x2p_data, prediction_traj, prediction_score, prediction_ids)
                    replace_counts["x2p"] += replace_success

            if args.x2c_pred is not None:
                if key in x2c_pred_dic:
                    x2c_data = x2c_pred_dic[key]
                    prediction_traj_joint_raw, prediction_score_joint, reactor_first, replace_success \
                        = load_reactor_data(x2c_data, prediction_traj, prediction_score, prediction_ids)
                    replace_counts["x2c"] += replace_success

            if args.x2v_pred is not None:
                if key in x2v_pred_dic:
                    x2v_data = x2v_pred_dic[key]
                    prediction_traj_joint_raw, prediction_score_joint, reactor_first, replace_success \
                        = load_reactor_data(x2v_data, prediction_traj, prediction_score, prediction_ids)
                    replace_counts["x2v"] += replace_success

            if args.p2v_pred is not None:
                if key in p2v_pred_dic:
                    p2v_data = p2v_pred_dic[key]
                    prediction_traj_joint_raw, prediction_score_joint, reactor_first, replace_success \
                        = load_reactor_data(p2v_data, prediction_traj, prediction_score, prediction_ids)
                    replace_counts["p2v"] += replace_success

            if args.c2v_pred is not None:
                if key in c2v_pred_dic:
                    c2v_data = c2v_pred_dic[key]
                    prediction_traj_joint_raw, prediction_score_joint, reactor_first, replace_success \
                        = load_reactor_data(c2v_data, prediction_traj, prediction_score, prediction_ids)
                    replace_counts["c2v"] += replace_success

            if args.v2v_pred is not None:
                if key in v2v_pred_dic:
                    v2v_data = v2v_pred_dic[key]
                    prediction_traj_joint_raw, prediction_score_joint, reactor_first, replace_success \
                        = load_reactor_data(v2v_data, prediction_traj, prediction_score, prediction_ids)
                    replace_counts["v2v"] += replace_success

            # Reverse prediction ids if reactor is the first in the raw data.
            if replace_success and reactor_first:
                prediction_ids.reverse()

        prediction_traj_joint = prediction_traj_joint_raw.reshape(-1, 2, 80, 2)
        if args.order610:
            # Use fixed indices.
            top_k_indices = np.array([0, 6, 12, 18, 24, 30])

            if args.use_prediction and replace_success and reactor_first:
                top_k_indices = np.array([0, 1, 2, 3, 4, 5])
        else:
            # Obtain top k indices.
            top_k_indices = np.argsort(-prediction_score_joint)[:6]
        top_k_prediction_traj_joint = prediction_traj_joint[top_k_indices]
        top_k_prediction_score_joint = prediction_score_joint[top_k_indices]

        # Overwrite score using marginal score from agent 0, for marginal prediction only.
        if not replace_success and not args.marginal_joint_score:
            top_k_prediction_score_joint = prediction_score[0]

        # Normalize score.
        if args.normalize_score:
            top_k_prediction_score_joint = np.exp(top_k_prediction_score_joint) / (np.exp(top_k_prediction_score_joint).sum())

        for k in range(6):
            ScoredJointTrajectory = joint_prediction.joint_trajectories.add()
            ScoredJointTrajectory.confidence = top_k_prediction_score_joint[k]

            for c in range(2):
                ObjectTrajectory = ScoredJointTrajectory.trajectories.add()
                ObjectTrajectory.object_id = prediction_ids[c]
                Trajectory = ObjectTrajectory.trajectory

                interval = 5
                traj = top_k_prediction_traj_joint[k, c, (interval - 1)::interval, :]
                Trajectory.center_x[:] = traj[:, 0].tolist()
                Trajectory.center_y[:] = traj[:, 1].tolist()

    submission.submission_type = motion_submission_pb2.MotionChallengeSubmission.SubmissionType.INTERACTION_PREDICTION
    if args.description is not None:
        submission.description = args.description

    path = args.output_path
    with open(path, "wb") as f:
        f.write(submission.SerializeToString())

    os.system(f'tar -zcvf {path}.tar.gz {path}')
    os.system(f'rm {path}')

    if args.use_prediction:
        print("Replacement counts", replace_counts)

    print('Saving {} scenarios into {} examples at {}'.format(len(data), submission_count, args.output_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input_path', help='Path of input pickle file.')
    parser.add_argument('-o', '--output_path', help='Path of output pickle file to save.')
    parser.add_argument('-d', '--description', type=str, help='Model description used for Waymo submission.')

    parser.add_argument('--order610', action="store_true", help='Use 610 for indexing.')
    parser.add_argument('--marginal_joint_score', action="store_true", help='Use joint score for marginal prediction.')
    parser.add_argument('--normalize_score', action="store_true", help='Normalize score.')
    parser.add_argument('--use_prediction', action="store_true", help='Use I->R predictions..')
    parser.add_argument('--v2v_pred', default=None, help='Path of v2v reactor prediction result.')
    parser.add_argument('--c2v_pred', default=None, help='Path of c2v reactor prediction result.')
    parser.add_argument('--p2v_pred', default=None, help='Path of p2v reactor prediction result.')
    parser.add_argument('--x2c_pred', default=None, help='Path of x2c reactor prediction result.')
    parser.add_argument('--x2p_pred', default=None, help='Path of x2p reactor prediction result.')
    parser.add_argument('--x2v_pred', default=None, help='Path of x2v reactor prediction result.')
    args = parser.parse_args()

    generate_pb(args)


if __name__ == '__main__':
    main()
