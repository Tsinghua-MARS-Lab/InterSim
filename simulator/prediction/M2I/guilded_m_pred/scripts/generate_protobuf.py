# Usage
# python scripts/generate_protobuf.py -i ~/results/validation_interactive_rdensetnt_full.pickle -o ~/results/submission/validation_interactive_submission.pb

import argparse
import numpy as np
import os
import tqdm

import pickle5 as pickle
from waymo_open_dataset.protos import motion_submission_pb2


def replace_marginal_prediction(prediction_traj_joint_raw, prediction_ids_raw, marginal_pred, replace_by_dist_to_8s=False):
    """
    Replace marginal prediction at 3s and 5s.

    Parameters
    ----------
    prediction_traj_joint_raw : numpy.ndarray
        Raw joint prediction using marginal data.
    prediction_ids_raw : list
        Agent ids.
    marginal_pred : dict
        Data to replace.
    replace_by_dist_to_8s : bool
        Whether to replace by distance to raw reactor prediction.

    Returns
    -------
    prediction_traj_joint_result : numpy.ndarray
        Updated joint prediction after replacement.

    """
    # [2, 6, horizon, 2].
    prediction_traj = marginal_pred["rst"]
    prediction_horizon = prediction_traj.shape[2]
    # [2, 6].
    prediction_score = marginal_pred["score"]
    # [2].
    prediction_ids = marginal_pred["ids"]

    if len(prediction_ids) < 2:
        return prediction_traj_joint_raw
    if prediction_ids != prediction_ids_raw:
        prediction_ids.reverse()
        prediction_score = np.flip(prediction_score, 0)
        prediction_traj = np.flip(prediction_traj, 0)
    assert prediction_ids == prediction_ids_raw, "Prediction ids mismatch when replacing."

    prediction_traj_joint_result = np.copy(prediction_traj_joint_raw)
    prediction_traj_short_end = prediction_traj[:, :, -1]
    prediction_traj_raw_end = prediction_traj_joint_raw[:, :, prediction_horizon-1]

    if replace_by_dist_to_8s:
        # Replace with shorter predictions that are close to 8s:
        # Replace for both agents.
        for i in range(2):
            selected_indices = []

            # Find the closest sample in short-term prediction with respect to original prediction samples.
            for s in range(6):
                raw_state_s = prediction_traj_raw_end[s, i]
                new_state_dist = np.sum((prediction_traj_short_end[i] - raw_state_s[np.newaxis]) ** 2, -1)
                closest_indices = np.argsort(new_state_dist)
                unselected_closest_indices = [index for index in closest_indices if index not in selected_indices]
                selected_indices.append(unselected_closest_indices[0])

                prediction_traj_joint_result[s, i, prediction_horizon-1] = prediction_traj_short_end[i, unselected_closest_indices[0]]
    else:
        # Replace using probability from 3s/5s predictions.
        prediction_traj_1 = prediction_traj[0][:, np.newaxis].repeat(6, 1)
        prediction_traj_2 = prediction_traj[1][np.newaxis].repeat(6, 0)
        prediction_traj_joint_raw = np.stack([prediction_traj_1, prediction_traj_2], axis=-3)

        prediction_score_joint = (prediction_score[0][:, np.newaxis] + prediction_score[1][np.newaxis]).reshape(-1)
        prediction_traj_joint = prediction_traj_joint_raw.reshape(-1, 2, prediction_horizon, 2)

        top_indices = np.argsort(-prediction_score_joint)
        # Obtain top k indices.
        top_k_indices = top_indices[:6]
        top_k_prediction_traj_joint = prediction_traj_joint[top_k_indices]
        prediction_traj_joint_result[:, :, prediction_horizon-1] = top_k_prediction_traj_joint[:, :, -1]

    return prediction_traj_joint_result


def replace_reactor_prediction(reactor_prediction, reactor_new_data, replace_by_dist_to_8s=False):
    """
    Replace reactor prediction at 3s and 5s.

    Parameters
    ----------
    reactor_prediction : numpy.ndarray
        Raw reactor prediction.
    reactor_new_data : dict
        Data to replace.
    replace_by_dist_to_8s : bool
        Whether to replace by distance to raw reactor prediction.

    Returns
    -------
    reactor_prediction_result : numpy.ndarray
        Updated reactor prediction after replacement.

    """
    reactor_prediction_result = np.copy(reactor_prediction)
    reactor_prediction_new = reactor_new_data["rst"]
    replace_horizon = reactor_prediction_new.shape[2]

    if replace_by_dist_to_8s:
        # Replace by distance to raw reactor prediction.
        for s1 in range(6):
            selected_indices = []
            for s2 in range(6):
                reactor_state_raw = reactor_prediction[s1, s2, replace_horizon-1]
                reactor_new_dist = np.sum((reactor_prediction_new[s1, :, -1] - reactor_state_raw[np.newaxis]) ** 2, -1)
                closest_indices = np.argsort(reactor_new_dist)
                unselected_closest_indices = [index for index in closest_indices if index not in selected_indices]
                selected_indices.append(unselected_closest_indices[0])

                reactor_prediction_result[s1, s2, replace_horizon-1] = reactor_prediction_new[s1, unselected_closest_indices[0], -1]
    else:
        # Replace by raw order.
        reactor_prediction_result[:, :, replace_horizon-1] = reactor_prediction_new[:, :, -1]

    return reactor_prediction_result


def load_reactor_data(
        reactor_data,
        prediction_traj,
        prediction_score,
        prediction_ids,
        key,
        reactor_pred_3s_dic=None,
        reactor_pred_5s_dic=None):
    """
    Load prediction results from reactor predictions.

    Parameters
    ----------
    reactor_data : dict
        Dictionary of reactor data.
    prediction_traj : numpy.ndarray
        Marginal prediction.
    prediction_score : numpy.ndarray
        Marginal prediction scores.
    prediction_ids : list
        Agent ids for each marginal prediction.
    key : str
        Scenario id.
    reactor_pred_3s_dic : dict
        Dictionary of 3 second prediction data.
    reactor_pred_5s_dic: dict
        Dictionary of 5 second prediction data.

    Returns
    -------
    prediction_traj_joint : numpy.ndarray
        Joint prediction after combining marginal influencer prediction and conditional reactor prediction
    prediction_score_joint : numpy.ndarray
        Joint prediction score
    reactor_first : bool
        Whether reactor is the first in the marginal prediction
    replace_success : bool
        Whether reactor replacement is successful.
    """
    reactor_id = reactor_data["ids"]
    # Skip replacement if reactor id not found.
    if reactor_id not in prediction_ids:
        print(f"reactor id not found {reactor_id} / {prediction_ids}")
        return None, None, False, False

    reactor_prediction = reactor_data["rst"]
    reactor_score = reactor_data["score"]
    reactor_first = True
    replace_success = True
    # Obtain influencer prediction and score.
    if reactor_id == prediction_ids[1]:
        reactor_first = False
        influencer_prediction = prediction_traj[0][:, np.newaxis].repeat(6, 1)
        influencer_score = prediction_score[0]
    else:
        influencer_prediction = prediction_traj[1][:, np.newaxis].repeat(6, 1)
        influencer_score = prediction_score[1]

    # Replace reactor predictions.
    if reactor_pred_5s_dic is not None and key in reactor_pred_5s_dic:
        reactor_new_data = reactor_pred_5s_dic[key]
        reactor_prediction = replace_reactor_prediction(reactor_prediction, reactor_new_data, args.replace_by_dist_to_8s)
    if reactor_pred_3s_dic is not None and key in reactor_pred_3s_dic:
        reactor_new_data = reactor_pred_3s_dic[key]
        reactor_prediction = replace_reactor_prediction(reactor_prediction, reactor_new_data, args.replace_by_dist_to_8s)

    # Combine influencer and reactor trajectories and scores, where influencer always goes first.
    prediction_traj_joint = np.stack([influencer_prediction, reactor_prediction], axis=-3)
    prediction_score_joint = influencer_score[:, np.newaxis] + reactor_score
    prediction_score_joint = prediction_score_joint.reshape(-1)

    return prediction_traj_joint, prediction_score_joint, reactor_first, replace_success


def generate_pb(args):
    submission = motion_submission_pb2.MotionChallengeSubmission()
    submission.account_name = 'cyrushx@gmail.com'
    # submission.account_name = 'larksq@gmail.com'
    submission.unique_method_name = 'M2I'
    submission_count = 0

    # Load marginal predictions.
    with open(args.input_path, 'rb') as f:
        data = pickle.load(f)

    # Load reactor predictions by type.
    if args.v2v_pred is not None:
        with open(args.v2v_pred, 'rb') as f:
            v2v_pred_dic = pickle.load(f)

        # For v2v type, add an option to load 3s/5s predictions.
        if args.v2v_3s_pred is not None:
            with open(args.v2v_3s_pred, 'rb') as f:
                v2v_3s_pred_dic = pickle.load(f)
        else:
            v2v_3s_pred_dic = None

        if args.v2v_5s_pred is not None:
            with open(args.v2v_5s_pred, 'rb') as f:
                v2v_5s_pred_dic = pickle.load(f)
        else:
            v2v_5s_pred_dic = None

    if args.c2v_pred is not None:
        with open(args.c2v_pred, 'rb') as f:
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

    # Load marginal predictions from 3s and 5s models.
    if args.marginal_3s_pred is not None:
        with open(args.marginal_3s_pred, 'rb') as f:
            marginal_3s_pred_dic = pickle.load(f)
    if args.marginal_5s_pred is not None:
        with open(args.marginal_5s_pred, 'rb') as f:
            marginal_5s_pred_dic = pickle.load(f)

    replace_counts = {"v2v": 0, "c2v": 0, "p2v": 0, "x2c": 0, "x2p": 0, "x2v": 0}
    overlap_count = 0

    for key in tqdm.tqdm(data):
        scenario_prediction = submission.scenario_predictions.add()
        scenario_prediction.scenario_id = key.decode("utf-8")

        joint_prediction = scenario_prediction.joint_prediction

        # Read pairwise prediction data.
        prediction_data = data[key]
        # [2, 6, 80, 2].
        prediction_traj = prediction_data["rst"]
        # [2, 6].
        prediction_score = prediction_data["score"]
        # [2].
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
        reactor_first = False
        replace_success = False
        if args.use_prediction:
            if args.x2p_pred is not None:
                if key in x2p_pred_dic:
                    x2p_data = x2p_pred_dic[key]
                    prediction_traj_joint_raw, prediction_score_joint, reactor_first, replace_success \
                        = load_reactor_data(x2p_data, prediction_traj, prediction_score, prediction_ids, key)
                    replace_counts["x2p"] += replace_success

            if args.x2c_pred is not None:
                if key in x2c_pred_dic:
                    x2c_data = x2c_pred_dic[key]
                    prediction_traj_joint_raw, prediction_score_joint, reactor_first, replace_success \
                        = load_reactor_data(x2c_data, prediction_traj, prediction_score, prediction_ids, key)
                    replace_counts["x2c"] += replace_success

            if args.x2v_pred is not None:
                if key in x2v_pred_dic:
                    x2v_data = x2v_pred_dic[key]
                    prediction_traj_joint_raw, prediction_score_joint, reactor_first, replace_success \
                        = load_reactor_data(x2v_data, prediction_traj, prediction_score, prediction_ids, key)
                    replace_counts["x2v"] += replace_success

            if args.p2v_pred is not None:
                if key in p2v_pred_dic:
                    p2v_data = p2v_pred_dic[key]
                    prediction_traj_joint_raw, prediction_score_joint, reactor_first, replace_success \
                        = load_reactor_data(p2v_data, prediction_traj, prediction_score, prediction_ids, key)
                    replace_counts["p2v"] += replace_success

            if args.c2v_pred is not None:
                if key in c2v_pred_dic:
                    c2v_data = c2v_pred_dic[key]
                    prediction_traj_joint_raw, prediction_score_joint, reactor_first, replace_success \
                        = load_reactor_data(c2v_data, prediction_traj, prediction_score, prediction_ids, key)
                    replace_counts["c2v"] += replace_success

            if args.v2v_pred is not None:
                if key in v2v_pred_dic:
                    v2v_data = v2v_pred_dic[key]
                    prediction_traj_joint_raw, prediction_score_joint, reactor_first, replace_success \
                        = load_reactor_data(v2v_data, prediction_traj, prediction_score, prediction_ids,
                                            key, v2v_3s_pred_dic, v2v_5s_pred_dic)
                    replace_counts["v2v"] += replace_success

            # Reverse marginal prediction ids if reactor is the first in the reactor data.
            if replace_success and reactor_first:
                prediction_ids = [prediction_ids[1], prediction_ids[0]]

        prediction_traj_joint = prediction_traj_joint_raw.reshape(-1, 2, 80, 2)

        # Obtain 6 indices for sample selection.
        if args.order610:
            # Use fixed indices according to heuristics.
            top_k_indices = np.array([0, 6, 12, 18, 24, 30])

            # Use 160 rule when reactor influencer order is swapped.
            if args.use_prediction and replace_success and reactor_first:
                top_k_indices = np.array([0, 1, 2, 3, 4, 5])
        else:
            top_indices = np.argsort(-prediction_score_joint)
            # Obtain top k indices.
            top_k_indices = top_indices[:6]
        top_k_prediction_score_joint = prediction_score_joint[top_k_indices]

        # Filter overlapping trajectories between interacting agents. This does _not_ consider overlapping with other agents.
        # THIS SACRIFICE ACCURACY WITH LITTLE OVERLAP IMPROVEMENT AND SHOULD NOT BE USED.
        if args.filter_overlap:
            prediction_traj_joint_ordered = prediction_traj_joint[np.argsort(-prediction_score_joint)]
            prediction_distance = np.min(np.sqrt(np.sum((prediction_traj_joint_ordered[:, 0] - prediction_traj_joint_ordered[:, 1]) ** 2, -1)), -1)
            # Make sure the first sample is not overlapping.
            non_overlapping_masks = prediction_distance > 5.0
            non_overlapping_indices = np.where(non_overlapping_masks)[0]

            if non_overlapping_indices.shape[0] > 0:
                first_non_overlapping_index = non_overlapping_indices[0]

                if first_non_overlapping_index in top_k_indices:
                    top_k_indices = np.insert(top_k_indices[top_k_indices != first_non_overlapping_index], 0, first_non_overlapping_index)
                else:
                    top_k_indices = np.insert(top_k_indices[: 5], 0, first_non_overlapping_index)
            else:
                overlap_count += 1

        top_k_prediction_traj_joint = prediction_traj_joint[top_k_indices]

        # Replace marginal prediction results at shorter horizons.
        if not replace_success:
            if args.marginal_3s_pred is not None and key in marginal_3s_pred_dic:
                top_k_prediction_traj_joint = replace_marginal_prediction(top_k_prediction_traj_joint, prediction_ids, marginal_3s_pred_dic[key], args.replace_by_dist_to_8s)
            if args.marginal_5s_pred is not None and key in marginal_5s_pred_dic:
                top_k_prediction_traj_joint = replace_marginal_prediction(top_k_prediction_traj_joint, prediction_ids, marginal_5s_pred_dic[key], args.replace_by_dist_to_8s)

        # Overwrite score using marginal score from agent 0.
        if args.marginal_score or (not replace_success and not args.marginal_joint_score):
            top_k_prediction_score_joint = prediction_score[0]

        # Normalize score so that they are positive values.
        top_k_prediction_score_joint = np.exp(top_k_prediction_score_joint) / (np.exp(top_k_prediction_score_joint).sum())

        # Save to protobufs.
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

    if args.filter_overlap:
        print("Overlap counts", overlap_count)

    print('Saving {} scenarios into {} examples at {}'.format(len(data), submission_count, args.output_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input_path', help='Path of input pickle file.')
    parser.add_argument('-o', '--output_path', help='Path of output pickle file to save.')
    parser.add_argument('-d', '--description', type=str, help='Model description used for Waymo submission.')

    parser.add_argument('--order610', action="store_true", help='Use 610 for indexing.')
    parser.add_argument('--filter_overlap', action="store_true", help='Filter overlapping trajectories.')
    parser.add_argument('--marginal_score', action="store_true", help='Use marginal score for all prediction.')
    parser.add_argument('--marginal_joint_score', action="store_true", help='Use joint score for marginal prediction.')

    parser.add_argument('--use_prediction', action="store_true", help='Use I->R predictions..')
    parser.add_argument('--v2v_pred', default=None, help='Path of v2v reactor prediction result.')
    parser.add_argument('--c2v_pred', default=None, help='Path of c2v reactor prediction result.')
    parser.add_argument('--p2v_pred', default=None, help='Path of p2v reactor prediction result.')
    parser.add_argument('--x2c_pred', default=None, help='Path of x2c reactor prediction result.')
    parser.add_argument('--x2p_pred', default=None, help='Path of x2p reactor prediction result.')
    parser.add_argument('--x2v_pred', default=None, help='Path of x2v reactor prediction result.')

    # Replace predictions with 3s/5s results.
    parser.add_argument('--replace_by_dist_to_8s', action="store_true", help='Replace predictions based on distances to 8s trajectories. [NOT WORKING AND SHOULD NOT BE USED]')
    # Marginal prediction paths
    parser.add_argument('--marginal_3s_pred', default=None, help='Path of 3s marginal prediction.')
    parser.add_argument('--marginal_5s_pred', default=None, help='Path of 5s marginal prediction.')
    # Reactor prediction paths
    parser.add_argument('--v2v_3s_pred', default=None, help='Path of v2v reactor prediction 3s result.')
    parser.add_argument('--v2v_5s_pred', default=None, help='Path of v2v reactor prediction 5s result.')

    args = parser.parse_args()

    generate_pb(args)


if __name__ == '__main__':
    main()
