import sys
import os
import importlib.util
import argparse
import pickle
from tqdm import tqdm
import sys
from interactions.detect_relations import get_relation_on_crossing
import numpy as np
from sklearn import metrics

def main(args):
    prediction_result_path = args.prediction_result
    ground_truth_path = args.gt_relation
    assert prediction_result_path is not None, 'pass prediction result to compute metrics'
    assert ground_truth_path is not None, 'pass ground truth label path to compute metrics'
    # load pickle from path
    if os.path.exists(prediction_result_path):
        with open(prediction_result_path, "rb") as f:
            prediction_dictionary = pickle.load(f)
    else:
        print(f"Error: cannot load ground truth relation from {prediction_result_path}")
        return None

    if os.path.exists(ground_truth_path):
        with open(ground_truth_path, "rb") as f:
            gt_labels_dictionary = pickle.load(f)
    else:
        print(f"Error: cannot load ground truth relation from {ground_truth_path}")
        return None
    # init confusion matrix for results
    metrics_result = {
        'Precision': None,
        'Recall': None,
        'Accuracy': None,
    }

    prediction_result = {
        'pred': [],
        'gt': []
    }

    # loop predictions
    for each_scenario_id in tqdm(prediction_dictionary):
        if each_scenario_id not in gt_labels_dictionary:
            print(f'{each_scenario_id} in prediction not found in ground truth, please check if they are on the same dataset')
            continue
        pred = prediction_dictionary[each_scenario_id]
        agent_dic = convert_prediction_into_agentdic(pred)
        edges_from_prediction = get_relation_on_crossing(agent_dic)
        edges_from_gt = gt_labels_dictionary[each_scenario_id]
        prediction_result = update_predictions(edges_from_prediction, edges_from_gt, prediction_result)

    # use average='micro' if you are using one hot labels
    metrics_result['Precision'] = metrics.precision_score(prediction_result['gt'], prediction_result['pred'])
    metrics_result['Recall'] = metrics.recall_score(prediction_result['gt'], prediction_result['pred'])
    metrics_result['Accuracy'] = metrics.accuracy_score(prediction_result['gt'], prediction_result['pred'])

    print(f'Metrics Compute Complete: \n'
          f'{metrics_result}')


def convert_prediction_into_agentdic(prediction_results):
    """
    Overwrite this method to suit your saved prediction result
    """
    agent_dic = {}
    default_shape = [4.726, 1.842]
    for each_agent in prediction_results:
        agent_dic[each_agent] = {}
        prediction_k = prediction_results[each_agent]
        predict_trajectory = prediction_k[0]  # select the first one
        agent_dic[each_agent]['pose'] = predict_trajectory
        agent_dic[each_agent]['shape'] = [default_shape[0], default_shape[1], 0]
    return agent_dic


def update_predictions(edges_from_prediction, edges_from_gt, prediction_result):
    all_agents = []
    for each in edges_from_prediction:
        all_agents += each[:2]
    for each in edges_from_gt:
        all_agents += each[:2]
    all_agents = set(all_agents)
    edges_grid_pred = np.zeros((len(all_agents), len(all_agents)))
    edges_grid_gt = np.zeros((len(all_agents), len(all_agents)))
    for each in edges_from_prediction:
        agent1, agent2 = each[:2]
        edges_grid_pred[all_agents.index(agent1), all_agents.index(agent2)] = 1
    for each in edges_from_gt:
        agent1, agent2 = each[:2]
        edges_grid_gt[all_agents.index(agent1), all_agents.index(agent2)] = 1
    prediction_result['pred'] += edges_grid_pred.flatten().tolist()
    prediction_result['gt'] += edges_grid_gt.flatten().tolist()
    return prediction_result


if __name__ == '__main__':
    """
    Require python higher than 3.7 for pickle loading
    Compute the metrics for loaded prediction trajectories
    """
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--prediction_result', type=str, default=None)
    parser.add_argument('--gt_relation', type=str, default=None)
    args_p = parser.parse_args()
    main(args_p)