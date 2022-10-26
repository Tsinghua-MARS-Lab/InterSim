# Usage: python visualize_pairwise_predictions.py -i ~/.../validation_interactive/validation_interactive_tfexample.tfrecord-00000-of-00150 -f ../results/validation_interactive_v_rdensetnt_full.pickle -o tmp_plot

import argparse
import os

import numpy as np
import tensorflow as tf
import tqdm

import pickle5 as pickle

import numpy as np
import tensorflow as tf

from matplotlib import cm
import matplotlib.pyplot as plt

from google.protobuf import text_format
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import motion_metrics_pb2

# Example field definition
roadgraph_features = {
    'roadgraph_samples/dir':
        tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
    'roadgraph_samples/id':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/type':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/valid':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/xyz':
        tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
}

# Features of other agents.
state_features = {
    'state/id':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/type':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/is_sdc':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/tracks_to_predict':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/current/bbox_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/height':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/length':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/timestamp_micros':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/valid':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/vel_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/width':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/z':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/future/bbox_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/height':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/length':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/timestamp_micros':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/valid':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/vel_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/width':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/z':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/past/bbox_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/height':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/length':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/timestamp_micros':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/valid':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/vel_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/width':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/z':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
}

traffic_light_features = {
    'traffic_light_state/current/state':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/valid':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/x':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/y':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/z':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/past/state':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/valid':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/x':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/y':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/z':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
}

features_description = {}
features_description.update(roadgraph_features)
features_description.update(state_features)
features_description.update(traffic_light_features)
features_description['scenario/id'] = tf.io.FixedLenFeature([1], tf.string, default_value=None)
features_description['state/objects_of_interest'] = tf.io.FixedLenFeature([128], tf.int64, default_value=None)


def _parse(value):
    decoded_example = tf.io.parse_single_example(value, features_description)

    past_states = tf.stack([
        decoded_example['state/past/x'],
        decoded_example['state/past/y'],
        decoded_example['state/past/length'],
        decoded_example['state/past/width'],
        decoded_example['state/past/bbox_yaw'],
        decoded_example['state/past/velocity_x'],
        decoded_example['state/past/velocity_y']
    ], -1)

    cur_states = tf.stack([
        decoded_example['state/current/x'],
        decoded_example['state/current/y'],
        decoded_example['state/current/length'],
        decoded_example['state/current/width'],
        decoded_example['state/current/bbox_yaw'],
        decoded_example['state/current/velocity_x'],
        decoded_example['state/current/velocity_y']
    ], -1)

    input_states = tf.concat([past_states, cur_states], 1)[..., :2]

    future_states = tf.stack([
        decoded_example['state/future/x'],
        decoded_example['state/future/y'],
        decoded_example['state/future/length'],
        decoded_example['state/future/width'],
        decoded_example['state/future/bbox_yaw'],
        decoded_example['state/future/velocity_x'],
        decoded_example['state/future/velocity_y']
    ], -1)

    gt_future_states = tf.concat([past_states, cur_states, future_states], 1)

    past_is_valid = decoded_example['state/past/valid'] > 0
    current_is_valid = decoded_example['state/current/valid'] > 0
    future_is_valid = decoded_example['state/future/valid'] > 0
    gt_future_is_valid = tf.concat(
        [past_is_valid, current_is_valid, future_is_valid], 1)

    # If a sample was not seen at all in the past, we declare the sample as
    # invalid.
    sample_is_valid = tf.reduce_any(
        tf.concat([past_is_valid, current_is_valid], 1), 1)

    inputs = {
        'input_states': input_states,
        'gt_future_states': gt_future_states,
        'gt_future_is_valid': gt_future_is_valid,
        'object_type': decoded_example['state/type'],
        'tracks_to_predict': decoded_example['state/tracks_to_predict'] > 0,
        'interactive_tracks_to_predict': decoded_example['state/objects_of_interest'] > 0,
        'sample_is_valid': sample_is_valid,
    }
    return inputs, decoded_example


def create_figure_and_axes(size_pixels=1000):
    """Initializes a unique figure and axes for plotting."""
    fig, ax = plt.subplots(1, 1)

    # Sets output image to pixel resolution.
    dpi = 100
    size_inches = size_pixels / dpi
    fig.set_size_inches([size_inches, size_inches])
    fig.set_dpi(dpi)
    fig.set_facecolor('white')
    ax.set_facecolor('white')
    ax.xaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='y', colors='black')
    fig.set_tight_layout(True)
    ax.grid(False)
    return fig, ax


def get_viewport(all_states, all_states_mask):
    """Gets the region containing the data.

    Args:
      all_states: states of agents as an array of shape [num_agents, num_steps,
        2].
      all_states_mask: binary mask of shape [num_agents, num_steps] for
        `all_states`.

    Returns:
      center_y: float. y coordinate for center of data.
      center_x: float. x coordinate for center of data.
      width: float. Width of data.
    """
    valid_states = all_states[all_states_mask]
    all_y = valid_states[..., 1]
    all_x = valid_states[..., 0]

    center_y = (np.max(all_y) + np.min(all_y)) / 2
    center_x = (np.max(all_x) + np.min(all_x)) / 2

    range_y = np.ptp(all_y)
    range_x = np.ptp(all_x)

    width = max(range_y, range_x)

    return center_y, center_x, width


def visualize_one_sample(decoded_example, inf_prediction, rea_prediction, args):
    # [num_agents, num_past_steps, 2] float32.
    past_states = tf.stack(
        [decoded_example['state/past/x'], decoded_example['state/past/y']],
        -1).numpy()
    past_states_mask = decoded_example['state/past/valid'].numpy() > 0.0

    # [num_agents, 1, 2] float32.
    current_states = tf.stack(
        [decoded_example['state/current/x'], decoded_example['state/current/y']],
        -1).numpy()
    current_states_mask = decoded_example['state/current/valid'].numpy() > 0.0

    # [num_agents, num_future_steps, 2] float32.
    future_states = tf.stack(
        [decoded_example['state/future/x'], decoded_example['state/future/y']],
        -1).numpy()
    future_states_mask = decoded_example['state/future/valid'].numpy() > 0.0

    # [num_points, 3] float32.
    roadgraph_xyz = decoded_example['roadgraph_samples/xyz'].numpy()

    num_agents, num_past_steps, _ = past_states.shape
    num_future_steps = future_states.shape[1]

    # [num_agens, num_past_steps + 1 + num_future_steps, depth] float32.
    all_states = np.concatenate([past_states, current_states, future_states], 1)

    # [num_agens, num_past_steps + 1 + num_future_steps] float32.
    all_states_mask = np.concatenate(
        [past_states_mask, current_states_mask, future_states_mask], 1)

    # Filter out non interactive states.
    state_ids = decoded_example['state/id'].numpy()
    interactive_mask = decoded_example['state/objects_of_interest'] > 0
    interactive_states = all_states[interactive_mask]
    interactive_states_mask = all_states_mask[interactive_mask]
    interactive_states_id = state_ids[interactive_mask]

    center_y, center_x, width = get_viewport(interactive_states, interactive_states_mask)
    width = width * 1.5

    # Create figure and axes.
    fig, ax = create_figure_and_axes(size_pixels=1000)

    # Plot roadgraph.
    rg_pts = roadgraph_xyz[:, :2].T
    ax.plot(rg_pts[0, :], rg_pts[1, :], 'k.', alpha=0.6, ms=1)

    agent1_xy_past = interactive_states[0, :11][interactive_states_mask[0, :11]]
    agent2_xy_past = interactive_states[1, :11][interactive_states_mask[1, :11]]
    agent1_xy_future = interactive_states[0, 11:][interactive_states_mask[0, 11:]]
    agent2_xy_future = interactive_states[1, 11:][interactive_states_mask[1, 11:]]

    # Plot agent observed position.
    ax.plot(
        agent1_xy_past[:, 0],
        agent1_xy_past[:, 1],
        linestyle="-",
        marker=None,
        alpha=0.8,
        linewidth=5,
        color='r',
    )
    ax.plot(
        agent2_xy_past[:, 0],
        agent2_xy_past[:, 1],
        linestyle="-",
        marker=None,
        alpha=0.8,
        linewidth=5,
        color='b',
    )

    # Plot agent future position.
    ax.plot(
        agent1_xy_future[:, 0],
        agent1_xy_future[:, 1],
        linestyle="-",
        marker=None,
        alpha=0.6,
        linewidth=5,
        color='lightcoral',
    )
    ax.plot(
        agent2_xy_future[:, 0],
        agent2_xy_future[:, 1],
        linestyle="-",
        marker=None,
        alpha=0.6,
        linewidth=5,
        color='c',
    )

    # Plot prediction samples.
    prediction_traj = inf_prediction['rst']
    prediction_id = inf_prediction['ids']

    # Make sure the order is correct.
    if prediction_id[0] == interactive_states_id[0]:
        agent1_xy_pred = prediction_traj[0]
        agent2_xy_pred = prediction_traj[1]
    else:
        agent1_xy_pred = prediction_traj[1]
        agent2_xy_pred = prediction_traj[0]

    # Replace marginal samples with reactor if exists.
    if rea_prediction is not None:
        reactor_prediction_traj = rea_prediction["rst"]
        reactor_id = rea_prediction["ids"]

        fontsize = 15
        if reactor_id == interactive_states_id[0]:
            agent1_xy_pred = reactor_prediction_traj[0]
            ax.text(agent1_xy_past[-1, 0], agent1_xy_past[-1, 1] + 1, "Reactor", fontsize=fontsize)
            ax.text(agent2_xy_past[-1, 0], agent2_xy_past[-1, 1] + 1, "Influencer", fontsize=fontsize)
        else:
            agent2_xy_pred = reactor_prediction_traj[0]
            ax.text(agent1_xy_past[-1, 0], agent1_xy_past[-1, 1] + 1, "Influencer", fontsize=fontsize)
            ax.text(agent2_xy_past[-1, 0], agent2_xy_past[-1, 1] + 1, "Reactor", fontsize=fontsize)

    prediction_sample_index = 0
    ax.plot(
        agent1_xy_pred[prediction_sample_index, :, 0],
        agent1_xy_pred[prediction_sample_index, :, 1],
        linestyle="-",
        marker=None,
        alpha=1,
        linewidth=2.5,
        color='darkmagenta',
    )
    ax.plot(
        agent2_xy_pred[prediction_sample_index, :, 0],
        agent2_xy_pred[prediction_sample_index, :, 1],
        linestyle="-",
        marker=None,
        alpha=1,
        linewidth=2.5,
        color='darkgreen',
    )

    # Plot starting positions.
    ax.plot(agent1_xy_past[-1, 0],
            agent1_xy_past[-1, 1],
            markersize=15,
            color="r",
            marker="*",
            markeredgecolor='black')
    ax.plot(agent2_xy_past[-1, 0],
            agent2_xy_past[-1, 1],
            markersize=15,
            color="b",
            marker="*",
            markeredgecolor='black')

    # Plot last predicted position.
    ax.plot(agent1_xy_pred[prediction_sample_index, -1, 0],
            agent1_xy_pred[prediction_sample_index, -1, 1],
            markersize=8,
            color="darkmagenta",
            marker="o")
    ax.plot(agent2_xy_pred[prediction_sample_index, -1, 0],
            agent2_xy_pred[prediction_sample_index, -1, 1],
            markersize=8,
            color="darkgreen",
            marker="o")

    # Set axes.  Should be at least 10m on a side and cover 160% of agents.
    size = max(10, width * 1.0)
    ax.axis([
        -size / 2 + center_x, size / 2 + center_x, -size / 2 + center_y,
        size / 2 + center_y
    ])
    ax.set_aspect('equal')

    scenario_id = decoded_example['scenario/id'].numpy()[0].decode("utf-8")
    os.makedirs(args.output_path, exist_ok=True)
    plt.savefig(os.path.join(args.output_path, '{}.png'.format(scenario_id)), bbox_inches='tight')
    plt.close()
    # import IPython; IPython.embed()


def visualize(args):
    dataset = tf.data.TFRecordDataset(os.path.join(args.input_path), compression_type='')

    # Load marginal prediction data, including both interactive agents.
    with open(args.influencer_prediction_path, 'rb') as f:
        influencer_prediction_data = pickle.load(f)

    # Load reactor prediction data.
    # This is optional, and the script will plot marginal predictions if this path is not provided.
    if args.reactor_prediction_path is not None:
        with open(args.reactor_prediction_path, 'rb') as f:
            reactor_prediction_data = pickle.load(f)
    else:
        reactor_prediction_data = None

    for data in dataset:
        parsed = tf.io.parse_single_example(data, features_description)
        scenario_id = parsed['scenario/id'].numpy()[0]

        # Visualize reactor prediction if it exists.
        if reactor_prediction_data is not None:
            if scenario_id in reactor_prediction_data and scenario_id in influencer_prediction_data:
                reactor_prediction = reactor_prediction_data[scenario_id]
                influencer_prediction = influencer_prediction_data[scenario_id]
                visualize_one_sample(parsed, influencer_prediction, reactor_prediction, args)
        elif scenario_id in influencer_prediction_data:
            influencer_prediction = influencer_prediction_data[scenario_id]
            if len(influencer_prediction["ids"]) == 2:
                visualize_one_sample(parsed, influencer_prediction, None, args)
        else:
            pass

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input_path', help='Path of input tf example.')
    parser.add_argument('-o', '--output_path', help='Path of output images to save.')
    parser.add_argument('-f', '--influencer_prediction_path', help='Path of influencer prediction pickle file.')
    parser.add_argument('-r', '--reactor_prediction_path', help='Path of reactor prediction pickle file.')

    args = parser.parse_args()

    visualize(args)


if __name__ == '__main__':
    main()
