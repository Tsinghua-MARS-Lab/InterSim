# P4P: Conflict-Aware Motion Prediction for Planning in Autonomous Driving

[Paper](https://arxiv.org/abs/2211.01634)

We used codes in this repository for the experiments to test and prove the importance of modeling conflict relations. 

These codes rely on the simulator of InterSim to load data, iterate through time steps, and visualization. You need to install InterSim before continue.

Also install sklearn and tqdm by pip install or following their website's instructions.

```
pip install tqdm
pip install scikit-learn
```

### Todos

- [ ] Add Instructions to Generate Auto-Labeling Ground Truth Relation Labels 
- [ ] Support Relation Prediction Metrics for NuPlan Dataset
- [ ] Support Metrics Computing for Training Set of the Waymo Motion Prediction Dataset
- [ ] Compute Relation Prediction Results for the MTR Predictor

## Gauging Relation Prediction Results

We provide an auto-labeling ground truth relation of the interactive validation set of the Waymo Motion Prediction Dataset.

To test your predictor, you need to export your prediction results in a pickle file first. After that,
run the following command to measure relation prediction metrics (recall, precision, and accuracy) for your prediction results.

The script is 

```
python compute_relation_metrics.py --prediction_result your_trajectory_prediction.pkl --gt_relation gt_relations/gt_direct_relation_WOMD_validation_interactive.pickle 
```

## Other Useful functions with InterSim

You can visualize the ground truth labels with InterSim. Or generate your own ground truth label with `simulator/interactions/detect_relations.py`

