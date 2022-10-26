def generate_protobuf(output_dir, file_name,
                      prediction_trajectory_list, prediction_score_list,
                      ground_truth_trajectory_list, ground_truth_is_valid_list, object_type_list, scenario_id_list, object_id_list):
    submission = motion_submission_pb2.MotionChallengeSubmission()
    submission.account_name = 'gujunru123@gmail.com'
    submission.unique_method_name = 'Anonymous610'
    submission.submission_type = motion_submission_pb2.MotionChallengeSubmission.SubmissionType.MOTION_PREDICTION

    for prediction_trajectory, prediction_score, \
        ground_truth_trajectory, ground_truth_is_valid, object_type, scenario_id, object_id in \
            zip(prediction_trajectory_list, prediction_score_list,
                ground_truth_trajectory_list, ground_truth_is_valid_list, object_type_list, scenario_id_list, object_id_list):

        predict_num = len(prediction_trajectory)
        # prediction_set = motion_submission_pb2.PredictionSet()

        scenario_prediction = submission.scenario_predictions.add()
        prediction_set = scenario_prediction.single_predictions
        scenario_prediction.scenario_id = scenario_id

        for i in range(predict_num):

            # SingleObjectPrediction
            prediction = prediction_set.predictions.add()
            prediction.object_id = object_id[i]

            for k in range(6):
                # ScoredTrajectory
                scored_trajectory = prediction.trajectories.add()
                scored_trajectory.confidence = prediction_score[i, k]
                trajectory = scored_trajectory.trajectory

                interval = 5
                traj = prediction_trajectory[i, k, (interval - 1)::interval, :]
                trajectory.center_x[:] = traj[:, 0].numpy().tolist()
                trajectory.center_y[:] = traj[:, 1].numpy().tolist()

    path = os.path.join(output_dir, file_name)
    with open(path, "wb") as f:
        f.write(submission.SerializeToString())

    os.system(f'tar -zcvf {path}.tar.gz {path}')
