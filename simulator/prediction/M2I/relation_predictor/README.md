# M2I: From Factored Marginal Trajectory Prediction to Interactive Prediction

[Paper] [Project Page]

TODO:

- [ ] **Code Style** Eliminate global variables.
- [ ] **Data Loading** Forward a batch of examples by padding instead of looping through each example.
- [ ] **Structural** Support one-shot trajectory predictions.


## Prepare Your Dataset

Currently, we only support loading data from the [Waymo Open Dataset](https://waymo.com/open/data/motion/#). You can login and download any tf.Example proto file from the interactive validation/testing dataset for a quick testing. Note not all scenarios in the **training** dataset have the interactive flag. Loading those ones without the interactive flag will increase the time for loading data at each epoch.

## Quick Start

Requires:

* Python 3.6
* pytorch 1.6+

Install packages into a Conda environment (Cython, tensorflow, waymo-open-dataset, etc.):

``` bash
conda env create -f conda.cuda111.yaml
conda activate M2I
```

Download our prediction results or our pre-trained models from [Google Drive](https://drive.google.com/drive/folders/1QDWprt5osGimSlfqYZj1rqzr2HzCyeIs) and run these commands for a quick prediction. See more details about these models in the following sections.

### 1st: Relation Prediction
Download our [relation prediction results](hhttps://drive.google.com/drive/folders/1QDWprt5osGimSlfqYZj1rqzr2HzCyeIs) on interactive validation dataset (vehicle to vehicle only) or run the following command to predict with the pre-trained relation model (also trained with vehicle to vehicle only):

```  bash
OUTPUT_DIR=waymo.r.densetnt.relation.v2v.wd_p3; \
DATA_DIR=./validation_interactive/; \
RELATION_GT_DIR=./validation_interactive_gt_relations.pickle; \
python -m src.run --waymo --data_dir ${DATA_DIR} \
--config relation.yaml --output_dir ${OUTPUT_DIR} \
--future_frame_num 80 \
--relation_file_path ${RELATION_GT_DIR} --agent_type vehicle \
--distributed_training 1 -e --nms_threshold 7.2 \
--validation_model 25 --relation_pred_threshold 0.9
```

### 2nd: Marginal Prediction
Download our [marginal prediction results](https://drive.google.com/drive/folders/1QDWprt5osGimSlfqYZj1rqzr2HzCyeIs) on interactive validation dataset (vehicle only) or run the following command to predict with the pre-trained marginal prediction model (trained with vehicle trajectories):

```  bash
Todo: insert marginal prediction command 
```

### 3rd: Conditional Prediction
Download our pre-trained conditional prediction model to predict trajectories of the reactors by running:

```  bash
OUTPUT_DIR=waymo.rdensetnt.reactor.Tgt-Rgt.raster_inf.wd_p3.v2v; \
DATA_DIR=./validation_interactive/; \
RELATION_GT_DIR=./validation_interactive_gt_relations.pickle; \
RELATION_PRED_DIR=./thresholdP9.r.densetnt.relation.v2v.wd_p3.VAL; \
INFLUENCER_PRED_DIR=./validation_interactive_v_rdensetnt_full.pickle; \
python -m src.run --waymo --data_dir ${DATA_DIR} \
--output_dir ${OUTPUT_DIR} --config conditional_pred.yaml \
--relation_file_path ${RELATION_GT_DIR} \
--relation_pred_file_path ${RELATION_PRED_DIR} \
--influencer_pred_file_path ${INFLUENCER_PRED_DIR} \
--future_frame_num 80 \
-e --eval_rst_saving_number 0 \
--eval_exp_path ${RESULT_EXPORT_PATH}
```

Change the RELATION_PRED_DIR to the directory of your relation prediction result and change INFLUENCER_PRED_DIR to the directory of your marginal prediction result. This command will output 6 predictions conditioned on #eval_rst_saving_number of the influencer predictions. Change eval_rst_saving_number from 0 to 5 to get 6 groups of conditional predictions.


## Performance

Results of different models on vehicle motion forecasting:

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Model</th>
    <th class="tg-0pky">Type</th>
    <th class="tg-0pky">minFDE</th>
    <th class="tg-0pky">MR</th>
    <th class="tg-0pky">mAP</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky" rowspan="3">Validation</td>
    <td class="tg-0pky">Vehicle</td>
    <td class="tg-0pky">5.49</td>
    <td class="tg-0pky">0.55</td>
    <td class="tg-0pky">0.18</td>
  </tr>
  <tr>
    <td class="tg-0pky">Pedstrian</td>
    <td class="tg-0pky">3.61</td>
    <td class="tg-0pky">0.60</td>
    <td class="tg-0pky">0.06</td>
  </tr>
  <tr>
    <td class="tg-0pky">Cyclist</td>
    <td class="tg-0pky">6.26</td>
    <td class="tg-0pky">0.73</td>
    <td class="tg-0pky">0.04</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="3">Test</td>
    <td class="tg-0pky">Vehicle</td>
    <td class="tg-0pky">5.65</td>
    <td class="tg-0pky">0.57</td>
    <td class="tg-0pky">0.16</td>
  </tr>
  <tr>
    <td class="tg-0pky">Pedstrian</td>
    <td class="tg-0pky">3.73</td>
    <td class="tg-0pky">0.60</td>
    <td class="tg-0pky">0.06</td>
  </tr>
  <tr>
    <td class="tg-0pky">Cyclist</td>
    <td class="tg-0pky">6.16</td>
    <td class="tg-0pky">0.74</td>
    <td class="tg-0pky">0.03</td>
  </tr>
</tbody>
</table>

## Training

You need to filter the interactive scenarios from the training dataset for training since Waymo Open Dataset does not provide an interactive specific training dataset. Check ```'state/objects_of_interest'``` against each agent to spot interactive scenarios. We used all scenarios with two agents with positive ```'state/objects_of_interest'``` for training relation prediction and conditional trajectory prediction.

### Training Marginal Predictor

We trained our marginal predictor on the whole training dataset. Use the following command to train your own marginal predictor:

```bash
#TODO
```

### Training Relation Predictor

Download the ground truth relation data and train vehicle2vehicle relation predictor first. We recommend modeling v2v, v2p, v2c separately with three separate training. Change the flag ```pair_vv```, ```pair_vc```, ```pair_vp``` to change the scenarios loaded.

```bash
DATA_DIR=./training_interactive/; \
RELATION_GT_DIR=./training_interactive_gt_relations.pickle; \
python -m src.run --do_train --waymo --data_dir ${DATA_DIR} \
--output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 16 --sub_graph_batch_size 1024  --core_num 16 \
--future_frame_num 80 \
--relation_file_path ${RELATION_GT_DIR} --weight_decay 0.3 --agent_type vehicle \
--other_params train_relation pair_vv \
l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 raster \
--distributed_training 8
```

### Training Conditional Predictor

Train the conditional predictor with ground truth influencer trajectory and relations. Change agent type if you want to predict another type of reactor. Change the flag ```pair_vv``` if you want to predict against different types of interactions.

```bash
DATA_DIR=./training_interactive/; \
RELATION_GT_DIR=./training_interactive_gt_relations.pickle; \
python -m src.run --do_train --waymo --data_dir ${DATA_DIR} \
--output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 64 --sub_graph_batch_size 4096  --core_num 10 \
--future_frame_num 80 --agent_type vehicle \
--relation_file_path ${RELATION_GT_DIR} --weight_decay 0.3 \
--infMLP 8 --other_params train_reactor gt_relation_label gt_influencer_traj pair_vv raster_inf raster \
l1_loss densetnt goals_2D enhance_global_graph laneGCN point_sub_graph laneGCN-4 stride_10_2 \
--distributed_training 8
```

# Citation

If you found this useful in your research, please consider citing

...



