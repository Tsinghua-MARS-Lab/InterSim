## Generate protobufs

### Validation Interactive
Command usage:
```bash
python scripts/generate_protobuf.py -h
```

Combine marginal prediction with v2v (P9) and x2v (P5) reactor prediction:
```bash
python scripts/generate_protobuf.py -i label_files/validation_interactive_rdensetnt_full.pickle --use_prediction --v2v_pred label_files/6x6_validation_interactive_reactor_Tgt_Rgt_p3.x2v.RthesholdP9_model30_1pred.pickle --x2v_pred label_files/6x6_validation_interactive_Tgt_Rgt_rasterInf_x2v_RthesholdP5.2c2p_model30.pickle -d val_reactor_v2v_P9_x2v_P5 -o vi_reactor_v2v_P9_x2v_P5
```

Replace marginal with 3s and 5s prediction:
```bash
python scripts/generate_protobuf.py -i label_files/validation_interactive_rdensetnt_full.pickle --use_prediction --v2v_pred label_files/6x6_validation_interactive_reactor_Tgt_Rgt_p3.x2v.RthesholdP9_model30_1pred.pickle --marginal_3s_pred label_files/validation_interactive_3s_rdensetnt_2.5.pickle --marginal_5s_pred label_files/validation_interactive_5s_rdensetnt_5.5.pickle --x2v_pred label_files/6x6_validation_interactive_Tgt_Rgt_rasterInf_x2v_RthesholdP5.2c2p_model30.pickle -d val_reactor_v2v_P9_x2v_P5_marg_3s2.5_5s5.5_marg_score -o vi_reactor_v2v_P9_x2v_P5_marg_3s2.5_5s5.5_marg_score
```

Replace reactor with 3s and 5s reactor prediction [THIS DOES NOT IMPROVE RESULTS]:
```bash
python scripts/generate_protobuf.py -i label_files/validation_interactive_rdensetnt_full.pickle --use_prediction --v2v_pred label_files/6x6_validation_interactive_reactor_Tgt_Rgt_p3.x2v.RthesholdP9_model30_1pred.pickle --v2v_3s_pred label_files/merged_3s_validation_interactive_reactor_Tgt_Rgt_RthresholdP9_v2v_1pred.pickle --v2v_5s_pred label_files/merged_5s_validation_interactive_reactor_Tgt_Rgt_RthresholdP9_v2v_1pred.pickle -d val_reactor_v2v_P9_3s5s_raw -o vi_reactor_v2v_P9_3s5s_raw
```

### Testing Interactive
```bash
python scripts/generate_protobuf.py -i label_files/testing_interactive_rdensetnt_full.pickle --use_prediction --v2v_pred label_files/fixed_merged_testing_interactive_reactor_Tgt_Rgt_p3.v2v.RthesholdP9_model30_1pred.pickle --marginal_3s_pred label_files/testing_interactive_3s_rdensetnt_2.5.pickle --marginal_5s_pred label_files/testing_interactive_5s_rdensetnt_5.5.pickle --x2v_pred label_files/6x6_test_interactive_Tgt_Rgt_rasterInf_x2v_RthresholdP5P5.2c2p_model30.pickle -o ti_reactor_v2v_P9_x2v_P5_marg_3s2.5_5s5.5_marg_score_fixed_no_descp
```


