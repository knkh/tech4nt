
# Building various datasets

 


    # To utilise  Full sample market data, use 
    --start_date 2026-01-01 \
    --end_date 2026-08-31  \

    # For demonstration purpose, datasets are generated for the period
    --start_date 2026-08-22 \
    --end_date 2026-05-31  \

    # windowed dataset for demo
    instr_01                  windows=1,251
instr_03                  windows=1,251
instr_04                  windows=1,251
instr_05                  windows=1,251
instr_08                  windows=1,251
instr_09                  windows=1,251
instr_10                  windows=1,251
instr_11                  windows=1,251
/Users/knkh/Documents/my_project/clone_files/data/part_2/build_windowed_dataset_umap.py:71: UserWarning: no explicit representation of timezones available for np.datetime64
  timestamp_all = np.asarray(timestamp_all, dtype="datetime64[ns]")
✅ UMAP dataset: (10008, 64, 30)

Note: Missing instruments do not have featured data or labeled data, hence no windowed dataset waas generated. 

    # train patchtst model - demo
    (venv_prj) ...../my_project  $ python train_patchtst_fair_validation.py 
  Model: PatchTST | device: mps
Dataset shape: (10008, 64, 30); labels: {'DOWN': np.int64(2734), 'FLAT': np.int64(4458), 'UP': np.int64(2816)}
Split: train=[0:6004), purge=76, validation=[6080:8006), test=[8082:10008)
Train class weights: {'DOWN': np.float32(1.1534), 'FLAT': np.float32(0.7302), 'UP': np.float32(1.1164)}
Epoch 01: train_loss=0.9483 validation_directional_f1=0.6424
Epoch 02: train_loss=0.5811 validation_directional_f1=0.7817
Epoch 03: train_loss=0.4638 validation_directional_f1=0.7907
Epoch 04: train_loss=0.4423 validation_directional_f1=0.7876
Epoch 05: train_loss=0.4362 validation_directional_f1=0.7968
Epoch 06: train_loss=0.4130 validation_directional_f1=0.7991
Epoch 07: train_loss=0.4123 validation_directional_f1=0.8002
Epoch 08: train_loss=0.3921 validation_directional_f1=0.7923
Epoch 09: train_loss=0.3826 validation_directional_f1=0.7794
Epoch 10: train_loss=0.3709 validation_directional_f1=0.7745
Epoch 11: train_loss=0.3569 validation_directional_f1=0.7837
Epoch 12: train_loss=0.3484 validation_directional_f1=0.7692
Epoch 13: train_loss=0.3327 validation_directional_f1=0.7763
Epoch 14: train_loss=0.3184 validation_directional_f1=0.7568
Epoch 15: train_loss=0.2974 validation_directional_f1=0.7499
Early stopping. Best epoch: 7.
Training: loss=0.3736  accuracy=0.8309  macro_f1=0.8353  directional_f1=0.8595
  DOWN precision=0.805 recall=0.933 f1=0.865 support=1662
  FLAT precision=0.885 recall=0.709 f1=0.787 support=2625
  UP   precision=0.798 recall=0.919 f1=0.854 support=1717
  Confusion matrix (rows = actual, columns = predicted)
                 Pred DOWN  Pred FLAT    Pred UP
  Actual DOWN        1,551        105          6
  Actual FLAT          372      1,860        393
  Actual UP              3        136      1,578
Validation: loss=0.5018  accuracy=0.7684  macro_f1=0.7746  directional_f1=0.8002
  DOWN precision=0.782 recall=0.828 f1=0.805 support=547
  FLAT precision=0.778 recall=0.676 f1=0.724 support=853
  UP   precision=0.744 recall=0.856 f1=0.796 support=526
  Confusion matrix (rows = actual, columns = predicted)
                 Pred DOWN  Pred FLAT    Pred UP
  Actual DOWN          453         92          2
  Actual FLAT          123        577        153
  Actual UP              3         73        450
FINAL UNTOUCHED TEST: loss=0.5062  accuracy=0.7664  macro_f1=0.7724  directional_f1=0.7917
  DOWN precision=0.696 recall=0.870 f1=0.773 support=486
  FLAT precision=0.802 recall=0.676 f1=0.734 support=911
  UP   precision=0.795 recall=0.826 f1=0.810 support=529
  Confusion matrix (rows = actual, columns = predicted)
                 Pred DOWN  Pred FLAT    Pred UP
  Actual DOWN          423         62          1
  Actual FLAT          183        616        112
  Actual UP              2         90        437
Saved comparable checkpoint and results to: /Users/knkh/Documents/my_project/clone_files/data/part_2/demo_data/ptst_cm_demo_validation

    # train tcn model - demo
    (venv_prj) ....../my_project  $ python train_patchtst_fair_validation.py 
Model: PatchTST | device: mps
Dataset shape: (10008, 64, 30); labels: {'DOWN': np.int64(2734), 'FLAT': np.int64(4458), 'UP': np.int64(2816)}
Split: train=[0:6004), purge=76, validation=[6080:8006), test=[8082:10008)
Train class weights: {'DOWN': np.float32(1.1534), 'FLAT': np.float32(0.7302), 'UP': np.float32(1.1164)}
Epoch 01: train_loss=0.9483 validation_directional_f1=0.6424
Epoch 02: train_loss=0.5811 validation_directional_f1=0.7817
Epoch 03: train_loss=0.4638 validation_directional_f1=0.7907
Epoch 04: train_loss=0.4423 validation_directional_f1=0.7876
Epoch 05: train_loss=0.4362 validation_directional_f1=0.7968
Epoch 06: train_loss=0.4130 validation_directional_f1=0.7991
Epoch 07: train_loss=0.4123 validation_directional_f1=0.8002
Epoch 08: train_loss=0.3921 validation_directional_f1=0.7923
Epoch 09: train_loss=0.3826 validation_directional_f1=0.7794
Epoch 10: train_loss=0.3709 validation_directional_f1=0.7745
Epoch 11: train_loss=0.3569 validation_directional_f1=0.7837
Epoch 12: train_loss=0.3484 validation_directional_f1=0.7692
Epoch 13: train_loss=0.3327 validation_directional_f1=0.7763
Epoch 14: train_loss=0.3184 validation_directional_f1=0.7568
Epoch 15: train_loss=0.2974 validation_directional_f1=0.7499
Early stopping. Best epoch: 7.
Training: loss=0.3736  accuracy=0.8309  macro_f1=0.8353  directional_f1=0.8595
  DOWN precision=0.805 recall=0.933 f1=0.865 support=1662
  FLAT precision=0.885 recall=0.709 f1=0.787 support=2625
  UP   precision=0.798 recall=0.919 f1=0.854 support=1717
  Confusion matrix (rows = actual, columns = predicted)
                 Pred DOWN  Pred FLAT    Pred UP
  Actual DOWN        1,551        105          6
  Actual FLAT          372      1,860        393
  Actual UP              3        136      1,578
Validation: loss=0.5018  accuracy=0.7684  macro_f1=0.7746  directional_f1=0.8002
  DOWN precision=0.782 recall=0.828 f1=0.805 support=547
  FLAT precision=0.778 recall=0.676 f1=0.724 support=853
  UP   precision=0.744 recall=0.856 f1=0.796 support=526
  Confusion matrix (rows = actual, columns = predicted)
                 Pred DOWN  Pred FLAT    Pred UP
  Actual DOWN          453         92          2
  Actual FLAT          123        577        153
  Actual UP              3         73        450
FINAL UNTOUCHED TEST: loss=0.5062  accuracy=0.7664  macro_f1=0.7724  directional_f1=0.7917
  DOWN precision=0.696 recall=0.870 f1=0.773 support=486
  FLAT precision=0.802 recall=0.676 f1=0.734 support=911
  UP   precision=0.795 recall=0.826 f1=0.810 support=529
  Confusion matrix (rows = actual, columns = predicted)
                 Pred DOWN  Pred FLAT    Pred UP
  Actual DOWN          423         62          1
  Actual FLAT          183        616        112
  Actual UP              2         90        437
Saved comparable checkpoint and results to: /Users/knkh/Documents/my_project/clone_files/data/part_2/demo_data/ptst_cm_demo_validation




