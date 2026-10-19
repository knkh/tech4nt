
# patchtst training results

 $ python train_patchtst_fair_validation.py 

Model: PatchTST | device: mps
Dataset shape: (74240, 64, 30); labels: {'DOWN': np.int64(23173), 'FLAT': np.int64(28598), 'UP': np.int64(22469)}
Split: train=[0:44544), purge=76, validation=[44620:59392), test=[59468:74240)
Train class weights: {'DOWN': np.float32(1.0819), 'FLAT': np.float32(0.7909), 'UP': np.float32(1.1272)}
Epoch 01: train_loss=0.5786 validation_directional_f1=0.8294
Epoch 02: train_loss=0.4786 validation_directional_f1=0.8249
Epoch 03: train_loss=0.4706 validation_directional_f1=0.8270
Epoch 04: train_loss=0.4670 validation_directional_f1=0.8267
Epoch 05: train_loss=0.4633 validation_directional_f1=0.8281
Epoch 06: train_loss=0.4618 validation_directional_f1=0.8235
Epoch 07: train_loss=0.4584 validation_directional_f1=0.8245
Epoch 08: train_loss=0.4560 validation_directional_f1=0.8253
Epoch 09: train_loss=0.4541 validation_directional_f1=0.8253
Early stopping. Best epoch: 1.
Training: loss=0.4692  accuracy=0.7832  macro_f1=0.7884  directional_f1=0.8229
  DOWN precision=0.786 recall=0.871 f1=0.826 support=13385
  FLAT precision=0.773 recall=0.673 f1=0.720 support=18311
  UP   precision=0.792 recall=0.849 f1=0.820 support=12848
  Confusion matrix (rows = actual, columns = predicted)
                 Pred DOWN  Pred FLAT    Pred UP
  Actual DOWN       11,657      1,707         21
  Actual FLAT        3,145     12,322      2,844
  Actual UP             31      1,908     10,909
Validation: loss=0.4699  accuracy=0.7780  macro_f1=0.7801  directional_f1=0.8294
  DOWN precision=0.794 recall=0.860 f1=0.826 support=4779
  FLAT precision=0.727 recall=0.642 f1=0.681 support=5444
  UP   precision=0.812 recall=0.855 f1=0.833 support=4549
  Confusion matrix (rows = actual, columns = predicted)
                 Pred DOWN  Pred FLAT    Pred UP
  Actual DOWN        4,110        667          2
  Actual FLAT        1,053      3,493        898
  Actual UP             12        647      3,890
FINAL UNTOUCHED TEST: loss=0.4777  accuracy=0.7761  macro_f1=0.7713  directional_f1=0.8368
  DOWN precision=0.815 recall=0.873 f1=0.843 support=4967
  FLAT precision=0.673 recall=0.610 f1=0.640 support=4786
  UP   precision=0.823 recall=0.839 f1=0.831 support=5019
  Confusion matrix (rows = actual, columns = predicted)
                 Pred DOWN  Pred FLAT    Pred UP
  Actual DOWN        4,335        622         10
  Actual FLAT          968      2,921        897
  Actual UP             14        796      4,209
Saved comparable checkpoint and results to: /Users/knkh/Documents/my_project/clone_files/data/part_2/ptst_cm_validation


# tcn train results

python train_tcn_fair_validation.py 

odel: TCN | device: mps
Dataset shape: (74240, 64, 30); labels: {'DOWN': np.int64(23173), 'FLAT': np.int64(28598), 'UP': np.int64(22469)}
Split: train=[0:44544), purge=76, validation=[44620:59392), test=[59468:74240)
Train class weights: {'DOWN': np.float32(1.0819), 'FLAT': np.float32(0.7909), 'UP': np.float32(1.1272)}
Epoch 01: train_loss=0.5440 validation_directional_f1=0.8262
Epoch 02: train_loss=0.4808 validation_directional_f1=0.8230
Epoch 03: train_loss=0.4695 validation_directional_f1=0.8255
Epoch 04: train_loss=0.4630 validation_directional_f1=0.8258
Epoch 05: train_loss=0.4572 validation_directional_f1=0.8262
Epoch 06: train_loss=0.4518 validation_directional_f1=0.8279
Epoch 07: train_loss=0.4490 validation_directional_f1=0.8273
Epoch 08: train_loss=0.4452 validation_directional_f1=0.8264
Epoch 09: train_loss=0.4419 validation_directional_f1=0.8289
Epoch 10: train_loss=0.4350 validation_directional_f1=0.8258
Epoch 11: train_loss=0.4317 validation_directional_f1=0.8255
Epoch 12: train_loss=0.4261 validation_directional_f1=0.8219
Epoch 13: train_loss=0.4226 validation_directional_f1=0.8227
Epoch 14: train_loss=0.4161 validation_directional_f1=0.8228
Epoch 15: train_loss=0.4115 validation_directional_f1=0.8229
Epoch 16: train_loss=0.4062 validation_directional_f1=0.8226
Epoch 17: train_loss=0.3994 validation_directional_f1=0.8204
Early stopping. Best epoch: 9.
Training: loss=0.4255  accuracy=0.8058  macro_f1=0.8106  directional_f1=0.8410
  DOWN precision=0.817 recall=0.870 f1=0.843 support=13385
  FLAT precision=0.802 recall=0.704 f1=0.750 support=18311
  UP   precision=0.799 recall=0.883 f1=0.839 support=12848
  Confusion matrix (rows = actual, columns = predicted)
                 Pred DOWN  Pred FLAT    Pred UP
  Actual DOWN       11,649      1,716         20
  Actual FLAT        2,576     12,895      2,840
  Actual UP             26      1,471     11,351
Validation: loss=0.4696  accuracy=0.7778  macro_f1=0.7798  directional_f1=0.8289
  DOWN precision=0.805 recall=0.852 f1=0.828 support=4779
  FLAT precision=0.726 recall=0.642 f1=0.682 support=5444
  UP   precision=0.801 recall=0.861 f1=0.830 support=4549
  Confusion matrix (rows = actual, columns = predicted)
                 Pred DOWN  Pred FLAT    Pred UP
  Actual DOWN        4,074        699          6
  Actual FLAT          979      3,497        968
  Actual UP             11        620      3,918
FINAL UNTOUCHED TEST: loss=0.4744  accuracy=0.7795  macro_f1=0.7746  directional_f1=0.8395
  DOWN precision=0.830 recall=0.860 f1=0.845 support=4967
  FLAT precision=0.680 recall=0.613 f1=0.645 support=4786
  UP   precision=0.811 recall=0.858 f1=0.834 support=5019
  Confusion matrix (rows = actual, columns = predicted)
                 Pred DOWN  Pred FLAT    Pred UP
  Actual DOWN        4,274        678         15
  Actual FLAT          865      2,935        986
  Actual UP             11        702      4,306
Saved comparable checkpoint and results to: /Users/knkh/Documents/my_project/clone_files/data/part_2/tcn_cm_validation


