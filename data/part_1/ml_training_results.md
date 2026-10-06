

# Training results: two models trained on diabetes data

Dataset shape : (768, 9)
Target counts :
Outcome
0    500
1    268
Name: count, dtype: int64


====================================================
  Logistic Regression
====================================================
  Accuracy : 0.7078  (70.78%)

                 precision    recall  f1-score   support

No Diabetes (0)      0.752     0.820     0.785       100
   Diabetes (1)      0.600     0.500     0.545        54

       accuracy                          0.708       154
      macro avg      0.676     0.660     0.665       154
   weighted avg      0.699     0.708     0.701       154

  Confusion matrix saved → confusion_Logistic_Regression.png

====================================================
  KNN (k=5)
====================================================
  Accuracy : 0.7532  (75.32%)

                 precision    recall  f1-score   support

No Diabetes (0)      0.798     0.830     0.814       100
   Diabetes (1)      0.660     0.611     0.635        54

       accuracy                          0.753       154
      macro avg      0.729     0.721     0.724       154
   weighted avg      0.750     0.753     0.751       154

  Confusion matrix saved → confusion_KNN_k5.png

====================================================
  SUMMARY — Model Accuracy Comparison
====================================================
                     Accuracy Accuracy %
KNN (k=5)            0.753247     75.32%
Logistic Regression  0.707792     70.78%

Comparison chart saved → model_comparison.png


