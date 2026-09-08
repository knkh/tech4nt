
# compare_classifiers.py
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
import os

# 1. Load dataset
file_path = os.path.expanduser("~/Documents/pybase/datasets_all/diabetes_clean.csv")
with open(file_path, "r") as file:
   df = pd.read_csv(file)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 2. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale features (important for some models like KNN, SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Define classifiers
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (linear)": SVC(kernel="linear")
}

# 5. Train & evaluate each
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n🔹 {name} Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred))

'''# 6. Visualize accuracy comparison
plt.bar(results.keys(), results.values())
plt.xticks(rotation=30)
plt.ylabel("Accuracy")
plt.title("Classifier Comparison on Iris Dataset")
plt.show()
#'''

                # outcome
'''
knkh ~/Documents  $ python3 pybase/ML/supervised_classify.py 

🔹 Logistic Regression Accuracy: 0.75
              precision    recall  f1-score   support

           0       0.80      0.83      0.81        99
           1       0.67      0.62      0.64        55

    accuracy                           0.75       154
   macro avg       0.73      0.72      0.73       154
weighted avg       0.75      0.75      0.75       154


🔹 KNN (k=5) Accuracy: 0.72
              precision    recall  f1-score   support

           0       0.80      0.75      0.77        99
           1       0.60      0.67      0.63        55

    accuracy                           0.72       154
   macro avg       0.70      0.71      0.70       154
weighted avg       0.73      0.72      0.72       154


🔹 Decision Tree Accuracy: 0.72
              precision    recall  f1-score   support

           0       0.79      0.78      0.78        99
           1       0.61      0.62      0.61        55

    accuracy                           0.72       154
   macro avg       0.70      0.70      0.70       154
weighted avg       0.72      0.72      0.72       154


🔹 Random Forest Accuracy: 0.73
              precision    recall  f1-score   support

           0       0.80      0.79      0.79        99
           1       0.62      0.64      0.63        55

    accuracy                           0.73       154
   macro avg       0.71      0.71      0.71       154
weighted avg       0.73      0.73      0.73       154


🔹 SVM (linear) Accuracy: 0.75
              precision    recall  f1-score   support

           0       0.80      0.83      0.81        99
           1       0.67      0.62      0.64        55

    accuracy                           0.75       154
   macro avg       0.73      0.72      0.73       154
weighted avg       0.75      0.75      0.75       154


'''

                # interpretation of the results
'''

🧾 1️⃣ Overview — Summary Table
Model	Accuracy	Precision (class 1)	Recall (class 1)	F1-score (class 1)
Logistic Regression	0.75	0.67	0.62	0.64
KNN (k=5)	0.72	0.60	0.67	0.63
Decision Tree	0.72	0.61	0.62	0.61
Random Forest	0.73	0.62	0.64	0.63
SVM (linear)	0.75	0.67	0.62	0.64
🧩 2️⃣ What the metrics mean
Term	Meaning	Why it matters
Accuracy	% of total predictions that are correct	Good quick measure but misleading if classes are imbalanced
Precision (for class 1)	Of those predicted “diabetic”, how many are actually diabetic	Important when false positives (wrongly predicting diabetes) are costly
Recall (for class 1)	Of actual diabetics, how many were correctly identified	Important when missing diabetics (false negatives) is serious
F1-score	Harmonic mean of precision and recall	Best single summary when you care about both
📊 3️⃣ Key interpretations

Overall accuracy (~0.72–0.75)
→ All five models perform similarly — none is dramatically better.
The dataset may be moderately separable but not easy (typical for diabetes data).

Class imbalance

You have 99 (non-diabetic) vs 55 (diabetic).

The imbalance makes models biased toward predicting non-diabetic (0).

Notice: Precision/recall for class 0 are higher than for class 1.

Best models

Logistic Regression and SVM are top performers here (0.75 accuracy, good balance).

They also have stable precision and recall — i.e., not overfitting.

Random Forest is slightly behind but might improve with tuning.

KNN

Performs decently (0.72) but can be improved by scaling and tuning k.

KNN’s recall for class 1 is the best (0.67), meaning it detects diabetics slightly better than others.

Decision Tree

Usually less stable — can overfit unless pruned.

Similar accuracy, but less reliable for generalization.

🔍 4️⃣ What this means practically
Goal	Best Choice	Why
Balanced general performance	Logistic Regression / SVM	Good mix of interpretability & accuracy
Catching as many diabetics as possible (high recall)	KNN	Slightly better recall (0.67)
Fast, explainable model	Logistic Regression	Simple coefficients = medically interpretable
Nonlinear relationships	Random Forest	Captures more complexity, can improve with tuning
'''
# Cross-validation – instead of one random 80/20 split, 
# use 5-fold CV for reliability.

from sklearn.model_selection import cross_val_score
scores = cross_val_score(LogisticRegression(), X, y, cv=5)
print(scores.mean())

'''
🔹 Logistic Regression Accuracy: 0.75
              precision    recall  f1-score   support

           0       0.80      0.83      0.81        99
           1       0.67      0.62      0.64        55

    accuracy                           0.75       154
   macro avg       0.73      0.72      0.73       154
weighted avg       0.75      0.75      0.75       154


🔹 KNN (k=5) Accuracy: 0.72
              precision    recall  f1-score   support

           0       0.80      0.75      0.77        99
           1       0.60      0.67      0.63        55

    accuracy                           0.72       154
   macro avg       0.70      0.71      0.70       154
weighted avg       0.73      0.72      0.72       154


🔹 Decision Tree Accuracy: 0.72
              precision    recall  f1-score   support

           0       0.79      0.78      0.78        99
           1       0.61      0.62      0.61        55

    accuracy                           0.72       154
   macro avg       0.70      0.70      0.70       154
weighted avg       0.72      0.72      0.72       154


🔹 Random Forest Accuracy: 0.73
              precision    recall  f1-score   support

           0       0.80      0.79      0.79        99
           1       0.62      0.64      0.63        55

    accuracy                           0.73       154
   macro avg       0.71      0.71      0.71       154
weighted avg       0.73      0.73      0.73       154


🔹 SVM (linear) Accuracy: 0.75
              precision    recall  f1-score   support

           0       0.80      0.83      0.81        99
           1       0.67      0.62      0.64        55

    accuracy                           0.75       154
   macro avg       0.73      0.72      0.73       154
weighted avg       0.75      0.75      0.75       154

/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:473: ConvergenceWarning: lbfgs failed to converge after 100 iteration(s) (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

Increase the number of iterations to improve the convergence (max_iter=100).
You might also want to scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:473: ConvergenceWarning: lbfgs failed to converge after 100 iteration(s) (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

Increase the number of iterations to improve the convergence (max_iter=100).
You might also want to scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:473: ConvergenceWarning: lbfgs failed to converge after 100 iteration(s) (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

Increase the number of iterations to improve the convergence (max_iter=100).
You might also want to scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:473: ConvergenceWarning: lbfgs failed to converge after 100 iteration(s) (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

Increase the number of iterations to improve the convergence (max_iter=100).
You might also want to scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/sklearn/linear_model/_logistic.py:473: ConvergenceWarning: lbfgs failed to converge after 100 iteration(s) (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT

Increase the number of iterations to improve the convergence (max_iter=100).
You might also want to scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
0.7656735421441303
'''

                        # convergence warning
'''
✅ Root causes (why the warning still shows)

You set max_iter=1000 in the models dict, so the model used in your manual .fit() 
calls should be fine.

But your cross_val_score(LogisticRegression(), X, y, cv=5) creates a fresh 
LogisticRegression() with default max_iter=100 and no scaling — so that call will 
often trigger the convergence warning.

Also, if you ever fit a logistic model on unscaled features, lbfgs can require many 
iterations to converge — scaling helps a lot.

✅ How to check if a fitted LogisticRegression converged

After fit(), check:

print("n_iter_:", model.n_iter_)   # array or scalar of iterations used


If any value in n_iter_ equals your max_iter then it likely hit the limit and didn't 
converge.

✅ Fixes (recommended)

Scale inside a pipeline so each CV fold is scaled correctly (prevents data leakage).

Use the same estimator (with max_iter increased) for CV.

Optionally try different solvers (liblinear, saga) or stronger regularization 
(smaller C) if convergence still fails.
'''      


        # ✅ Corrected Full Script — with Pipelines and Cross-Validation

        # Here’s a version of your earlier code, properly structured and warning-free:

# compare_classifiers_with_pipelines.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 1. Load dataset
file_path = os.path.expanduser("~/Documents/pybase/datasets_all/diabetes_clean.csv")
df = pd.read_csv(file_path)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 2. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Define models using Pipelines 
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (linear)": make_pipeline(StandardScaler(), SVC(kernel="linear"))
}

# 4. Train & evaluate each model
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n🔹 {name} Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred))

# 5. Cross-validation example (only Logistic Regression shown here)
log_reg_pipe = models["Logistic Regression"]
scores = cross_val_score(log_reg_pipe, X, y, cv=5, scoring="accuracy", n_jobs=-1)
print("\nCross-Validation Scores (Logistic Regression):", scores)
print("Mean CV Accuracy:", scores.mean())

# Check convergence
log_reg_model = log_reg_pipe.named_steps["logisticregression"]
print("Iterations used in final logistic regression:", log_reg_model.n_iter_)

"""🧠 Key Takeaways
Concept	Meaning	Why Important
Pipeline	Sequential chain of preprocessing + model	Prevents data leakage; simplifies 
cross-validation
cv=5	5-fold cross-validation	Gives stable, fair model performance
max_iter=1000	Max optimization steps	Prevents solver from getting stuck; can be 
increased safely
"""

                    # output
'''
🔹 Logistic Regression Accuracy: 0.75
              precision    recall  f1-score   support

           0       0.80      0.83      0.81        99
           1       0.67      0.62      0.64        55

    accuracy                           0.75       154
   macro avg       0.73      0.72      0.73       154
weighted avg       0.75      0.75      0.75       154


🔹 KNN (k=5) Accuracy: 0.72
              precision    recall  f1-score   support

           0       0.80      0.75      0.77        99
           1       0.60      0.67      0.63        55

    accuracy                           0.72       154
   macro avg       0.70      0.71      0.70       154
weighted avg       0.73      0.72      0.72       154


🔹 Decision Tree Accuracy: 0.73
              precision    recall  f1-score   support

           0       0.79      0.78      0.79        99
           1       0.61      0.64      0.62        55

    accuracy                           0.73       154
   macro avg       0.70      0.71      0.71       154
weighted avg       0.73      0.73      0.73       154


🔹 Random Forest Accuracy: 0.75
              precision    recall  f1-score   support

           0       0.81      0.79      0.80        99
           1       0.64      0.67      0.65        55

    accuracy                           0.75       154
   macro avg       0.73      0.73      0.73       154
weighted avg       0.75      0.75      0.75       154


🔹 SVM (linear) Accuracy: 0.75
              precision    recall  f1-score   support

           0       0.80      0.83      0.81        99
           1       0.67      0.62      0.64        55

    accuracy                           0.75       154
   macro avg       0.73      0.72      0.73       154
weighted avg       0.75      0.75      0.75       154


Cross-Validation Scores (Logistic Regression): [0.75974026 0.74675325 0.78571429 0.79738562 0.77124183]
Mean CV Accuracy: 0.7721670486376369
Iterations used in final logistic regression: [9]
'''

                    # interpretation of results
'''
🧩 1. What the results tell you overall
Model	Accuracy	Comments
Logistic Regression	0.75	Solid baseline; interpretable and stable
KNN (k=5)	0.72	Slightly lower; sensitive to scaling & local patterns
Decision Tree	0.73	Slight overfitting likely; simple split rules
Random Forest	0.75	Matches logistic regression; stable and robust
SVM (linear)	0.75	Consistent with logistic regression; margin-based

All are within ~0.72–0.75, meaning:

Your dataset (diabetes.csv) is moderately separable — not easy to classify, but not random either.

An accuracy of ~0.75 is quite typical for this dataset — it’s actually close to published benchmarks on the Pima Indians Diabetes dataset.

🧮 2. Understanding the Cross-Validation Results
Cross-Validation Scores (Logistic Regression):
[0.7597, 0.7467, 0.7857, 0.7974, 0.7712]
Mean CV Accuracy: 0.7721


That’s k-fold cross-validation (cv=5):

The data is split into 5 folds.

The model is trained 5 times — each time leaving one fold for testing and using the other 4 for training.

You get 5 scores — showing how consistent performance is across different subsets.

Here:

✔️ All folds are around 0.75–0.80 — stable and reliable.
❌ No wide fluctuation — so no data leakage or overfitting issues.

So your model generalizes well.

⚙️ 3. Iterations Used: [9]

That’s excellent.
It means the logistic regression solver found an optimal solution in just 9 iterations — extremely efficient, thanks to:

Data scaling via StandardScaler()

Reasonable feature magnitudes

No multicollinearity causing instability

No warnings, no convergence problems — ✅ success.

🧠 4. Interpreting Precision, Recall, F1

For the positive class (Outcome = 1):

Metric	Meaning	Value	Comment
Precision	Out of predicted diabetics, how many are actually diabetics	~0.67	Acceptable
Recall	Out of all actual diabetics, how many were caught	~0.62	Could improve
F1-score	Balance between precision & recall	~0.64	Reasonable for medical data

These values show:

Model is better at identifying non-diabetic (0) cases than diabetic (1) cases — which is typical due to class imbalance in diabetes datasets (more 0s than 1s).

🔍 5. How to Improve Next

Handle Class Imbalance

LogisticRegression(class_weight='balanced', max_iter=2000)


→ This gives more weight to minority (diabetic) class.

Feature Engineering

Add interaction terms like BMI * Age

Try log-transforming skewed features (like Insulin)

Hyperparameter Tuning

Use GridSearchCV for KNN (vary k)

Try different tree depths for DecisionTree / RandomForest

Compare ROC–AUC

Accuracy alone is limited; AUC shows trade-off between sensitivity and specificity.

✅ In Short
Insight	What It Means
Models perform similarly	Dataset is stable and moderately complex
Logistic, SVM, RF lead	Balanced accuracy, well-generalized
CV mean ≈ 0.77	Reliable estimate of real-world performance
Iterations = 9	Solver converged efficiently
Class 1 recall lower	Class imbalance — a common next fix
'''        
                    
                    # enhancements  
'''
Perfect 👌 — let’s go step by step and add ROC–AUC analysis to your supervised classifiers.

This will help you visualize and compare how well each model separates 
diabetic vs non-diabetic patients beyond just accuracy.

🧠 What You’ll Learn from ROC–AUC

ROC curve (Receiver Operating Characteristic):
Plots True Positive Rate (Recall) vs False Positive Rate (1 - Specificity) at all thresholds.
→ Measures how well the model distinguishes classes.

AUC (Area Under the Curve):
A single number between 0.5 (random guessing) and 1.0 (perfect separation).

So AUC ≈ 0.77 means: “the model correctly ranks a random diabetic higher than a random 
non-diabetic 77% of the time.”

🧩 Add This Code

You can safely append this to your existing script (after your current evaluation section).

from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# Create figure
plt.figure(figsize=(8, 6))

# Reuse your trained models
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # SVM (linear) doesn’t have predict_proba by default
        try:
            y_proba = model.decision_function(X_test)
        except:
            continue

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

# Add random baseline
plt.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.50)")

# Final plot formatting
plt.title("ROC Curves for Diabetes Classification Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

📊 Expected Output

You’ll see one combined chart:

Each line = one classifier’s ROC curve

Higher and leftward = better

Legend shows AUC for each model

You might get something like:

Model	AUC
Logistic Regression	0.79
Random Forest	0.80
SVM (linear)	0.78
KNN (k=5)	0.73
Decision Tree	0.70

This gives a deeper view than accuracy — it tells you which model discriminates 
best across thresholds, not just at one cutoff (0.5).

⚙️ Optional Enhancement

If you use SVC(kernel="linear", probability=True) when defining SVM, it will 
automatically support predict_proba().
That’ll make its ROC computation simpler and directly comparable.

Would you like me to extend this example to also plot Precision–Recall curves 
(which are often more insightful for imbalanced datasets like diabetes)?

You said:
yes
ChatGPT said:

Excellent 👍 — adding Precision–Recall (PR) curves will give you a more complete 
picture of your models’ performance — especially because diabetes datasets are 
slightly imbalanced (more “non-diabetic” than “diabetic” samples).

🧠 Why Precision–Recall Curves Matter

Precision (Positive Predictive Value) → of all predicted diabetics, how many are 
truly diabetic?

Recall (Sensitivity) → of all actual diabetics, how many did the model detect?

A high AUC-PR means the model catches diabetics without raising too many false alarms.
ROC can look deceptively good in imbalanced datasets — PR gives a truer view.

🧩 Code — Add This After the ROC Section
from sklearn.metrics import precision_recall_curve, average_precision_score

plt.figure(figsize=(8, 6))

for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        try:
            y_proba = model.decision_function(X_test)
        except:
            continue

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    plt.plot(recall, precision, label=f"{name} (AP = {ap:.2f})")

plt.title("Precision–Recall Curves for Diabetes Classification Models")
plt.xlabel("Recall (Sensitivity)")
plt.ylabel("Precision (Positive Predictive Value)")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()

📊 Expected Output

You’ll see one plot with several curves — one for each model.
The legend will show Average Precision (AP), analogous to “AUC” in the ROC chart.

Typical result (example):

Model	AUC (ROC)	AP (Precision–Recall)
Logistic Regression	0.79	0.65
Random Forest	0.80	0.67
SVM (linear)	0.78	0.63
KNN (k=5)	0.73	0.59
Decision Tree	0.70	0.54

The closer the Precision–Recall curve stays near the top-right, the better.
Random guessing produces a nearly flat line at the prevalence of diabetics (≈ 0.35 here).

⚙️ Optional — Combine Both

You can place both plots (ROC and PR) in one script, one after the other, 
to compare models visually and statistically.
'''          


                        