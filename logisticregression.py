from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_auc_score,f1_score,precision_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

csvPath = r"C:\Users\YSK\Desktop\1030520813_patternrecognition_final_project\breast-cancer-attributes.csv"
X = np.loadtxt(csvPath, delimiter=",")
csvPath = r"C:\Users\YSK\Desktop\1030520813_patternrecognition_final_project\breast-cancer.csv"
dataset = pd.read_csv(csvPath)
y = dataset['diagnosis']
le=LabelEncoder()
y=le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
n_folds = 5
n_features = 10
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
scores = []
f1_scores = []
for train_index, val_index in kf.split(X_train): 
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    X_train_fold_selected = X_train_fold[:, :n_features]
    X_val_fold_selected = X_val_fold[:, :n_features]
    clf = LogisticRegression()
    clf.fit(X_train_fold_selected, y_train_fold)
    score = clf.score(X_val_fold_selected, y_val_fold)
    scores.append(score)
    y_pred = clf.predict(X_val_fold_selected)
    f1 = f1_score(y_val_fold, y_pred)
    f1_scores.append(f1)

print("Cross validation score:", sum(scores) / n_folds)
print("F1 score:", sum(f1_scores) / n_folds)
X_train = X_train[:, :n_features]
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test[:, :n_features])
cMatrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cMatrix)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print('Sensitivity : ', cMatrix[0, 0]/(cMatrix[0, 0]+cMatrix[0, 1]))
print('Specificity : ', cMatrix[1, 1]/(cMatrix[1, 0]+cMatrix[1, 1]))
print('AUC :',roc_auc_score(y_test, y_pred))
print('f1-score:', f1_score(y_test, y_pred))
print('Precision: %.3f' % precision_score(y_test, y_pred))