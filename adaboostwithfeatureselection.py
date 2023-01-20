from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_auc_score,f1_score,precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

csvPath = r"C:\Users\YSK\Desktop\1030520813_patternrecognition_final_project\breast-cancer-attributes.csv"
X = np.loadtxt(csvPath, delimiter=",")
csvPath = r"C:\Users\YSK\Desktop\1030520813_patternrecognition_final_project\breast-cancer.csv"
dataset = pd.read_csv(csvPath)
y = dataset['diagnosis']
le=LabelEncoder()
y=le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
parameters = {'n_estimators': [10, 50, 100, 200],  # Set up the parameters for the grid search
              'learning_rate': [0.1, 0.5, 1.0, 2.0]}

adaboost = AdaBoostClassifier() # Create the AdaBoost classifier
grid_search = GridSearchCV(adaboost, parameters, cv=5) # Create the grid search object
grid_search.fit(X_train, y_train)  # Fit the grid search object to the training data
print("Best parameters:", grid_search.best_params_) # Print the best parameters

adaboost = AdaBoostClassifier(**grid_search.best_params_) # Use the best parameters to create the final AdaBoost classifier
selector = RFECV(estimator=adaboost,cv=5,step=1)
selector = selector.fit(X_train, y_train)
print(selector.support_)
X_train_selected = X_train[:, selector.support_]
adaboost.fit(X_train_selected, y_train)

X_test = X_test[:, selector.support_]
y_pred = adaboost.predict(X_test)
cMatrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cMatrix)
classificationReport = classification_report(y_test, y_pred)
print("Classification Report:\n", classificationReport)
print("Accuracy:", accuracy_score(y_test, y_pred))
print('Sensitivity : ', cMatrix[0, 0]/(cMatrix[0, 0]+cMatrix[0, 1]))
print('Specificity : ', cMatrix[1, 1]/(cMatrix[1, 0]+cMatrix[1, 1]))
print('AUC :',roc_auc_score(y_test, y_pred))
print('f1-score:', f1_score(y_test, y_pred))
print('Precision:',precision_score(y_test, y_pred))

