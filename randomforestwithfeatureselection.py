import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_auc_score,f1_score,precision_score

csvPath = r"C:\Users\YSK\Desktop\1030520813_patternrecognition_final_project\breast-cancer.csv"
dataset = pd.read_csv(csvPath)
dataset = dataset.drop(['id'],axis=1)
le = LabelEncoder()
dataset['diagnosis'] = le.fit_transform(dataset['diagnosis'])

X = dataset.drop(columns=['diagnosis'])
y = dataset['diagnosis']
le=LabelEncoder()
y=le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)

clf_feature_selection = RandomForestClassifier(random_state=42) 
rfecv = RFECV(estimator=clf_feature_selection, 
              step=1, 
              cv=5, 
              scoring = 'roc_auc',
             )         

clf = RandomForestClassifier(random_state=42) 
n_estimators = [50,100,150,200,250]
max_features = ['auto', 'sqrt', 'log2']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

params_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}        

CV_rfc = rfc_cv = RandomizedSearchCV(estimator = clf,  param_distributions = params_grid, verbose = 0, cv = 5, n_iter = 100)

pipeline  = Pipeline([('feature_sele',rfecv),
                      ('clf_cv',CV_rfc)])

pipeline.fit(X_train, y_train)
y_pred=pipeline.predict(X_test)

cMatrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cMatrix)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print('Sensitivity : ', cMatrix[0, 0]/(cMatrix[0, 0]+cMatrix[0, 1]))
print('Specificity : ', cMatrix[1, 1]/(cMatrix[1, 0]+cMatrix[1, 1]))
print('ROC :',roc_auc_score(y_test, y_pred))
print('f1-score:', f1_score(y_test, y_pred))
print('Precision:',precision_score(y_test, y_pred))



