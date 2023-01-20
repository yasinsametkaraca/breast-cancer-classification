import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_auc_score,f1_score,precision_score,roc_curve,auc
from sklearn.ensemble import GradientBoostingClassifier

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
clf_feature_selection = GradientBoostingClassifier(n_estimators=50) 
rfecv = RFECV(estimator=clf_feature_selection, 
              step=1, 
              cv=5, 
              scoring = 'roc_auc',
             )        
clf = GradientBoostingClassifier(n_estimators=50) 
CV_gbc = GridSearchCV(clf, 
                      param_grid={'n_estimators':[20,50,100,200], 'learning_rate': [0,10,0.25,0.5,0.75]},
                      cv= 5, scoring = 'roc_auc')

pipeline  = Pipeline([('feature_sele',rfecv),
                      ('clf_cv',CV_gbc)])
pipeline.fit(X_train, y_train)
y_pred=pipeline.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
cMatrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cMatrix)
classificationReport = classification_report(y_test, y_pred)
print("Classification Report:\n", classificationReport)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy, "\n")
print('Sensitivity : ', cMatrix[0, 0]/(cMatrix[0, 0]+cMatrix[0, 1]))
print('Specificity : ', cMatrix[1, 1]/(cMatrix[1, 0]+cMatrix[1, 1]))
print('f1-score:', f1_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print("AUC : ", auc(fpr, tpr))



