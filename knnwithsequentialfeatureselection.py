from mlxtend.feature_selection import SequentialFeatureSelector
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_auc_score,f1_score,precision_score,roc_curve,auc

csvPath = r"C:\Users\YSK\Desktop\1030520813_patternrecognition_final_project\breast-cancer-attributes.csv"
X = np.loadtxt(csvPath, delimiter=",")
csvPath = r"C:\Users\YSK\Desktop\1030520813_patternrecognition_final_project\breast-cancer.csv"
dataset = pd.read_csv(csvPath)
y = dataset['diagnosis']
le=LabelEncoder()
y=le.fit_transform(y)

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.3, random_state=42)
model = KNeighborsClassifier(n_neighbors=2)
model.fit(X_train, y_train)
print('Training accuracy:', np.mean(model.predict(X_train) == y_train)*100)
print('Test accuracy:', np.mean(model.predict(X_test) == y_test)*100)

sfs = SequentialFeatureSelector(KNeighborsClassifier(n_neighbors=3), 
          k_features=15, 
          forward=True, 
          floating=False, 
          verbose=2,
          scoring='accuracy',
          cv=5)
sfs = sfs.fit(X_train, y_train)    

print("Selected features:", sfs.k_feature_idx_)
X_train=X_train[:, sfs.k_feature_idx_]
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test[:, sfs.k_feature_idx_])        
fpr, tpr, thresholds = roc_curve(y_test, predictions)
cMatrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n", cMatrix) 
print("Classification Report:\n", classification_report(y_test, predictions))
print("Accuracy:", accuracy_score(y_test, predictions))
print('Sensitivity : ', cMatrix[0, 0]/(cMatrix[0, 0]+cMatrix[0, 1]))
print('Specificity : ', cMatrix[1, 1]/(cMatrix[1, 0]+cMatrix[1, 1]))
print('f1-score:', f1_score(y_test, predictions))
print('Precision: %.3f' % precision_score(y_test, predictions))
print("AUC : ", auc(fpr, tpr))