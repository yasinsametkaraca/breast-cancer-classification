import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score,precision_score,roc_curve,auc

csvPath = r"C:\Users\YSK\Desktop\1030520813_patternrecognition_final_project\breast-cancer.csv"
dataset = pd.read_csv(csvPath)
sc = StandardScaler()
dataset = dataset.drop(['id'],axis=1)
le = LabelEncoder()
dataset['diagnosis'] = le.fit_transform(dataset['diagnosis'])
X = dataset.drop(columns=['diagnosis'])
y = dataset['diagnosis']
X=sc.fit_transform(X)
le=LabelEncoder()
y=le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
k_list = list(range(1,12))
cv_scores = []
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores)
    print(cv_scores)
# K = 2 best score
classifier = KNeighborsClassifier(n_neighbors=2, metric='euclidean', p = 2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
cMatrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cMatrix) 
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print('Sensitivity : ', cMatrix[0, 0]/(cMatrix[0, 0]+cMatrix[0, 1]))
print('Specificity : ', cMatrix[1, 1]/(cMatrix[1, 0]+cMatrix[1, 1]))
print('f1-score:', f1_score(y_test, y_pred))
print('Precision: %.3f' % precision_score(y_test, y_pred))
print("AUC : ", auc(fpr, tpr))
