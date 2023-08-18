import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_auc_score,f1_score,precision_score

csvPath = r"C:\Users\YSK\Desktop\1030520813_patternrecognition_final_project\breast-cancer.csv"
dataset = pd.read_csv(csvPath)
sc = StandardScaler()

dataset = dataset.drop(['id'], axis=1)
le = LabelEncoder()
dataset['diagnosis'] = le.fit_transform(dataset['diagnosis'])

X = dataset.drop(columns=['diagnosis'])
y = dataset['diagnosis']

X = sc.fit_transform(X)
le = LabelEncoder()
y = le.fit_transform(y)

classifier = DecisionTreeClassifier()
kfold=KFold(n_splits=5)
accs =list()
for fold_index,(train_idx,test_idx) in enumerate(kfold.split(X,y)):
    print("Fold: ",fold_index+1)
    X_train,X_test = X[train_idx],X[test_idx]
    y_train,y_test = y[train_idx],y[test_idx]
    
    classifier.fit(X_train,y_train)
    acc = classifier.score(X_test,y_test)
    print("Accuracy in Fold: ",acc)
    accs.append(acc)
    
print("Mean Accuracy: ", sum(accs)/len(accs))
y_pred = classifier.predict(X_test)
cMatrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cMatrix)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print('Sensitivity : ', cMatrix[0, 0]/(cMatrix[0, 0]+cMatrix[0, 1]))
print('Specificity : ', cMatrix[1, 1]/(cMatrix[1, 0]+cMatrix[1, 1]))
print('AUC :',roc_auc_score(y_test, y_pred))
print('f1-score:', f1_score(y_test, y_pred))
print('Precision: %.3f' % precision_score(y_test, y_pred))

