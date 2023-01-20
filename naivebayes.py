import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_auc_score,f1_score,precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

csvPath = r"C:\Users\YSK\Desktop\1030520813_patternrecognition_final_project\breast-cancer.csv"
dataset = pd.read_csv(csvPath)
dataset = dataset.drop(['id'],axis=1)
le = LabelEncoder()
dataset['diagnosis'] = le.fit_transform(dataset['diagnosis'])

X = dataset.drop(columns=['diagnosis'])
sc = StandardScaler()
X=sc.fit_transform(X)

y = dataset['diagnosis']
le=LabelEncoder()
y=le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cMatrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cMatrix)
classificationReport = classification_report(y_test, y_pred)
print("Classification Report:\n", classificationReport)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy, "\n")
print('Sensitivity : ', cMatrix[0, 0]/(cMatrix[0, 0]+cMatrix[0, 1]))
print('Specificity : ', cMatrix[1, 1]/(cMatrix[1, 0]+cMatrix[1, 1]))
print('ROC :',roc_auc_score(y_test, y_pred))
print('f1-score:', f1_score(y_test, y_pred))
print('Precision: %.3f' % precision_score(y_test, y_pred))