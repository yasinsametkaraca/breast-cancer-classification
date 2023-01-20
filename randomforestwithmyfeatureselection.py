import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_auc_score,f1_score,precision_score
from sklearn.preprocessing import LabelEncoder
csvPath = r"C:\Users\YSK\Desktop\1030520813_patternrecognition_final_project\breast-cancer.csv"
dataset = pd.read_csv(csvPath)
dataset = dataset.drop(['id'],axis=1)
le = LabelEncoder()
dataset['diagnosis'] = le.fit_transform(dataset['diagnosis'])
X = dataset.drop(columns=['diagnosis'])
y = dataset['diagnosis']
classifier=RandomForestClassifier()         # I Create a random forest classifier
skf = StratifiedKFold(n_splits=5)       # Use Stratified K-Fold cross-validation to evaluate the attribute selection
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    classifier.fit(X_train, y_train)              # Fit the classifier to the training data and get the feature importances
    feature_importances = classifier.feature_importances_
    N = 10         # Select the top N most important features
    top_features = X_train.columns[feature_importances.argsort()[::-1][:N]]          # Select the top N most important features from the test set
    X_test = X_test[top_features]
X_train = X[top_features]         # Select the top N most important features from the entire dataset
classifier=RandomForestClassifier() 
model = classifier.fit(X_train, y)           # Fit the classifier to the entire dataset using the selected features
y_pred = model.predict(X_test)
cMatrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cMatrix)
print("Classification Report:\n", classification_report(y_test, y_pred))






