import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

#Load dataset
data = pd.read_csv("spam_detection_dataset.csv")

#Separate X and y
X = data.drop('is_spam', axis=1)
y = data['is_spam']

#Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Train Logistic Regression Model
model = LogisticRegression(class_weight= {0:1, 1:3})
model.fit(X_train, y_train)

#Prediction
pred = model.predict(X_test)

#Evaluation
print("Accuracy: ", accuracy_score(y_test, pred))
print("Confusion Matrix: ", confusion_matrix(y_test, pred))
print("Classification report :", classification_report(y_test, pred))

#Save Model

joblib.dump(model, 'spamDetection.pkl')
