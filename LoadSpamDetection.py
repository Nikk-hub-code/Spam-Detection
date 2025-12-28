from sklearn.preprocessing import StandardScaler
import joblib

#Load Model
model = joblib.load('spamDetection.pkl')

#User Input
num_link = int(input("Enter number of links: "))
num_words = int(input("Enter number of words: "))
has_offer = int(input("Enter has offer (0 or 1): "))
sender_score = float(input("Enter sender score: "))
all_caps = int(input("Enter all caps (0 or 1): "))

new_email = [num_link, num_words, has_offer, sender_score, all_caps]

scaler = StandardScaler()
new_email_transform = scaler.fit_transform([new_email])

#Prediction
prediction = model.predict(new_email_transform)

if prediction == 0:
    print("Email is not spam")
elif prediction == 1:

    print("Email is spam")
