import joblib
import warnings
warnings.filterwarnings('ignore')

#Load Model
model = joblib.load('D:\Programs\Python\Projects\spamDetection.pkl')

#User Input
num_link = int(input("Enter number of links: "))
num_words = int(input("Enter number of words: "))
has_offer = int(input("Enter has offer (0 or 1): "))
sender_score = float(input("Enter sender score: "))
all_caps = int(input("Enter all caps (0 or 1): "))

new_email = [num_link, num_words, has_offer, sender_score, all_caps]

scaler = joblib.load('D:\Programs\Python\Projects\spamDetectionScaler.pkl')
new_email_transform = scaler.transform([new_email])

#Prediction
prediction = model.predict(new_email_transform)

if prediction[0] == 0:
    print("Email is not spam")
elif prediction[0] == 1:
    print("Email is spam")
