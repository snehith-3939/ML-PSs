import nltk
from nltk.corpus import movie_reviews
import pandas as pd
import matplotlib.pyplot as plt
import random   
import numpy as np 
from sklearn.linear_model import LogisticRegression  # For training the model
from sklearn.metrics import accuracy_score  # For checking accuracy 

# Download the dataset for accessing it
nltk.download('movie_reviews') 

# Positive reviews
all_positive_reviews = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids('pos')]

# Negative reviews
all_negative_reviews = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids('neg')]

print("Number of Positive Reviews:", len(all_positive_reviews))
print("Number of Negative Reviews:", len(all_negative_reviews))
print('\n')
print("Type of positive reviews : ",type(all_positive_reviews))
print('The type of each positive review : ', type(all_positive_reviews[0]))


# Representation of reviews using Matplotlib

#Specifing dimensions(size) of figure
fig = plt.figure(figsize=(6,6))

#Assigning labels and sizes for the figure
labels = 'Positive Reviews', 'Negative Reviews'
sizes = [len(all_positive_reviews), len(all_negative_reviews)]

#Plotting a pie chart with following specifications
plt.pie(sizes , labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)

#Equal aspect ratio to ensure pie chart is a circle
plt.axis('equal')
#Display the pie chart
plt.show()

# Lets start the Preprocessing of the data

# Here we have to do the following tasks :
# Tokenizing the strings
# Lowercasing
# Removing stop words and punctuation
# Stemming
# Removing handles and urls

# We here take help of pre built function for preprocessing

from utils import process_tweet # Import the process_tweet function

preprocessed_positive_reviews = [ process_tweet(review) for review in all_positive_reviews ]
preprocessed_negative_reviews =[ process_tweet(review) for review in all_negative_reviews ]

print(preprocessed_positive_reviews[0])
print(preprocessed_negative_reviews[0])

# Building word frequencies

from utils import  build_freqs

#concatenate both the lists
reviews = all_positive_reviews + all_negative_reviews

print('Number of reviews : ', len(reviews))

# make a numpy array representing labels of the reviews

positive_labels = np.ones(len(all_positive_reviews))
negative_labels = np.zeros(len(all_negative_reviews))

Labels = np.concatenate((positive_labels,negative_labels))

#create frequency dictionary
freqs = build_freqs(reviews, Labels)

# check data type
print(f'type(freqs) = {type(freqs)}')

# check length of the dictionary
print(f'len(freqs) = {len(freqs)}')

print(freqs)

#Training dataset (split the data for training and testing purposes)
train_pos = all_positive_reviews[:800]
train_neg = all_negative_reviews[:800]

train_dataset = train_pos+train_neg

print('Number of reviews for training : ', len(train_dataset))

#

def process_review(review, freqs):
    words = process_tweet(review)  # Tokenize the review (simple split by space)
    
    # Initialize the sums for positive and negative frequencies
    positive_sum = 0
    negative_sum = 0
    
    # Loop through each word in the review
    for word in words:
        
          positive_sum += freqs.get((word, 1.0), 0)  # Sum positive frequency
          negative_sum += freqs.get((word, 0.0), 0)  # Sum negative frequency

    # Set the bias (1 for all reviews)
    bias = 1
    # Determine sentiment (this can be based on total positive or negative sum)
    sentiment = 1 if positive_sum > negative_sum else 0  # 1 for positive, 0 for negative
    
    return positive_sum, negative_sum, bias, sentiment


# Create a list of results for each review
data = []
for review in train_dataset:
    positive_sum, negative_sum, bias, sentiment = process_review(review, freqs)
    data.append([positive_sum, negative_sum, bias, sentiment])


# Training a Dataset

# Create a DataFrame from the results
df = pd.DataFrame(data, columns=["Positive_Frequency_Sum", "Negative_Frequency_Sum", "Bias", "Sentiment"])

# Print the DataFrame
print(df.head())  # Display the first 5 rows of the DataFrame

# Features: Positive_Frequency_Sum, Negative_Frequency_Sum, Bias
X = df[["Positive_Frequency_Sum", "Negative_Frequency_Sum", "Bias"]].values

# Labels: Sentiment
y = df["Sentiment"].values

# Check the shapes of X and y
print("Features Shape:", X.shape)
print("Labels Shape:", y.shape)

# Create the Logistic Regression model
model = LogisticRegression()

# Train the model on the entire dataset
model.fit(X, y)

print("Model training on the entire dataset is complete!")

#Creating Dataset for testing

test_pos = all_positive_reviews[800:1000]
test_neg = all_negative_reviews[800:1000]

test_dataset = test_pos+test_neg

print('Number of reviews for testing : ', len(test_dataset))


# Create a list of results for each review

data_test = []
for review in test_dataset:
    positive_sum, negative_sum, bias, sentiment = process_review(review, freqs)
    data_test.append([positive_sum, negative_sum, bias, sentiment])


# Training a Dataset

# Create a DataFrame from the results
df = pd.DataFrame(data, columns=["Positive_Frequency_Sum", "Negative_Frequency_Sum", "Bias", "Sentiment"])

# Print the DataFrame
print(df.head())  # Display the first 5 rows of the DataFrame

# Create a DataFrame for the testing dataset
df_test = pd.DataFrame(data_test, columns=["Positive_Frequency_Sum", "Negative_Frequency_Sum", "Bias", "Sentiment"])

# Features: Positive_Frequency_Sum, Negative_Frequency_Sum, Bias
X_test = df_test[["Positive_Frequency_Sum", "Negative_Frequency_Sum", "Bias"]].values

# Labels: Sentiment
y_test = df_test["Sentiment"].values

# Check the shapes of X_test and y_test
print("Testing Features Shape:", X_test.shape)
print("Testing Labels Shape:", y_test.shape)

# Predict the sentiment for the testing dataset
y_pred = model.predict(X_test)

# Print the first 10 predictions and the actual labels
print("Predictions:", y_pred[:10])
print("Actual Labels:", y_test[:10])

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Function to predict sentiment for a user-provided review
def predict_sentiment(review, freqs, model):
    # Preprocess the review using process_tweet
    words = process_tweet(review)  # Tokenize, lowercase, remove stop words, etc.
    
    # Initialize the sums for positive and negative frequencies
    positive_sum = 0
    negative_sum = 0
    
    # Calculate frequency sums for the input review
    for word in words:
        positive_sum += freqs.get((word, 1.0), 0)  # Sum positive frequency
        negative_sum += freqs.get((word, 0.0), 0)  # Sum negative frequency

    # Bias term
    bias = 1
    
    # Create feature array
    features = np.array([[positive_sum, negative_sum, bias]])
    
    # Make prediction
    prediction = model.predict(features)
    
    # Convert prediction to sentiment label
    sentiment = "Positive (Good)" if prediction[0] == 1 else "Negative (Bad)"
    
    return sentiment

# Allow user to input a review for prediction
while True:
    user_review = input("Enter a movie review (or type 'exit' to quit): ")
    if user_review.lower() == 'exit':
        print("Exiting sentiment predictor. Goodbye!")
        break
    
    # Predict the sentiment of the input review
    result = predict_sentiment(user_review, freqs, model)
    print(f"The sentiment of the review is: {result}")
