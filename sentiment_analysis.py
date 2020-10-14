import nltk
import numpy as np

from sklearn.utils import shuffle
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup

wordnet_lemmatiser = WordNetLemmatizer()
stopwords = set(w.rstrip() for w in open('stopwords.txt'))

electronics_positive_reviews = BeautifulSoup(open('electronics/positive.review').read(), features='html5lib')
electronics_positive_reviews = electronics_positive_reviews.findAll('review_text')

electronics_negative_reviews = BeautifulSoup(open('electronics/negative.review').read(), features='html5lib')
electronics_negative_reviews = electronics_negative_reviews.findAll('review_text')

def my_tokeniser(string):
    string = string.lower()
    tokens = nltk.tokenize.word_tokenize(string)
    tokens = [each_token for each_token in tokens if len(each_token) > 2]
    tokens = [wordnet_lemmatiser.lemmatize(each_token) for each_token in tokens]
    tokens = [each_token for each_token in tokens if each_token not in stopwords]
    return tokens

word_index_map = {}
current_index = 0
positive_tokenised = []
negative_tokenised = []
original_reviews = []

for each_review in electronics_positive_reviews:
    original_reviews.append(each_review.text)
    tokens = my_tokeniser(each_review.text)
    positive_tokenised.append(tokens)

    for each_token in tokens:
        if each_token not in word_index_map:
            word_index_map[each_token] = current_index
            current_index += 1

for each_review in electronics_negative_reviews:
    original_reviews.append(each_review.text)
    tokens = my_tokeniser(each_review.text)
    negative_tokenised.append(tokens)

    for each_token in tokens:
        if each_token not in word_index_map:
            word_index_map[each_token] = current_index
            current_index += 1

print(f"len(word_index_map): {len(word_index_map)}")

def tokens_to_vector(tokens, label):
    x = np.zeros(len(word_index_map) + 1)

    for each_token in tokens:
        index = word_index_map[each_token]
        x[index] += 1
    
    x = x / x.sum()
    x[-1] = label
    return x

token_count = len(positive_tokenised) + len(negative_tokenised)
data = np.zeros((token_count, len(word_index_map) + 1))
index = 0

for each_token in positive_tokenised:
    data[index,:] = tokens_to_vector(tokens, 1)
    index += 1

for each_token in negative_tokenised:
    data[index,:] = tokens_to_vector(tokens, 0)
    index += 1

original_reviews, data = shuffle(original_reviews, data)

input_data = data[:,:-1]
target_data = data[:,-1]

train_input = input_data[:-100,]
train_target = target_data[:-100,]

test_input = input_data[-100:,]
test_target = target_data[-100:,]

model = LogisticRegression()
model.fit(train_input, train_target)
print(f"Train accuracy: {model.score(train_input, train_target)}")
print(f"Test accuracy: {model.score(test_input, test_target)}")

threshold = 0.5

for each_word, each_index in word_index_map.items():
    weight = model.coef_[0][each_index]

    if weight > threshold or weight < -threshold:
        print(each_word, weight)

predictions = model.predict(input_data)
predict_probability = model.predict_proba(input_data)[:,1]

minimum_probability_if_review_is_positive = 1
maximum_probability_if_review_is_negative = 0
wrong_positive_review = None
wrong_negative_review = None
wrong_positive_prediction = None
wrong_negative_prediction = None

for each_index in range (token_count):
    probability = predict_probability[each_index]
    target = target_data[each_index]

    if target == 1 and probability < 0.5:
        if probability < minimum_probability_if_review_is_positive:
            wrong_positive_review = original_reviews[each_index]
            wrong_positive_prediction = predictions[each_index]
            minimum_probability_if_review_is_positive = probability

    elif target == 0 and probability > 0.5:
        if probability > maximum_probability_if_review_is_negative:
            wrong_negative_review = original_reviews[each_index]
            wrong_negative_prediction = predictions[each_index]
            maximum_probability_if_review_is_negative = probability

print(f"Most wrong positive review: probability = {minimum_probability_if_review_is_positive}, prediction = {wrong_positive_prediction}")
print(wrong_positive_review)
print(f"Most wrong negative review: probability = {maximum_probability_if_review_is_negative}, prediction = {wrong_negative_prediction}")
print(wrong_negative_review)