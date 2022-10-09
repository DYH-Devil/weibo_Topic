import joblib
import os

if(os.path.exists('../model/topic_words.dat')) :
    topic_words = joblib.load('../model/topic_words.dat')

print(topic_words)