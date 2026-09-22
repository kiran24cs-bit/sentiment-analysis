
import sklearn
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
import joblib
model = joblib.load("model.pkl")
vec=joblib.load("vec.pkl")
en=joblib.load("en.pkl")
from nltk.corpus import stopwords
nltk.download("stopwords")
from nltk.stem import PorterStemmer
stem=PorterStemmer()
words=stopwords.words("english")
def fun(test):
  new=[]
  for word in test:
    new.append(stem.stem(word))
  return " ".join(new)
test=input("enter your review (quit to exit)")
while(test.lower()!="quit"):
  test=test.lower()
  test=test.split(" ")
  test=[word for word in test if word not in words]
  test=fun(test)
  # test="accept"
  test=vec.transform([test])
  # test=" ".join(test)
  pre=model.predict(test)
  print(pre[0])
  print("The review is ",(en.inverse_transform([pre[0]])[0]))
  test=input("enter your review (quit to exit)")
