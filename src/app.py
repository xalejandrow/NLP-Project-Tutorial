import pandas as pd
import pickle
import numpy as np
import re
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

#0.
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv')
#1. Transform
df = df_raw.copy()
df = df.drop_duplicates().reset_index(drop = True)
def url(text):
    return re.sub(r'(https://www|https://)', '', text)
df['url_limpia'] = df['url'].apply(url).apply(caracteres_no_alfanumericos).apply(esp_multiple)
df['is_spam'] = df['is_spam'].apply(lambda x: 1 if x == True else 0)
#2. Preproccesing and model MB
vec = CountVectorizer().fit_transform(df['url_limpia'])
#3. Split 
X_train, X_test, y_train, y_test = train_test_split(vec, df['is_spam'], stratify = df['is_spam'], random_state = 2207)
#4. Model
classifier = SVC(C = 1.0, kernel = 'linear', gamma = 'auto')
classifier.fit(X_train, y_train)
#5. Randomized search to select hyperparameters
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
grid = GridSearchCV(SVC(random_state=1234),param_grid,verbose=2)
grid.fit(X_train,y_train)
best_model = grid.best_estimator_
#6. Save best model
pickle.dump(best_model, open('../models/best_model.pickle', 'wb'))