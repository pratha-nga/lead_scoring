#imports

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.mixture import GMM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, coo_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle

#csv Load

accountDesc = pd.read_csv("/salesforceData/self_made/accountDescription.csv",
                          usecols = ["Phone","Industry","AnnualRevenue","NumberOfEmployees","Qualified","name","description"])
#Feature Scaling

accountDesc["Phone"] = accountDesc["Phone"].fillna(0)
accountDesc["Phone"] = accountDesc["Phone"].map(lambda x : 1 if x!= 0 else x)

accountDesc["Industry"] = accountDesc["Industry"].fillna("")
accountDesc["description"] = accountDesc["description"].fillna("")

accountDesc["NumberOfEmployees"] = accountDesc["NumberOfEmployees"].fillna(0)

accountDesc['Description'] = accountDesc[['Industry', 'description']].apply(lambda x: ' '.join(x.astype(str)), axis=1)

tfidf = TfidfVectorizer(stop_words= 'english')
tfidf_mat = tfidf.fit(accountDesc['Description'])
tfidf_matrix = tfidf_mat.transform(accountDesc['Description'])
tfidf_matrix = tfidf_matrix.toarray()
yy = tfidf_matrix[2]
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
features = accountDesc.drop(columns = ["Industry","description","AnnualRevenue","name","Qualified","Description"])

features = features.values
stack = hstack([coo_matrix(features),coo_matrix(cosine_sim)]).toarray()


# train test split

X_train, X_test, Y_train, Y_test = train_test_split(stack,accountDesc["Qualified"],shuffle = True,test_size=0.3)

#Gaussian mixture model

gmm = GMM(n_components=2).fit(X_train)
# save the model to disk

filename = '/Users/standarduser/Documents/salesforceData/self_made/leadScoring/model/finalized_model1.sav'
pickle.dump(gmm, open(filename, 'wb'))
#y_labels = gmm.predict(X_train)
loaded_model = pickle.load(open(filename, 'rb'))
y_labels_2 = gmm.predict(X_train)

#Predict
probs = gmm.predict_proba(X_train)
prob = np.exp(probs )

#Scores
score_sample,s = gmm.score_samples(X_train[0:1])
score = gmm.score(X_train)

matrix = confusion_matrix(y_labels_2, Y_train)

## demo
tfid_x = 1
tfid_y = 22066

demo = [1,500,"Apparel & Fashion Saks Fifth Avenue was the vision of Horace Saks and Bernard Gimbel."]
tfid_matrix_demo = tfidf.fit_transform([demo[2]])
cosine_2 = cosine_similarity(tfid_matrix_demo,tfidf_matrix )

