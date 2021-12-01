import joblib
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import faiss
list = ['Ancient','British','Indo-Islamic','Maratha','Not a Monument', 'Sikh']
with open('images.npy', 'rb') as f:
        X= np.load(f)
with open('labels.npy', 'rb') as g:     
        Y= np.load(g)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)


clf = neighbors.KNeighborsClassifier(10)

clf.fit(X_train, Y_train)
filename= "C:\\Users\\anany\\OneDrive\\Documents\\IIITD\\Git Project\\KNN.sav"
joblib.dump(clf, filename)
print('Done')
