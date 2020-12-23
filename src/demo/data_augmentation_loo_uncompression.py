import pickle
import sys
import numpy as np
from skeleton_utils import nameangles
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

with open(sys.argv[1], 'rb') as f:
    dumpdict = pickle.load(f)
X_train = list()
Y_train = list()
X_test = list()
Y_test = list()

i = 0

for trainsample in dumpdict['train']:
    X_train.append(trainsample[0])
    Y_train.append(trainsample[1])

for testsample in dumpdict['test']:
    X_test.append(testsample[0])
    Y_test.append(testsample[1])

print(X_train[0])
clf = LinearSVC(random_state=0, tol=1e-5)

clf = clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
print(classification_report(Y_test, Y_pred), end="\n")
print(confusion_matrix(Y_test, Y_pred), end="\n")
print(clf.score(X_test, Y_test), end="\n")
