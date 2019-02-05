# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pickle
import itertools

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import VotingClassifier
from scipy import stats
import json

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# funcion para expandir las matrices en una lista y tomar la parte triangular superior    
def unfold_data(data_list): 
    output = np.zeros((len(data_list), len(data_list[0][np.tril_indices(data_list[0].shape[0],1)])))
    for i,matrix in enumerate(data_list):
        output[i,:] = matrix[np.tril_indices(data_list[0].shape[0],1)]
    return output

def matrixPlot(matrix, name):
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(matrix, cmap=plt.cm.Blues)
    plt.colorbar(heatmap)
    plt.title(name + ' Correlation Matrix')
    plt.savefig('matrix_' + name + '.png') # guardar graficos en .png
    plt.clf() # clean buffer

def zScore(dato):
  for i in range(len(dato)):
    dato[i] = stats.zscore(dato[i], axis=1, ddof=1)
  return dato

def getClassifiers():
   n_estimators = 16   # cantidad de arboles
   max_depth = 4 # maxima profundidad de los arboles

   clf1 = LogisticRegression()
   clf2 = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
   clf3 = GaussianNB()
   return clf1, clf2, clf3

def fitModelwithCrossValidation(data, target, name):
   print name  

   clf1, clf2, clf3 = getClassifiers()

   n_folds = 5   # cantidad de folds
   cv = StratifiedKFold(target, n_folds=n_folds) # crear objeto de cross validation estratificada

   cv_target = np.array([])
   cv_prediction = np.array([])
   cv_probas = np.array([])
   cv_importances = np.zeros((n_folds, data.shape[1]))

   for i, (train, test) in enumerate(cv):
       X_train = data[train] # crear sets de entrenamiento y testeo para el fold
       X_test = data[test]
       y_train = target[train]
       y_test = target[test]

       clf1 = clf1.fit(X_train,y_train)
       clf2 = clf2.fit(X_train,y_train)
       clf3 = clf3.fit(X_train,y_train)
       clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
       clf = clf.fit(X_train,y_train)

       preds = clf.predict(X_test)
       probas = clf.predict_proba(X_test)

       cv_target = np.concatenate((cv_target, y_test), axis=0) # concatenar los resultados
       cv_prediction = np.concatenate((cv_prediction, preds), axis=0)
       cv_probas = np.concatenate((cv_probas, probas[:,1]), axis=0)


   cnf_matrix = confusion_matrix(cv_target, cv_prediction)
   class_names = [0,1,2,3,4]
   # Plot non-normalized confusion matrix
   plt.figure()
   plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

   # Plot normalized confusion matrix
   plt.figure()
   plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

   plt.show()

#   fpr, tpr, thresholds = roc_curve(cv_target,  cv_probas) # obtener la curva ROC
#   plt.plot(fpr,tpr,label=name) # plotear curva ROC

#   print "AUC Entrenamiento del Voting Classifier usando Cross-Validation:", auc(fpr, tpr)
#   return auc(fpr,tpr)

# cargar matrices de correlacion

def somnolencia_class(somnolencia_z):
  if somnolencia_z <= -1:
    return 0
  if -1 < somnolencia_z <= -0.5:
    return 1
  if -0.5 < somnolencia_z <= 0:
    return 2
  if 0 < somnolencia_z <= 1:
    return 3
  if 1 < somnolencia_z:
    return 4


import os
import csv
import numpy
import sys

rootPath = '../../informe/IMGs/Open/'
somnolencia_z = json.load(open(rootPath + 'somnolencia_z.json'))
somnolencia_z = {k:float(v.replace(',','.')) for k,v in somnolencia_z.items()}

metric = sys.argv[1]
resting_data = {'beta':[], 'theta':[],'alpha':[]}
target = []

for band in ['beta','theta','alpha']:
  rootPath = '../../informe/IMGs/Open/ninios_'+band+'/'+metric+'/'
  for file in os.listdir(rootPath):
    if file[-4:] == '.csv' and file[:5].upper() in somnolencia_z:
      reader = csv.reader(open(rootPath + file, "rb"), delimiter=",")
      x = list(reader)
      x = numpy.array(x).astype("float")

      resting_data[band].append(x)
      target.append(somnolencia_class(somnolencia_z[file[:5].upper()]))

target = np.array(target)

resting_data = {k:np.array(v) for k,v in resting_data.items()}

res = []

print target

data = np.concatenate((unfold_data(resting_data['beta']),unfold_data(resting_data['theta']), unfold_data(resting_data['alpha'])), axis=1)
res.append(metric+"\t"+'allBands' +"\t"+str(fitModelwithCrossValidation(data, target[data.shape[0]:data.shape[0]*2], 'allBands')))

#data = unfold_data(resting_data['alpha'])
#res.append(metric+"\t"+'alpha' +"\t"+str(fitModelwithCrossValidation(data, target[data.shape[0]*2:], 'alpha')))

#data = unfold_data(resting_data['theta'])
#res.append(metric+"\t"+'theta' +"\t"+str(fitModelwithCrossValidation(data, target[data.shape[0]:data.shape[0]*2], 'theta')))

#data = unfold_data(resting_data['beta'])
#res.append(metric+"\t"+'beta' +"\t"+str(fitModelwithCrossValidation(data, target[:data.shape[0]], 'beta')))

print res