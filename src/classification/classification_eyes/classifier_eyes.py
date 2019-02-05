# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pickle
import itertools

from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import VotingClassifier
from scipy import stats
import json

# funcion para elegir el umbral que resulte en valores balanceados en la diagonal    
def get_optimal_thr_diagonal_cm(probs, target, step): 
    difference = np.zeros((len(np.arange(0,1,step))))
    n=-1
    for thr in np.arange(0,1,step):
        preds_thresholded = np.zeros(len(probs))
        n=n+1
        preds_thresholded[np.where(probs>thr)[0]] = 1
        cm = confusion_matrix(target, preds_thresholded).astype(float)
        cm[0,:] = cm[0,:]/float(sum(cm[0,:]))
        cm[1,:] = cm[1,:]/float(sum(cm[1,:]))
        difference[n] = abs(cm[0,0] - cm[1,1])
    loc = np.where( difference==min(difference))[0]
    return np.arange(0,1,step)[loc][0]

def getTrainDataTarget(set1,set2):

   # seleccionar fases de suenio que se van a comparar

   target1 = np.zeros(set1.shape[0])
   target2 = np.ones(set2.shape[0])


   data = np.concatenate((set1,set2), axis=0)
   target = np.concatenate((target1, target2), axis=0)

   print target

   return data, target

def maxValues(values):
  electrodes = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

  res = []
  count = 0
  for i in range(14):
    for j in range(i):
      res.append((i,j,values[count]))
      count += 1

  return map(lambda x: (electrodes[x[0]],electrodes[x[1]],x[-1]),sorted(res,key=lambda x:-x[-1]))

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

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('eyes_' + title.lower().replace(" ","_") + ".png")

# funcion para expandir las matrices en una lista y tomar la parte triangular superior    
def unfold_data(data_list): 
    output = np.zeros((len(data_list), len(data_list[0][np.tril_indices(data_list[0].shape[0],-1)])))
    for i,matrix in enumerate(data_list):
        output[i,:] = matrix[np.tril_indices(data_list[0].shape[0],-1)]
    return output

def matrixPlot(matrix, name):
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(matrix, cmap=plt.cm.Blues)
    plt.colorbar(heatmap)
    plt.title(name + ' Correlation Matrix')
    plt.savefig('matrix_' + name + '.png') # guardar graficos en .png
    plt.clf() # clean buffer

def getClassifiers():
   n_estimators = 16   # cantidad de arboles
   max_depth = 4 # maxima profundidad de los arboles

   clf1 = LogisticRegression()
   clf2 = BernoulliNB()
   clf3 = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
#   clf2 = GaussianNB()
#   clf3 = GaussianNB()
   return clf1, clf2, clf3

def fitModelwithCrossValidation(data, target, name,cvs):
   print name  

   clf1, clf2, clf3 = getClassifiers()

   cv_target = np.array([])
   cv_prediction = np.array([])
   cv_probas = np.array([])

   importantes = []
   for cv in cvs:
     for train, test in cv.split(data, target):

         X_train = data[train] # crear sets de entrenamiento y testeo para el fold
         X_test = data[test]
         y_train = target[train]
         y_test = target[test]

         clf1 = clf1.fit(X_train,y_train)
         clf2 = clf2.fit(X_train,y_train)
         clf3 = clf3.fit(X_train,y_train)
         clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
#         clf = clf.fit(X_train,y_train)
         rfe = RFE(clf1, 5)
#         clf = clf3.fit(X_train,y_train)

         clf = rfe.fit(X_train,y_train)

#         print("Selected Features: %s") % clf.support_
#         print("Feature Ranking: %s") % clf.ranking_

         importantes.append(map(lambda x: float(x), clf.support_))
#         importantes.append(clf.feature_importances_)

         preds = clf.predict(X_test)
         probas = clf.predict_proba(X_test)

         cv_target = np.concatenate((cv_target, y_test), axis=0) # concatenar los resultados
         cv_prediction = np.concatenate((cv_prediction, preds), axis=0)
         cv_probas = np.concatenate((cv_probas, probas[:,1]), axis=0)

   print maxValues(np.mean(importantes,axis=0))
   preds_thr = np.zeros(len(cv_target))
   thr_final = get_optimal_thr_diagonal_cm(cv_probas, cv_target, 0.01)
   preds_thr[np.where(cv_probas>thr_final)[0]] = 1
   cm = confusion_matrix(cv_target, preds_thr).astype(float)
   cm[0,:] = cm[0,:]/float(sum(cm[0,:])) # obtener matricz de confusion normalizada
   cm[1,:] = cm[1,:]/float(sum(cm[1,:]))

   fpr, tpr, thresholds = roc_curve(cv_target,  cv_probas) # obtener la curva ROC
   line, = plt.plot(fpr,tpr,label=name + " (AUC = " + str(round(auc(fpr,tpr),2)) + ")") # plotear curva ROC
   return auc(fpr,tpr), line, cm

# cargar matrices de correlacion

import os
import csv
import numpy
import sys

if len(sys.argv) > 1:
  metric = sys.argv[1]
else:
  metric = "correlation"

resting_data_open = {'beta':[], 'theta':[],'alpha':[]}
target = []

for band in ['beta','theta','alpha']:
  rootPath = '../../../informe/IMGs/Open/eyesOpen_'+band+'/'+metric+'/'
  for file in os.listdir(rootPath):
    if file[-4:] == '.csv':
      reader = csv.reader(open(rootPath + file, "rb"), delimiter=",")
      x = list(reader)
      x = numpy.array(x).astype("float")

      resting_data_open[band].append(x)

resting_data_closed = {'beta':[], 'theta':[],'alpha':[]}
target = []

for band in ['beta','theta','alpha']:
  rootPath = '../../../../informe/IMGs/Closed/eyesClosed_'+band+'/'+metric+'/'
  for file in os.listdir(rootPath):
    if file[-4:] == '.csv':
      reader = csv.reader(open(rootPath + file, "rb"), delimiter=",")
      x = list(reader)
      x = numpy.array(x).astype("float")

      resting_data_closed[band].append(x)


res = []
lines = []
cms = []

# Usamos cross validation para validar que el voting classifier funciona ok utilizando los datos de entrenamiento
set1 = np.concatenate((unfold_data(resting_data_open['theta']), unfold_data(resting_data_open['beta']), unfold_data(resting_data_open['alpha'])), axis=1)
set2 = np.concatenate((unfold_data(resting_data_closed['theta']), unfold_data(resting_data_closed['beta']), unfold_data(resting_data_closed['alpha'])), axis=1)
data, target = getTrainDataTarget(set1,set2)

n_folds = 6   # cantidad de folds
cvs = [StratifiedKFold(n_splits=n_folds) for _ in range(10)] # crear objeto de cross validation estratificada

auc_value, line,cm = fitModelwithCrossValidation(data, target, u'α+θ+β',cvs)
lines.append(line)
cms.append(cm)
res.append(metric+"\t"+'allBands' +"\t"+str(auc_value))

set1 = unfold_data(resting_data_open['alpha'])
set2 = unfold_data(resting_data_closed['alpha'])
data, target = getTrainDataTarget(set1,set2)

auc_value, line,cm = fitModelwithCrossValidation(data, target, u'α',cvs)
lines.append(line)
cms.append(cm)
res.append(metric+"\t"+'alpha' +"\t"+str(auc_value))

set1 = unfold_data(resting_data_open['theta'])
set2 = unfold_data(resting_data_closed['theta'])
data, target = getTrainDataTarget(set1,set2)

auc_value, line, cm = fitModelwithCrossValidation(data, target, u'θ',cvs)
lines.append(line)
cms.append(cm)
res.append(metric+"\t"+'theta' +"\t"+str(auc_value))

set1 = unfold_data(resting_data_open['beta'])
set2 = unfold_data(resting_data_closed['beta'])
data, target = getTrainDataTarget(set1,set2)

auc_value,line,cm = fitModelwithCrossValidation(data, target, u'β',cvs)
lines.append(line)
cms.append(cm)
res.append(metric+"\t"+'beta' +"\t"+str(auc_value))

x = [0.0, 1.0]
random, = plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='random (AUC = 0.5)')
lines.append(random)

plt.legend(handles=lines)
plt.title("Clasificador de Ojos Abiertos/Cerrados: Curva ROC")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig("eyes_roc.png")

plt.clf()

casos = [u'α+θ+β',u'α',u'θ',u'β']
for i in range(len(cms)):
   plot_confusion_matrix(cms[i], ["Ojos Abiertos","Ojos Cerrados"],title=casos[i]+' Matriz de Confusion')
   plt.clf()

print res
