# -*- coding: utf-8 -*-

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import VotingClassifier

from scipy.stats import zscore
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import itertools
import os
import csv
import sys

# Función para elegir el umbral que resulte en valores balanceados en la diagonal
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

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
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
    plt.savefig('sleep_' + title.lower().replace(" ","_") + ".png")

def maxValues(values):
  electrodes = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

  res = []
  count = 0
  for i in range(14):
    for j in range(i):
      res.append((i,j,values[count]))
      count += 1

  return map(lambda x: (electrodes[x[0]],electrodes[x[1]],x[-1]),sorted(res,key=lambda x:-x[-1]))

# Funcion para expandir las matrices en una lista y tomar la parte triangular superior    
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

def fitModelwithCrossValidation(data, target, name,cvs):
   n_estimators = 16 # cantidad de arboles
   max_depth = 4 # maxima profundidad de los arboles
   clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

   cv_target = np.array([])
   cv_prediction = np.array([])
   cv_probas = np.array([])

   main_attributes = []
   for cv in cvs:
     for train, test in cv.split(data, target):
         X_train = data[train] # crear sets de entrenamiento y testeo para el fold
         X_test = data[test]
         y_train = target[train]
         y_test = target[test]

         rfe = RFE(clf, 5)

         clf = rfe.fit(X_train,y_train)

         main_attributes.append(map(lambda x: float(x), clf.support_))

         preds = clf.predict(X_test)
         probas = clf.predict_proba(X_test)

         cv_target = np.concatenate((cv_target, y_test), axis=0) # concatenar los resultados
         cv_prediction = np.concatenate((cv_prediction, preds), axis=0)
         cv_probas = np.concatenate((cv_probas, probas[:,1]), axis=0)

   print(maxValues(np.mean(main_attributes,axis=0)))

   preds_thr = np.zeros(len(cv_target))
   thr_final = get_optimal_thr_diagonal_cm(cv_probas, cv_target, 0.01)
   preds_thr[np.where(cv_probas>thr_final)[0]] = 1
   cm = confusion_matrix(cv_target, preds_thr).astype(float)
   cm[0,:] = cm[0,:]/float(sum(cm[0,:])) # obtener matricz de confusion normalizada
   cm[1,:] = cm[1,:]/float(sum(cm[1,:]))

   fpr, tpr, thresholds = roc_curve(cv_target,  cv_probas) # obtener la curva ROC
   line, = plt.plot(fpr,tpr,label=name + " (AUC = " + str(round(auc(fpr,tpr),2)) + ")") # plotear curva ROC
   return auc(fpr,tpr), line, cm

def loadCorrelationMatrices():
  rootPathCorrelationsMatrix = sys.argv[1]
  targetPath = sys.argv[2]
  columnName = sys.argv[3]
  threshold = float(sys.argv[4])

  targetDF = pd.read_csv(targetPath)
  targetDF = targetDF.dropna(axis=0)
  targetDF[columnName] = targetDF[[columnName]].apply(zscore)

  data = []
  target = []

  for band in ['beta','theta','alpha']:
    for file in os.listdir(rootPathCorrelationsMatrix):
      if file[-4:] == '.csv' and targetDF['NCaso'].str.contains(file[:5]).any():
        reader = csv.reader(open(rootPathCorrelationsMatrix + file, "rb"), delimiter=",")
        x = list(reader)
        x = np.array(x).astype("float")

        data.append(x)
        target.append(any(targetDF[targetDF['NCaso'] == file[:5]][columnName] < threshold))

  target = np.array(target)
  data = np.array(data)

  data = unfold_data(data)
  return data, target

def run_experiment()
  data, target = loadCorrelationMatrices()

  n_folds = 6 # cantidad de folds
  cvs = [StratifiedKFold(n_splits=n_folds) for _ in range(10)] # crear objeto de cross validation estratificada

  lines = []
  cms = []

  auc_value, line,cm = fitModelwithCrossValidation(data, target, u'α',cvs)
  lines.append(line)
  cms.append(cm)

  print('auc_value' +"\t"+str(auc_value))

  x = [0.0, 1.0]
  random, = plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='random (AUC = 0.5)')
  lines.append(random)

  plt.legend(handles=lines)
  plt.title(f"Clasificador de {columnName}: Curva ROC")
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.savefig("roc.png")

  plt.clf()

  titles = [u'α+θ+β',u'α',u'θ',u'β']
  for i in range(len(cms)):
     plot_confusion_matrix(cms[i], [f"{columnName} Insuficiente",f"{columnName} Normal"],title=titles[i]+' Matriz de Confusion')
     plt.clf()

    
run_experiment()
