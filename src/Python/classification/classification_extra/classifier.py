# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pickle

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

def plotMeanMatrix(resting_data):
  matrixPlot(np.mean(zScore(resting_data),axis=0), "W")

def getTrainDataTarget():


   # seleccionar fases de suenio que se van a comparar
   set1 = np.concatenate((unfold_data(resting_data_female['theta']), unfold_data(resting_data_female['beta']), unfold_data(resting_data_female['alpha'])), axis=1)
   set2 = np.concatenate((unfold_data(resting_data_male['theta']), unfold_data(resting_data_male['beta']), unfold_data(resting_data_male['alpha'])), axis=1)

#   target1 = numpy.random.randint(0,2,size=set1.shape[0])
#   target2 = numpy.random.randint(0,2,size=set2.shape[0])
   target1 = np.zeros(set1.shape[0]) 
   target2 = np.ones(set2.shape[0])
   print target1
   print target2

   data = np.concatenate((set1,set2), axis=0)
   target = np.concatenate((target1, target2), axis=0)

   return data, target

def getClassifiers():
   n_estimators = 16   # cantidad de arboles
   max_depth = 4 # maxima profundidad de los arboles

   clf1 = LogisticRegression()
   clf2 = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
   clf3 = BernoulliNB()
   return clf1, clf2, clf3

def fitModelwithCrossValidation():
   data, target = getTrainDataTarget()

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

   preds_thr = np.zeros(len(cv_target))
   thr_final = get_optimal_thr_diagonal_cm(cv_probas, cv_target, 0.01)
   preds_thr[np.where(cv_probas>thr_final)[0]] = 1
   cm = confusion_matrix(cv_target, preds_thr).astype(float)
   cm[0,:] = cm[0,:]/float(sum(cm[0,:])) # obtener matricz de confusion normalizada
   cm[1,:] = cm[1,:]/float(sum(cm[1,:]))

   fpr, tpr, thresholds = roc_curve(cv_target,  cv_probas) # obtener la curva ROC
   plt.plot(fpr,tpr) # plotear curva ROC
   plt.title("Sex ROC Curves")
   plt.savefig("sex_roc.png")

   print "AUC Entrenamiento del Voting Classifier usando Cross-Validation:", auc(fpr, tpr)

def evaluateModel():
   data, target = getTrainDataTarget()

   # Entrenamos el modelo
   clf1, clf2, clf3 = getClassifiers()
   clf1 = clf1.fit(data,target)
   clf2 = clf2.fit(data,target)
   clf3 = clf3.fit(data,target)
   clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
   clf = clf.fit(data,target)

   # Evaluamos con datos de propofol
   propofol_data_awake = unfold_data(zScore(anesthesia_data['W']))
   propofol_data_awake = np.concatenate((propofol_data_awake, qtyOfSmallCorrelations(propofol_data_awake)), axis=1)

   propofol_data_loc = unfold_data(zScore(anesthesia_data['LOC']))
   propofol_data_loc = np.concatenate((propofol_data_loc, qtyOfSmallCorrelations(propofol_data_loc)), axis=1)

   set1 = propofol_data_awake
   set2 = propofol_data_loc

   target1 = np.zeros(set1.shape[0])
   target2 = np.ones(set2.shape[0])

   data = np.concatenate((set1,set2), axis=0)
   target = np.concatenate((target1, target2), axis=0)
   probas = clf.predict_proba(data)[:,1]

   preds_thr = np.zeros(len(target))
   thr_final = get_optimal_thr_diagonal_cm(probas, target, 0.01)

   preds_thr[np.where(probas>thr_final)[0]] = 1
   cm_propofol = confusion_matrix(target, preds_thr).astype(float)
   cm_propofol[0,:] = cm_propofol[0,:]/float(sum(cm_propofol[0,:])) # obtener matriz de confusion normalizada
   cm_propofol[1,:] = cm_propofol[1,:]/float(sum(cm_propofol[1,:]))

   fpr_propofol, tpr_propofol, thresholds_propofol = roc_curve(target, probas) # obtener la curva ROC

   plt.plot(fpr_propofol,tpr_propofol) # plotear curva ROC

   return auc(fpr_propofol,tpr_propofol)

# cargar matrices de correlacion

import os
import csv
import numpy
import sys

rootPath = '../../informe/IMGs/Open/'
sex = json.load(open(rootPath + 'sex.json'))

metric = sys.argv[1]
resting_data_female = {'beta':[], 'theta':[],'alpha':[]}
resting_data_male = {'beta':[], 'theta':[],'alpha':[]}
for band in ['beta','theta','alpha']:
  rootPath = '../../informe/IMGs/Open/ninios_'+band+'/'+metric+'/'
  for file in os.listdir(rootPath):
    if file[-4:] == '.csv':
      reader = csv.reader(open(rootPath + file, "rb"), delimiter=",")
      x = list(reader)
      x = numpy.array(x).astype("float")

#      np.random.shuffle(x) # Desordeno como verificación de que no da bien cualquier cosa

      if sex[file[:5].upper()] != 'Masculino':
        resting_data_female[band].append(x)
      else:
        resting_data_male[band].append(x)

resting_data_female = {k:np.array(v) for k,v in resting_data_female.items()}
resting_data_male = {k:np.array(v) for k,v in resting_data_male.items()}

# Graficamos las matrices de correlacion media de cada tipo de estado para ayudarnos a decidir que features pueden servir
#plotMeanMatrix(resting_data)

# Usamos cross validation para validar que el voting classifier funciona ok utilizando los datos de entrenamiento
fitModelwithCrossValidation()

# Calculamos el AUC al evaluar el modelo i iteraciones e imprimimos el promedio que resulta

'''
auc_total = []
i = 10
for x in xrange(0,i):
   auc_total.append(evaluateModel())
plt.title("Propofol ROC Curves")
plt.savefig("propofol_roc.png")
auc_total = np.array(auc_total)
print "AUC medio para datos de Propofol (" + str(i) + " iteraciones): " + str(np.mean(auc_total)) + " y desvio estandar de " + str(np.std(auc_total))
'''