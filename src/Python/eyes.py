import sys

from metricas import *
from repositorio import *
from avgPath import *

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset("tips")

def matrixPlot2(matrix, metricName, fileName, eegName):
	print('\n\n')
	print(fileName)
	print('\n\n')
	matrix = zScore(matrix)
	matrix = normalize(matrix)

	fig, ax = plt.subplots()
	heatmap = ax.pcolor(matrix, cmap=plt.cm.Blues)
	plt.colorbar(heatmap)

	ax.set_xticks(np.arange(np.array(matrix).shape[1]) + 0.5, minor=False)
	ax.set_yticks(np.arange(np.array(matrix).shape[0]) + 0.5, minor=False)
	ax.invert_yaxis()
	labels = emotivLabels
	ax.set_xticklabels(labels, minor=False)
	ax.set_yticklabels(labels, minor=False)

	basePath = IMGsDir + eegName + '/' + metricName + '/'
	print('\n\n')
	print(fileName)
	print('\n\n')
	path = basePath + fileName[:5] + '_'  + metricName
    #	path = basePath + prefix(fileName) + '_'  + metricName
	if not os.path.exists(basePath):
	    os.makedirs(basePath)
	plt.savefig(path + '.png') # guardar graficos en .png
	plt.clf() # clean buffer
	nparray = np.asarray(matrix)
	np.savetxt(path+".csv", nparray, delimiter=",")

def guardarMatrix2(metric, data, fileName, eegName,fmin,fmax):
	matrix = mapMetricMatrix(metric,data,fmin,fmax)
	matrixPlot2(matrix, metric.name, fileName, eegName)

def loadCorMatrixes(metrica, fmin, fmax,folder,basePath):
	setFiles = find_filenames('.set',basePath)
	for setFile in setFiles:
		path = basePath + setFile
		raw = mne.io.read_raw_eeglab(path)
		size = len(raw.ch_names)
		ultimo = size - 1
		rawChannels = [ raw[i][0][0] for i in range(size) if i != ultimo ]
		fileName = getFileName(setFile)
		if(metrica.name in metricasMatrix):
			matrix = metrica.apply(raw,fmin,fmax)
			matrixPlot2(matrix, metrica.name, fileName, folder)
		else:
			guardarMatrix2(metrica, rawChannels, fileName, folder,fmin,fmax)

def main():
#	for metrica in [correlation,spearman]:
	for metricaName in ['plv','imcoh','ppc','pli2_unbiased','wpli2_debiased','pli','coh']:
		metrica = comodin(metricaName)
		for name,f_min,f_max in [('theta',4,7.5),('alpha',8,14),('beta',15,25)]:
			loadCorMatrixes(metrica,f_min,f_max,'eyesOpen_'+name,DatasetsDirSource+'Open/emotiv/')
			loadCorMatrixes(metrica,f_min,f_max,'eyesClosed_'+name,DatasetsDirSource+'Closed/emotiv/')

main()