from metricas import *
from os import listdir
import os
import matplotlib.pyplot as plt
from copy import copy, deepcopy

def setEEGLOader(experimentType, eegType, metrica, fmin, fmax):
	rootPath = DatasetsDir+ experimentType + '/' + eegType
	setFiles = find_filenames('.set',rootPath)
	for setFile in setFiles:

		path = rootPath + '/' + setFile
		print(path)
		raw = mne.io.read_raw_eeglab(path)

		print("IMPRIMO CH_NAMES de RAW CHANNELS")
		print(raw.ch_names)

		size = len(raw.ch_names)
		ultimo = size - 1
		rawChannels = [ raw[i][0][0] for i in range(size) if i != ultimo ]

		fileName = getFileName(setFile)
		if(metrica.name in metricasMatrix):
			matrix = metrica.apply(raw,fmin,fmax)
			matrixPlot(matrix, metrica.name, fileName, eegType, experimentType)
		else:
			guardarMatrix(metrica, rawChannels, fileName, eegType,fmin,fmax, experimentType)

# Savers

def guardarMatrix(metric, data, fileName, eegName,fmin,fmax, experimentType):
	matrix = mapMetricMatrix(metric,data,fmin,fmax)
	matrixPlot(matrix, metric.name, fileName, eegName, experimentType)

# Matrix Operations

def mapMetricMatrix(metrica,rawChannels,fmin,fmax):

	res = []
	size = len(rawChannels)

	for i in range(size):
		aux = []
		for j in range(size):
			value = metrica.apply(rawChannels[i], rawChannels[j],fmin,fmax)
			aux.append(value)
		res.append(aux)

	return np.array(res)

def zScore(matrix):
	values = []
	for i in range(len(matrix)):
		for j in range(len(matrix[0])):
			if j < i:
				values.append(matrix[i][j])
	values = np.array(values)
	mean = np.mean(values)
	std = np.std(values)
	for i in range(len(matrix)):
		for j in range(len(matrix[0])):
			if j < i:
				matrix[i][j] = (matrix[i][j]-mean)/std
			else:
				matrix[i][j] = 0.0
	return matrix


def fixElectrodes(matrix):
	matrixCopy = deepcopy(matrix)
	for i in range(len(biosemi2emotiv)):
		for j in range(len(biosemi2emotiv)):
			if j < i:
				a = biosemi2emotiv[i]
				b = biosemi2emotiv[j]
				if a < b:
					matrix[i][j] = matrixCopy[b][a]
				else:
					matrix[i][j] = matrixCopy[a][b]
	return matrix

def normalize(matrix):
	minNegative = 0
	for i in range(len(matrix)):
		for j in range(len(matrix[0])):
			if j < i:
				value = matrix[i][j]
				if value < minNegative:
					minNegative = value
	for i in range(len(matrix)):
		for j in range(len(matrix[0])):
			if j < i:
				matrix[i][j] = matrix[i][j] - minNegative
	maximum = 0
	for i in range(len(matrix)):
		for j in range(len(matrix[0])):
			if j < i:
				value = matrix[i][j]
				if value > maximum:
					maximum = value
	for i in range(len(matrix)):
		for j in range(len(matrix[0])):
			if j < i:
				matrix[i][j] = matrix[i][j]/maximum
	return matrix

def matrixPlot(matrix, metricName, fileName, eegName,experimentType):
	if eegName == 'biosemi':
		matrix = fixElectrodes(matrix)
	matrix = zScore(matrix)
	matrix = normalize(matrix)

	fig, ax = plt.subplots()
	heatmap = ax.pcolor(matrix, cmap=plt.cm.Blues)
	plt.colorbar(heatmap)

	ax.set_xticks(np.arange(np.array(matrix).shape[1]) + 0.5, minor=False)
	ax.set_yticks(np.arange(np.array(matrix).shape[0]) + 0.5, minor=False)
	ax.invert_yaxis()
	if eegName == 'emotiv':
		labels = emotivLabels
	else:
		labels = biosemiLabels
	ax.set_xticklabels(labels, minor=False)
	ax.set_yticklabels(labels, minor=False)

	basePath = IMGsDir + experimentType + '/' + eegName + '/' + metricName + '/'
	path = basePath + prefix(fileName) + '_'  + metricName
	if not os.path.exists(basePath):
	    os.makedirs(basePath)
	plt.savefig(path + '.png') # guardar graficos en .png
#	plt.savefig(path + '.svg', format='svg', dpi=1200) # guardar graficos en .svg
	plt.clf() # clean buffer
	nparray = np.asarray(matrix)
	np.savetxt(path+".csv", nparray, delimiter=",")


# Helpers
def getFileName(path):
    return path.split("/")[-1].split(".")[-2]

def find_filenames(extension, path):
	suffix = extension
	filenames = listdir(path)
	return [filename for filename in filenames if filename.endswith( suffix )]

def prefix(name):
	res = ''
	for letra in name:
		if not str.isdigit(letra):
			res += letra
		else:
			break
	return res