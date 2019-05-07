from metrics import *
import os
import matplotlib.pyplot as plt

def computeCorrelationMatrix(metric,band,fmin,fmax,inputDir,outputDir):
	setFiles = find_filenames('.set',inputDir)
	for setFile in setFiles:
		path = inputDir + "/" + setFile
		print(path)
		raw = mne.io.read_raw_eeglab(path)

		size = len(raw.ch_names)
		ultimo = size - 1
		rawChannels = [ raw[i][0][0] for i in range(size) if i != ultimo ]

		fileName = getFileName(setFile)
		print(setFile)
		print(fileName)
		if(metric.name in metricsMatrix):
			matrix = metric.apply(raw,fmin,fmax)
			matrixPlot(matrix, metric.name, band, fileName, outputDir)
		else:
			saveMatrix(metric, band, rawChannels, fileName, fmin, fmax, outputDir)


# Helpers
def getFileName(path):
    return path.split("/")[-1].split(".")[-2]

def find_filenames(extension, path):
	suffix = extension
	filenames = os.listdir(path)
	return [filename for filename in filenames if filename.endswith( suffix )]


def saveMatrix(metric, band, data, fileName, fmin,fmax,outputDir):
	matrix = mapMetricMatrix(metric,data,fmin,fmax)
	matrixPlot(matrix, metric.name, band, fileName, outputDir)


def matrixPlot(matrix, metricName, band, fileName, outputDir):
	matrix = zScore(matrix)
	matrix = normalize(matrix)

	fig, ax = plt.subplots()
	heatmap = ax.pcolor(matrix, cmap=plt.cm.Blues)
	plt.colorbar(heatmap)

	ax.set_xticks(np.arange(np.array(matrix).shape[1]) + 0.5, minor=False)
	ax.set_yticks(np.arange(np.array(matrix).shape[0]) + 0.5, minor=False)
	ax.invert_yaxis()
	#labels = ['C26', 'D6', 'C32', 'D11', 'D23', 'D31', 'A15', 'A28', 'B11', 'B26', 'B30', 'C10', 'C6', 'C13']
	labels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
	ax.set_xticklabels(labels, minor=False)
	ax.set_yticklabels(labels, minor=False)

	basePath = outputDir + '/' + metricName + '/' + band + '/'
	path = basePath + fileName
	print(path)
	if not os.path.exists(basePath):
	    os.makedirs(basePath)
	plt.savefig(path + '.png') # save as .png
	#plt.savefig(path + '.svg', format='svg', dpi=1200) # save as .svg
	plt.clf() # clean buffer
	nparray = np.asarray(matrix)
	np.savetxt(path+".csv", nparray, delimiter=",")

# Matrix Operations

def mapMetricMatrix(metric,rawChannels,fmin,fmax):

	res = []
	size = len(rawChannels)

	for i in range(size):
		aux = []
		for j in range(size):
			value = metric.apply(rawChannels[i], rawChannels[j],fmin,fmax)
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
