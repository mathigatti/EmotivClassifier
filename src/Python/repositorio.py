from metricas import *
from os import listdir
import os
import matplotlib.pyplot as plt
import statistics
from copy import copy, deepcopy

# Helpers
def getFileName(path):
    return path.split("/")[-1].split(".")[-2]

def find_filenames(extension, path):
	suffix = extension
	filenames = listdir(path)
	return [filename for filename in filenames if filename.endswith( suffix )]

# EDF Loader
def setLoader(metrica,fmin,fmax):
	setEEGLOader('emotiv', metrica,fmin,fmax)
	setEEGLOader('biosemi', metrica,fmin,fmax)

def setLoaderBase(metrica,fmin,fmax,eeg):
	setEEGLOaderBase(eeg, metrica,fmin,fmax)

def setEEGLOaderBase(eegType,metrica,fmin,fmax):
	setFiles = find_filenames('.set',DatasetsDir+eegType)
	for setFile in setFiles:

		path = DatasetsDir + eegType + '/' + setFile
		print(path)
		raw = mne.io.read_raw_eeglab(path)
		size = len(raw.ch_names)
		ultimo = size - 1
		rawChannels = [ raw[i][0][0] for i in range(size) if i != ultimo ]

		fileName = getFileName(setFile)

		if metrica.name in ['spearman','correlation']:
			matrix = mapMetricMatrixBase(metrica,rawChannels,fmin,fmax, False)
			matrixPlot(matrix, metrica.name + '1', fileName, eegType)

			matrix = mapMetricMatrixBase(metrica,rawChannels,fmin,fmax,True)
			matrixPlot(matrix, metrica.name + '2', fileName, eegType)
		else:
			matrix = metrica.apply(raw,fmin,fmax,False)
			matrixPlot(matrix, metrica.name + '1', fileName, eegType)

			matrix = metrica.apply(raw,fmin,fmax,True)
			matrixPlot(matrix, metrica.name + '2', fileName, eegType)

def setEEGLOader(eegType,metrica,fmin,fmax):
	setFiles = find_filenames('.set',DatasetsDir+eegType)
	for setFile in setFiles:

		path = DatasetsDir + eegType + '/' + setFile
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
			matrixPlot(matrix, metrica.name, fileName, eegType)
		else:
			guardarMatrix(metrica, rawChannels, fileName, eegType,fmin,fmax)

# Spectrum

def spectrumsCorrelation():

	emotivSpectrums = []
	biosemiSpectrums = []
	frequencies = []

	for eegType in ['emotiv','biosemi']:
		setFiles = find_filenames('.set',DatasetsDir+eegType)
		for setFile in setFiles:
			path = DatasetsDir + eegType + '/' + setFile
			raw = mne.io.read_raw_eeglab(path)
			size = len(raw.ch_names)
			ultimo = size - 1
			rawChannels = [ raw[i][0][0] for i in range(size) if i != ultimo ]

			frequencies, db_den = metricasDict['spectrum'].apply(rawChannels)

			name = prefix(getFileName(setFile))
			if(eegType == 'emotiv'):
				emotivSpectrums.append([name,db_den])
			else:
				biosemiSpectrums.append([name,db_den])
	
	maximum = 30
	minimum = 0
	i_max = 0
	i_min = 0
	n = len(frequencies)
	for i in range(n):
		if frequencies[i] <= maximum:
			i_max = i
		if frequencies[n - i - 1] >= minimum:
			i_min = n-i-1

	res = []
	matchScores = []
	for v in emotivSpectrums:
		temp = []
		for w in biosemiSpectrums:
			temp.append((w[0],stats.pearsonr(v[1][i_min:i_max+1],w[1][i_min:i_max+1])[0]))
		temp = sorted(temp, key=lambda x: -x[1])

		position = positionOnList(v[0],temp)
		matchScores.append(position)
		res.append((v[0],position))

	print(res)
	print('La mediana es: ' + str(statistics.median(matchScores)))
	print('La media es: ' + str(statistics.mean(matchScores)))

# Savers

def guardarMatrix(metric, data, fileName, eegName,fmin,fmax):
	matrix = mapMetricMatrix(metric,data,fmin,fmax)
	matrixPlot(matrix, metric.name, fileName, eegName)

def guardarPlot(metric, data):

	array_x, array_y = mapMetricContinuous(metric,data)

	size = len(data)

	count = 0
	for i in range(size):
		for j in range(size):
			if i < j :
				plot(array_x[count],array_y[count], metric.name, i,j)
				count += 1

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

def mapMetricMatrixBase(metrica,rawChannels,fmin,fmax,firstHalf):
	sfreq = 128
	window = 30*sfreq

	first_samp = window
	last_samp = len(rawChannels[0]) - window/5
	last_samp = min(last_samp, first_samp+50000)

	if firstHalf:
		first_samp = int(first_samp)
		last_samp = int((last_samp - first_samp) /2)
	else:
		first_samp = int((last_samp - first_samp) /2)
		last_samp = int(last_samp)

	res = []
	size = len(rawChannels)

	for i in range(size):
		aux = []
		for j in range(size):
			value = metrica.apply(rawChannels[i][first_samp:last_samp], rawChannels[j][first_samp:last_samp],fmin,fmax)
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

def matrixPlot(matrix, metricName, fileName, eegName):
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

	basePath = IMGsDir + eegName + '/' + metricName + '/'
	path = basePath + prefix(fileName) + '_'  + metricName
	if not os.path.exists(basePath):
	    os.makedirs(basePath)
	plt.savefig(path + '.png') # guardar graficos en .png
#	plt.savefig(path + '.svg', format='svg', dpi=1200) # guardar graficos en .svg
	plt.clf() # clean buffer
	nparray = np.asarray(matrix)
	np.savetxt(path+".csv", nparray, delimiter=",")

# Comparator

def compararMatrices(path1,path2):
	matrix1 = np.loadtxt(open(path1, "rb"), delimiter=",")
	matrix2 = np.loadtxt(open(path2, "rb"), delimiter=",")
	return comparadorCuadratico(matrix1,matrix2)

def findBiosemiMatch(biosemiMatrix, metricName):
	emotivPath = IMGsDir+'emotiv'+'/'+metricName + '/'
	biosemiPath = IMGsDir+'biosemi'+'/'+metricName + '/'

	biosemiMatrixPath = biosemiPath + biosemiMatrix

	emotivFiles = find_filenames(metricName+'.csv', emotivPath)
	res = []
	for emotivMatrixPath in emotivFiles:
		name = getFileName(emotivMatrixPath)
		res.append((name, compararMatrices(biosemiMatrixPath,emotivPath + emotivMatrixPath)))
	return sorted(res, key=lambda x: x[1])

def prefix(name):
	res = ''
	for letra in name:
		if not str.isdigit(letra):
			res += letra
		else:
			break
	return res

def positionOnList(prefixToMatch, matchList):
	for i in range(len(matchList)):
		if prefix(matchList[i][0]) == prefixToMatch:
			return i

def findMatchs(metricName):
	biosemiFiles = find_filenames(metricName+'.csv', IMGsDir+'biosemi'+'/'+metricName)
	listOfMatchs = []

	values = []
	matches = []
	for biosemiMatrixPath in biosemiFiles:
		name = getFileName(biosemiMatrixPath)
		prefixToMatch = prefix(name)
		matchList = findBiosemiMatch(biosemiMatrixPath,metricName)
		position = positionOnList(prefixToMatch, matchList)

		values.append(position)

		listOfMatchs.append((name, position))

	return listOfMatchs, statistics.mean(values), statistics.median(values), statistics.stdev(values)

def findMatchsBase(metricName, eeg):
	eegFiles1 = find_filenames(metricName+'1.csv', IMGsDir+eeg+'/'+metricName+'1')
	eegFiles2 = find_filenames(metricName+'2.csv', IMGsDir+eeg+'/'+metricName+'2')
	listOfMatchs = []
	matches = []
	values = []

	for eegMatrixPath in eegFiles1:
		name = getFileName(eegMatrixPath)
		prefixToMatch = prefix(name)
		matchList = findOtherHalfMatch(eegMatrixPath,eegFiles2,metricName,eeg)
		matches.append(matchList)
		position = positionOnList(prefixToMatch, matchList)

		values.append(position)

		listOfMatchs.append((name, position))
	print(matches)
	return listOfMatchs, statistics.mean(values), statistics.median(values), statistics.stdev(values)


def findOtherHalfMatch(eegMatrix, eegFiles2,metricName,eeg):
	eegPath1 = IMGsDir+eeg+'/'+metricName + '1/'
	eegPath2 = IMGsDir+eeg+'/'+metricName + '2/'

	eegMatrixPath = eegPath1 + eegMatrix
	print(eegMatrixPath)
	res = []
	for eegMatrixPath2 in eegFiles2:
		name = getFileName(eegMatrixPath2)
		res.append((name, compararMatrices(eegMatrixPath,eegPath2 + eegMatrixPath2)))
		print(eegPath2 + eegMatrixPath2)
	return sorted(res, key=lambda x: x[1])

def spectrumsRaster():
	experimentName = 'Raster de Espectros'
	samplesPerFile = []

	for eegType in ['emotiv', 'biosemi']:
		setFiles = find_filenames('.set',DatasetsDir+eegType)

		matrix = []
		for setFile in setFiles:
			path = DatasetsDir + eegType + '/' + setFile

			raw = mne.io.read_raw_eeglab(path)
			size = len(raw.ch_names)
			ultimo = size - 1

			rawChannels = [ raw[i][0][0] for i in range(size) if i != ultimo ]

			matrixFile = [0 for _ in range(len(rawChannels))]
			for channel in rawChannels:
			    f, Pxx_den =signal.welch(channel,fs=128)
			    db_den = 10*np.log10(Pxx_den)
			    for i in range(len(matrixFile)):
			    	if matrixFile[i] < db_den[i]:
			    		matrixFile[i] = db_den[i]

			matrix.append(matrixFile)

		fig, ax = plt.subplots(figsize=(len(matrix[0]),len(matrix)))
		ax.grid(True)
		heatmap = ax.pcolor(matrix, cmap=plt.cm.Blues)
		plt.colorbar(heatmap)

		ax.set_xticks(np.arange(np.array(matrix).shape[1]) + 0.5, minor=False)
		ax.set_yticks(np.arange(np.array(matrix).shape[0]) + 0.5, minor=False)
		ax.invert_yaxis()


		labels = f
		plt.xticks(range(len(matrix[0])), labels);
		if eegType == 'emotiv':
			labels = emotivLabels
		else:
			labels = biosemiLabels
		plt.yticks(range(len(matrix)), labels);

		fileName = getFileName(setFile)

		path = IMGsDir + experimentName + '/' + prefix(fileName) + '_' + eegType
		plt.savefig(path + '.png') # guardar graficos en .png
		plt.clf() # clean buffer

		samplesPerFile.append( (fileName, len(rawChannels[0])) )

	print(samplesPerFile)

def processAllSpectrums():
	experimentName = 'espectros'
	samplesPerFile = []

	for eegType in ['emotiv', 'biosemi']:
		setFiles = find_filenames('.set',DatasetsDir+eegType)

		matrix = []
		for setFile in setFiles:
			path = DatasetsDir + eegType + '/' + setFile

			raw = mne.io.read_raw_eeglab(path)
			size = len(raw.ch_names)
			ultimo = size - 1

			rawChannels = [ raw[i][0][0] for i in range(size) if i != ultimo ]

			for channel in rawChannels:
				f,db_den = spectrum(channel)
				plt.plot(f,db_den)

			fileName = getFileName(setFile)

			path = IMGsDir + experimentName + '/' + prefix(fileName) + '_' + eegType

			plt.savefig(path + '.png') # guardar graficos en .png
			plt.clf() # clean buffer

		samplesPerFile.append( (fileName, len(rawChannels[0])) )

	print(samplesPerFile)
