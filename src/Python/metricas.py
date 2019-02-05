
from constants import *
from scipy import signal, stats
import numpy as np
from functools import partial
from mne.connectivity import spectral_connectivity
import mne
import math

from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Funciones Auxiliares

def rango(frequenciesList, fmin):
	result = 0
	for i in range(fmin,fmin+4):
		result += frequenciesList[i]
	return result

# De 4Hz a 7.5Hz
def theta(frequenciesList):
	result = 0
	for i in range(8,16):
		result += frequenciesList[i]
	return result

# De 8Hz a 14Hz
def alpha(frequenciesList):
	result = 0
	for i in range(16,29):
		result += frequenciesList[i]
	return result

# De 15Hz a 25Hz CORREGIR!!!!!!!
def beta(frequenciesList):
	result = 0
	for i in range(29,49):
		result += frequenciesList[i]
	return result

# Metricas

class Metrica:
    def __init__(self, implementation, name):
        self.apply = implementation
        self.name = name

def spectrum(channel):
    f, Pxx_den =signal.welch(channel,fs=128)
    db_den = 10*np.log10(Pxx_den)
    return f, db_den

def spectrumFunction(rawChannels):
	db_dens = []

	f = []
	for channel in rawChannels:
		f, db_den = spectrum(channel)
		db_dens.append(db_den)

	return f,np.array(db_dens).mean(0)

espectro = Metrica(spectrumFunction, "spectrum")

def rangoCoherenceFunction(signal_1, signal_2,fmin):
	#128 Hz frecuancia de muestreo de emotiv, biosemi se llevo tambien a esta frecuencia
	_,y = signal.coherence(signal_1, signal_2,fs=128)
	return rango(y,fmin)

rangoCoherence = Metrica(rangoCoherenceFunction, "coherenceRango")

def coherenceFunction(signal_1, signal_2,meanSpectrum):
	#128 Hz frecuancia de muestreo de emotiv, biosemi se llevo tambien a esta frecuencia
	_,y = signal.coherence(signal_1, signal_2,fs=128)
	return meanSpectrum(y)

alphaCoherence = Metrica(partial(coherenceFunction, meanSpectrum=alpha), "coherenceAlpha")

thetaCoherence = Metrica(partial(coherenceFunction, meanSpectrum=theta), "coherenceTheta")

betaCoherence = Metrica(partial(coherenceFunction, meanSpectrum=beta), "coherenceBeta")

def correlationFunction(signal_1, signal_2,fmin,fmax): 
	signal_1 = butter_bandpass_filter(signal_1, fmin, fmax, 128, order=5)
	signal_2 = butter_bandpass_filter(signal_2, fmin, fmax, 128, order=5)
	return stats.pearsonr(signal_1,signal_2)[0]

correlation = Metrica(correlationFunction, "correlation")

def spearmanFunction(signal_1, signal_2,fmin,fmax):
	signal_1 = butter_bandpass_filter(signal_1, fmin, fmax, 128, order=5)
	signal_2 = butter_bandpass_filter(signal_2, fmin, fmax, 128, order=5)
	return stats.spearmanr(signal_1,signal_2)[0]

spearman = Metrica(spearmanFunction, "spearman")

def h(raw_EEG_data, method,fmin,fmax):
	sfreq = raw_EEG_data.info['sfreq']  # the sampling frequency

	window = 30*sfreq
	epoch_size = 1000

	last_samp = int(raw_EEG_data.last_samp - window/3)

	t_events = np.arange(window, min(50000+window, last_samp), epoch_size)

	events = np.zeros((len(t_events), 3), dtype=np.int)
	events[:, 0] = t_events
	events[:, 2] = 1 # ID of the event

	event_id, tmin, tmax = 1, -0.2, 0.5

	epochs = mne.Epochs(raw_EEG_data, events, event_id, tmin, tmax, proj=False,
						baseline=(None, 0), preload=True)

	tmin = 0.0  # exclude the baseline period

	con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
	    epochs, method=method, mode='multitaper', sfreq=sfreq, fmin=fmin, fmax=fmax,
	    faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1)

	ch_names = epochs.ch_names
	print("IMPRIMO CHANNEL NAMES de EPOCHS")
	print(ch_names)

	con = con[0:14]
	matrix = []
	for lista in con:
		sublista = []
		for elem in lista[0:14]:
			sublista.append(elem[0])
		matrix.append(sublista)
	return np.array(matrix)

def pliFunction(raw_EEG_data,fmin,fmax):
	return h(raw_EEG_data,'coh',fmin,fmax)

pli = Metrica(pliFunction, "pli")

def plvFunction(raw_EEG_data,fmin,fmax):
	return h(raw_EEG_data,'plv',fmin,fmax)
	
plv = Metrica(plvFunction, "plv")

def cohFunction(raw_EEG_data,fmin,fmax):
	return h(raw_EEG_data,'coh',fmin,fmax)

coh = Metrica(cohFunction, "coh")

def hHalf(raw_EEG_data, method,fmin,fmax,firstHalf):
	sfreq = raw_EEG_data.info['sfreq']  # the sampling frequency

	window = 30*sfreq
	epoch_size = 2000

	print(raw_EEG_data.first_samp)
	print(raw_EEG_data.last_samp)

	first_samp = raw_EEG_data.first_samp + window
	last_samp = raw_EEG_data.last_samp - window/5
	last_samp = min(last_samp, first_samp+50000)

	if firstHalf:
		last_samp = int((last_samp - first_samp) /2)
	else:
		first_samp = int((last_samp - first_samp) /2)

	t_events = np.arange(first_samp, last_samp, epoch_size)
	events = np.zeros((len(t_events), 3), dtype=np.int)
	events[:, 0] = t_events
	events[:, 2] = 1 # ID of the event

	event_id, tmin, tmax = 1, -0.2, 0.5

	epochs = mne.Epochs(raw_EEG_data, events, event_id, tmin, tmax, proj=False,
						baseline=(None, 0), preload=True)


	tmin = 0.0  # exclude the baseline period

	con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
	    epochs, method=method, mode='multitaper', sfreq=sfreq, fmin=fmin, fmax=fmax,
	    faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1)

	con = con[0:14]
	matrix = []
	for lista in con:
		sublista = []
		for elem in lista[0:14]:
			sublista.append(elem[0])
		matrix.append(sublista)
	return np.array(matrix)

def pliHalfFunc(raw_EEG_data,fmin,fmax,firstHalf):
	return hHalf(raw_EEG_data,'coh',fmin,fmax,firstHalf)	

pliHalf = Metrica(pliHalfFunc, "pliHalf")


def plvFunction(raw_EEG_data,fmin,fmax):
	return h(raw_EEG_data,'plv',fmin,fmax)

comodin = lambda name : Metrica(lambda raw_EEG_data,fmin,fmax: h(raw_EEG_data,name,fmin,fmax), name)
comodinHalf = lambda name : Metrica(lambda raw_EEG_data,fmin,fmax,firstHalf: hHalf(raw_EEG_data,name,fmin,fmax,firstHalf), name)

# Variables globales

metricasDict = {pliHalf.name: pliHalf, espectro.name: espectro, spearman.name: spearman, rangoCoherence.name:rangoCoherence, coh.name:coh, plv.name:plv, pli.name:pli, betaCoherence.name:betaCoherence, thetaCoherence.name:thetaCoherence, alphaCoherence.name:alphaCoherence, correlation.name:correlation}

# Comparacion

def comparadorManhattan(matrix1, matrix2):
	res = 0
	for i in range(len(matrix1)):
		for j in range(len(matrix1[0])):
			if j < i:
				res += abs(matrix1[i,j] - matrix2[i,j])
	return res

def comparadorCuadratico(matrix1, matrix2):
	res = 0
	for i in range(len(matrix1)):
		for j in range(len(matrix1[0])):
			if j < i:
				res += pow(matrix1[i,j] - matrix2[i,j],2)
	return res

def comparadorCuadraticoPesado(matrix1, matrix2):
	n = len(matrix1)
	m = len(matrix1[0])
	weights = np.zeros((n, m))

	for i in range(n):
		for j in range(m):
			if j < i:
				e_i = emotivLabels[i]
				e_j = emotivLabels[j]
				weights[i][j] = math.exp(-1*lamda*electrodeDistances[e_i][e_j])

	c = sum(sum(weights))

	res = 0
	for i in range(n):
		for j in range(m):
			if j < i:
				res += weights[i][j]*pow(matrix1[i,j] - matrix2[i,j],2)/c
	return res

"""
Faltan implementar:

	weighted Symbolic Mutual Information (wSMI)

	Causalidad de Granger
	
	Synchronization Likelihood (Esta en matlab)

Ya implementadas:

	Phase-Locking Index (PLI)

	Correlaciones

	Coherence https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2151962/
"""