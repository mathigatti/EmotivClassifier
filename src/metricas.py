
from constants import *
from scipy import stats
import numpy as np
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

# Metricas

class Metrica:
    def __init__(self, implementation, name):
        self.apply = implementation
        self.name = name

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

comodin = lambda name : Metrica(lambda raw_EEG_data,fmin,fmax: h(raw_EEG_data,name,fmin,fmax), name)

# Variables globales

metricasDict = {spearman.name: spearman, correlation.name:correlation}