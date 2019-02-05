# Code inspired from:
# https://www.martinos.org/mne/stable/auto_examples/connectivity/plot_sensor_connectivity.html#sphx-glr-auto-examples-connectivity-plot-sensor-connectivity-py
# https://plot.ly/python/3d-network-graph/

from repositorio import *
import sys

import numpy as np
from scipy import linalg
import mne
from mne import io
from mne.connectivity import spectral_connectivity
from mne.datasets import sample
import statistics

import matplotlib.pyplot as plt
import networkx as nx
import scipy.stats
import random

def readConnectionsFile(experimentType, eegType, setFile):
  path = DatasetsDirSource+experimentType+'/'+eegType + '/' + setFile
  print(path)
  raw = io.read_raw_eeglab(path)

  t_events = np.arange(raw.first_samp + 2000, raw.last_samp - 2000, 1000)
  events = np.zeros((len(t_events), 3), dtype=np.int)
  events[:, 0] = t_events
  events[:, 2] = 1 # ID of the event

  event_id, tmin, tmax = 1, -0.2, 0.5

  epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
            baseline=(None, 0), preload=True)

  sfreq = raw.info['sfreq']  # the sampling frequency

  # Compute connectivity for band containing the evoked response.
  # We exclude the baseline period
  #fmin, fmax = 8., 16.
  fmin, fmax = 5., 24.
  tmin = 0.0  # exclude the baseline period

  con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
      epochs, method='pli', mode='multitaper', sfreq=sfreq, fmin=fmin, fmax=fmax,
      faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1)

  # the epochs contain an EOG channel, which we remove now
  ch_names = epochs.ch_names
  idx = [ch_names.index(name) for name in ch_names if name != 'STI 014']
  # Plot the sensor locations
  sens_loc = [raw.info['chs'][i]['loc'][:3] for i in idx]
  sens_loc = np.array(sens_loc)

  con = con[idx][:, idx]
  return ch_names, sens_loc, np.array(con[:, :, 0])

def discretizeMatrix(matrix):
    #matrix = abs(matrix)
    n_con = 20  # number of connections to show up
    threshold = np.sort(matrix, axis=None)[-n_con]

    ii, jj = np.where(abs(matrix) > threshold)

    highValues = zip(ii, jj)

    for i in range(len(matrix)):
    	for j in range(len(matrix[0])):
    		if (i,j) in highValues:
    			matrix[i,j] = 1
    		else:
    			matrix[i,j] = 0

    return matrix

def findAvgPaths():
  emotivChannelLocs = []
  emotivChannelNames = []
  experimentResults = []
  controlExperimentResults = []
  for experimentType in ['Open','Closed']:
    eegResults = {}
    for eegType in ['emotiv','biosemi']:
      setFiles = find_filenames('.set',DatasetsDirSource+experimentType+'/'+eegType)
      for setFile in setFiles:
        ch_names, sens_loc, con = readConnectionsFile(experimentType, eegType, setFile)

        con = zScore(con)
        con = discretizeMatrix(con)

        if(eegType == 'biosemi'):
            sens_loc = emotivChannelLocs
            ch_names = emotivChannelNames
            fixElectrodes(con)
        else:
            emotivChannelNames = ch_names
            emotivChannelLocs = sens_loc

        graph = nx.from_numpy_matrix(con)
		
        total = 0
        for subGraph in nx.connected_component_subgraphs(graph):
            total += nx.average_shortest_path_length(subGraph)
	    
        prefixName = prefix(setFile)

        if eegType == 'emotiv':
            eegResults[prefixName] = total
        else:
            eegResults[prefixName] = [total,eegResults[prefixName]]

    if experimentType == 'Open':
        values = map(list, zip(*eegResults.values()))
    else:
        aux = map(list, zip(*eegResults.values()))
        values[0] = values[0] + aux[0]
        values[1] = values[1] + aux[1]
    print(values)

  test = values[1]
  experimentResults.append(stats.wilcoxon(values[0],test))
  random.shuffle(test)
  controlExperimentResults.append(stats.wilcoxon(values[0],test))

  for i in range(len(experimentResults)):
    print("The experimentResults are:")
    print("\t" + str(experimentResults[i]))
    print("The controlExperimentResults are:")
    print("\t" + str(controlExperimentResults[i]))
