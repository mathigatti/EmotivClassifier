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

def readConnectionsFile(experimentType, eegType, setFile):
  path = DatasetsDirSource+experimentType+'/'+eegType + '/' + setFile
  raw = io.read_raw_eeglab(path)

  t_events = np.arange(raw.first_samp + 1000, raw.last_samp - 1000, 10000)
  events = np.zeros((len(t_events), 3), dtype=np.int)
  events[:, 0] = t_events
  events[:, 2] = 1 # ID of the event

  event_id, tmin, tmax = 1, -0.2, 0.5

  epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
            baseline=(None, 0), preload=True)

  sfreq = raw.info['sfreq']  # the sampling frequency

  # Compute connectivity for band containing the evoked response.
  # We exclude the baseline period
  fmin, fmax = 8., 16.
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

def assignedBrainSection(label):
  pt_izq = ['P7', 'T7', 'O1'] 
  pt_der = ['P8', 'T8', 'O2'] 
  frontal_izq = ['F3', 'F7', 'AF3']
  frontal_der = ['F4', 'F8', 'AF4']

  emotivLabels = [pt_izq, pt_der, frontal_der,frontal_izq]
  for i in range(len(emotivLabels)):
    if label in emotivLabels[i]:
      return i
  return -1

def main(argv):
  emotivChannelLocs = []
  emotivChannelNames = []
  
  lista = ['ak', 'gs', 'lt', 'mcf', 'mdp', 'mf', 'ml', 'mrv', 'mv', 'nmg', 'vg']

  for eegType in ['emotiv','biosemi']:
    setFilesOpen = find_filenames('.set',DatasetsDirSource+'Open'+'/'+eegType)
    setFilesClosed = find_filenames('.set',DatasetsDirSource+'Closed'+'/'+eegType)
    accum = []
    sens_loc = []
    ch_names = []
    for setFileOpen in setFilesOpen:
      for setFileClosed in setFilesClosed:
        if prefix(setFileOpen) in lista and prefix(setFileOpen) == prefix(setFileClosed):
          # con is a 3D array where the last dimension is size one since we averaged
          # over frequencies in a single band. Here we make it 2D
          ch_names, sens_loc, conOpen = readConnectionsFile('Open', eegType, setFileOpen)
          ch_names, sens_loc, conClosed = readConnectionsFile('Closed', eegType, setFileClosed)

          con = zScore(conOpen - conClosed)

          if accum == []:
            accum = np.zeros(np.shape(con))

          accum += con

    if(eegType == 'biosemi'):
      sens_loc = emotivChannelLocs
      ch_names = emotivChannelNames
      fixElectrodes(accum)
    else:
      emotivChannelNames = ch_names
      emotivChannelLocs = sens_loc

    grups = []
    res_matrix = np.zeros((4,4))

    n_nodes = len(ch_names[:-1])

    for i in range(n_nodes):
      for j in range(n_nodes):
        if i > j:
          section_i = assignedBrainSection(ch_names[i])
          section_j = assignedBrainSection(ch_names[j])
          if section_i != -1 and section_j != -1:         
            if section_i < section_j:
              res_matrix[section_i][section_j] += accum[i][j]
            else:
              res_matrix[section_j][section_i] += accum[i][j]

    #res_matrix = zScore(res_matrix)
    #res_matrix = normalize(res_matrix)


    fig, ax = plt.subplots()
    heatmap = ax.pcolor(res_matrix, cmap=plt.cm.Blues)
    plt.colorbar(heatmap)

    ax.set_xticks(np.arange(np.array(res_matrix).shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(np.array(res_matrix).shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()

    label = ['pt_izq','pt_der','frontal_der','frontal_izq']

    ax.set_xticklabels(label, minor=False)
    ax.set_yticklabels(label, minor=False)

    path = IMGsDirSource + eegType + 'brainSectionsConnectivity'
    plt.savefig(path + '.png') # guardar graficos en .png
  # plt.savefig(path + '.svg', format='svg', dpi=1200) # guardar graficos en .svg
    plt.clf() # clean buffer
    nparray = np.asarray(res_matrix)
    np.savetxt(path+".csv", nparray, delimiter=",")

if __name__ == "__main__":
  main(sys.argv[1:])