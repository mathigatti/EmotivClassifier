# Code inspired from:
# https://www.martinos.org/mne/stable/auto_examples/connectivity/plot_sensor_connectivity.html#sphx-glr-auto-examples-connectivity-plot-sensor-connectivity-py

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

  t_events = np.arange(raw.first_samp + 1000, raw.last_samp - 1000, 1000)
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

    G=nx.Graph()
    Xn=sens_loc[:, 0]
    Yn=sens_loc[:, 1]

    labels = {}
    for i in range(len(ch_names[:-1])):
      G.add_node(i,pos=(Xn[i],Yn[i]))
      labels[i] = ch_names[:-1][i]

    accum = accum/len(lista)

    temp = abs(accum)
    n_con = 20  # number of connections to show up
    threshold = np.sort(temp, axis=None)[-n_con]

    ii, jj = np.where(abs(accum) > threshold)

    edge_labels = {}
    min_dist = 0.6  # exclude sensors that are less than 5cm apart
    for i, j in zip(ii, jj):
        if i > j:
            distance = linalg.norm(sens_loc[i] - sens_loc[j])
            #print distance
            if distance > min_dist:
              G.add_edge(i,j,color=abs(accum[i, j]),weight=abs(accum[i, j])*5)
              edge_labels[(i,j)] = round(accum[i, j],2)

    pos=nx.get_node_attributes(G,'pos')

    edges = G.edges()
    colors = [G[u][v]['color'] for u,v in edges]
    weights = [G[u][v]['weight'] for u,v in edges]

    nx.draw(G,pos,node_color='#A0CBE2',edge_color=colors, width=weights,edge_cmap=plt.cm.Blues,with_labels=False)
    nx.draw_networkx_labels(G,pos, labels)
    nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)
    path = IMGsDirSource + eegType +'_diff_'+ str(n_con) + '.png'
    plt.savefig(path) # save as png
    plt.clf() # clean buffer

if __name__ == "__main__":
  main(sys.argv[1:])