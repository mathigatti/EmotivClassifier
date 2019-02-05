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


def main(argv):
  emotivChannelLocs = []
  emotivChannelNames = []
  for experimentType in ['Open','Closed']:
    for eegType in ['emotiv','biosemi']:
      setFiles = find_filenames('.set',DatasetsDirSource+experimentType+'/'+eegType)
      accum = []
      for setFile in setFiles:
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
        fmin, fmax = 2., 24.
        tmin = 0.0  # exclude the baseline period

        con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
            epochs, method='pli', mode='multitaper', sfreq=sfreq, fmin=fmin, fmax=fmax,
            faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1)

        # the epochs contain an EOG channel, which we remove now
        ch_names = epochs.ch_names
        idx = [ch_names.index(name) for name in ch_names if name != 'STI 014']
        con = con[idx][:, idx]

        # con is a 3D array where the last dimension is size one since we averaged
        # over frequencies in a single band. Here we make it 2D
        con = np.array(con[:, :, 0])

        con = zScore(con)

        if accum == []:
          accum = np.zeros(np.shape(con))
        accum += con

      # Plot the sensor locations
      sens_loc = [raw.info['chs'][i]['loc'][:3] for i in idx]

      sens_loc = np.array(sens_loc)
      mean_con = accum

      if(eegType == 'biosemi'):
        sens_loc = emotivChannelLocs
        ch_names = emotivChannelNames
        fixElectrodes(mean_con)
      else:
        emotivChannelNames = ch_names
        emotivChannelLocs = sens_loc


      print(sens_loc)
      print(ch_names)
      # Get the strongest connections
      #n_con = 30  # number of connections to show up
      #threshold = np.sort(mean_con, axis=None)[-n_con]

      G=nx.Graph()
      Xn=sens_loc[:, 0]
      Yn=sens_loc[:, 1]

      labels = {}
      for i in range(len(ch_names[:-1])):
        G.add_node(i,pos=(Xn[i],Yn[i]))
        labels[i] = ch_names[:-1][i]
      
      n_con = 20  # number of connections to show up
      threshold = np.sort(mean_con, axis=None)[-n_con]
      ii, jj = np.where(mean_con > threshold)

      # Remove close connections
      min_dist = 0.0  # exclude sensors that are less than 5cm apart
      edge_labels = {}
      for i, j in zip(ii, jj):
          if i > j:
            distance = linalg.norm(sens_loc[i] - sens_loc[j])
            #print distance
            if distance > min_dist:
              G.add_edge(i,j,color=abs(accum[i, j]),weight=abs(accum[i, j]))
              edge_labels[(i,j)] = round(accum[i, j],2)

      pos=nx.get_node_attributes(G,'pos')

      edges = G.edges()
      colors = [G[u][v]['color'] for u,v in edges]
      weights = [G[u][v]['weight'] for u,v in edges]

      nx.draw(G,pos,node_color='#A0CBE2',edge_color=colors, width=weights,edge_cmap=plt.cm.Blues,with_labels=False)
      nx.draw_networkx_labels(G,pos, labels)
      nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)
      path = IMGsDirSource + eegType + experimentType +'_mean_'+ str(n_con) + '.png'
      plt.savefig(path) # save as png
      plt.clf() # clean buffer


if __name__ == "__main__":
  main(sys.argv[1:])