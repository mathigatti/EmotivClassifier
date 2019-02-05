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


def findMatrixFiles(eeg,metricName):
  return list(map(lambda x : IMGsDir + eeg + '/' + metricName + "/" + x, find_filenames(metricName+'.csv', IMGsDir+eeg+'/'+metricName)))

def main(argv):

  sens_loc = [[ -3.75594867e-01,  8.84846057e-01,  2.75637356e-01],
 [ -8.08524163e-01,  5.87427189e-01, -3.48994967e-02],
 [ -5.45007446e-01,  6.73028145e-01,  5.00000000e-01],
 [ -8.82717635e-01,  3.38843553e-01,  3.25568154e-01],
 [ -9.99390827e-01,  6.11950389e-17, -3.48994967e-02],
 [ -8.08524163e-01, -5.87427189e-01, -3.48994967e-02],
 [ -3.08828750e-01, -9.50477158e-01, -3.48994967e-02],
 [  3.08828750e-01, -9.50477158e-01, -3.48994967e-02],
 [  8.08524163e-01, -5.87427189e-01, -3.48994967e-02],
 [  9.99390827e-01,  6.11950389e-17, -3.48994967e-02],
 [  8.82717635e-01,  3.38843553e-01,  3.25568154e-01],
 [  5.45007446e-01,  6.73028145e-01,  5.00000000e-01],
 [  8.08524163e-01,  5.87427189e-01, -3.48994967e-02],
 [  3.75594867e-01,  8.84846057e-01,  2.75637356e-01]]

  ch_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4', 'STI 014']

  emotivChannelLocs = []
  emotivChannelNames = []

  eegType = argv[0]
  metricName = argv[1]
  sujeto = argv[2]

  setFiles = findMatrixFiles(eegType,metricName)

  setFiles = list(filter(lambda x: sujeto + "_" in x, setFiles))

  accum = []
  for setFile in setFiles:
    con = np.loadtxt(open(setFile, "rb"), delimiter=",")
    print(con)
    if accum == []:
      accum = np.zeros(np.shape(con))
    accum += con

  # Plot the sensor locations
  sens_loc = np.array(sens_loc)
  mean_con = accum

#  if(eegType == 'biosemi'):
#    fixElectrodes(mean_con)

  G=nx.Graph()
  Xn=sens_loc[:, 0]
  Yn=sens_loc[:, 1]

  labels = {}
  for i in range(len(ch_names[:-1])):
    G.add_node(i,pos=(Xn[i],Yn[i]))
    labels[i] = ch_names[:-1][i]
  
  n_con = 40  # number of connections to show up

  print(mean_con)

  threshold = np.sort(mean_con, axis=None)[-n_con]
  ii, jj = np.where(mean_con > threshold)

  # Remove close connections
  min_dist = 0.0  # exclude sensors that are less than 5cm apart
  edge_labels = {}
  for i, j in zip(ii, jj):
    distance = linalg.norm(sens_loc[i] - sens_loc[j])
    if distance > min_dist:
      G.add_edge(i,j,color=abs(accum[i, j]),weight=abs(accum[i, j]))
      edge_labels[(i,j)] = round(accum[i, j],2)

  pos=nx.get_node_attributes(G,'pos')

  edges = G.edges()
  colors = [G[u][v]['color'] for u,v in edges]
  weights = [G[u][v]['weight'] for u,v in edges]

  nx.draw(G,pos,node_color='#A0CBE2',edge_color=colors, width=list(map(lambda x:x*10,weights)),edge_cmap=plt.cm.Blues,with_labels=False)
  nx.draw_networkx_labels(G,pos, labels)
  nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)
  path = eegType + "_" + experimentType.lower() +"_" + metricName  +"_" + sujeto +'_mean_'+ str(n_con) + '.png'
  plt.savefig(path) # save as png
  plt.clf() # clean buffer


if __name__ == "__main__":
  main(sys.argv[1:])