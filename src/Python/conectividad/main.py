import sys

from metricas import *
from repositorio import *

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset("tips")

def calculo(lista):
	n = len(lista)
	return str(len(list(filter(lambda x: x < (n-1)/2.0,lista))))

def main(argv):
	if argv[0] == 'base':
#		for metrica in [spearman]:

		for metricaName in ['pli']:
				metrica = comodinHalf(metricaName)
				for eeg in ['emotiv']:
					for name,f_min,f_max in [('alpha',8,14)]:
						setLoaderBase(metrica,f_min,f_max,eeg)

	else:
		for metricaName in ['imcoh','pli2_unbiased','pli','coh']:
				metrica = comodin(metricaName)
				for name,f_min,f_max in [('alpha',8,14)]:
					setLoader(metrica,f_min,f_max)

if __name__ == "__main__":
	main(sys.argv[1:])