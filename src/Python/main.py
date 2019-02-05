import sys

from metricas import *
from repositorio import *
from avgPath import *

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
	res = ''

	if argv[0] == 'pathTest':
		findAvgPaths()

	elif argv[0] == 'spectrum':
		spectrumsCorrelation()
	elif argv[0] == 'base':

		# if not argv[1] or argv[1] != 'N':
		# 	setLoader(metrica,0)
		# matches, mean, median = findMatchs(metrica.name)
		# print mean
		# print median
		# print '\n\n'
		# print matches

		#Correr con: python main.py pli 8
		for metrica in [correlation,spearman]:

#		for metricaName in ['plv','imcoh','ppc','pli2_unbiased','wpli2_debiased','pli','coh']:
#				metrica = comodinHalf(metricaName)
				for eeg in ['emotiv','biosemi']:
					allMatches = []
					bands = []
					logMatches = []

					toJsons = []
					for name,f_min,f_max in [('theta',4,7.5),('alpha',8,14),('beta',15,25)]:
						setLoaderBase(metrica,f_min,f_max,eeg)
						matches, mean, median, stdev = findMatchsBase(metrica.name,eeg)
						allMatches+=list(map(lambda x:x[1],matches))
						toJsons.append([name,list(map(lambda x:x[1],matches)),mean,stdev,median])
						bands += [name]*len(matches)
						logMatches.append(matches)

					for toJson in toJsons:
						res += eeg + " | " + toJson[0] + " | " + metrica.name + " | " + str(toJson[2]) + " +- " + str(toJson[3]) + " | " + calculo(toJson[1]) +"/"+str(len(toJson[1]))+ "\n"

					print(allMatches)
					print(bands)
					df = pd.DataFrame.from_dict({'band':bands, 'match':allMatches})

					fig, ax = plt.subplots()

					ax = sns.boxplot(x="band", y="match", data=df)
					ax = sns.swarmplot(x="band", y="match", data=df, color=".25")
					ax.axhline(6.5, color="red")

					plt.savefig('boxplots/'+metrica.name+"_" +eeg+"_base.pdf")

					bandNames = ['theta','alpha','beta']
					with open('boxplots/' + metrica.name+"_" +eeg+"_base.txt",'w') as f:
						i = 0
						for match in logMatches:
							f.write(bandNames[i]+'\n')
							for value in match:
								f.write(str(value)+"\n")
							i+=1
							f.write("\n")


	else:
		#metrica = metricasDict[argv[0]]
		# if not argv[1] or argv[1] != 'N':
		# 	setLoader(metrica,0)
		# matches, mean, median = findMatchs(metrica.name)
		# print mean
		# print median
		# print '\n\n'
		# print matches

		#Correr con: python main.py pli 8
#		for metrica in [correlation,spearman]:
		for metricaName in ['plv','imcoh','ppc','pli2_unbiased','wpli2_debiased','pli','coh']:
				metrica = comodin(metricaName)
				allMatches = []
				bands = []
				logMatches = []

				toJsons = []
#				for name,f_min,f_max in [('beta',15,25)]:
				for name,f_min,f_max in [('theta',4,7.5),('alpha',8,14),('beta',15,25)]:
					setLoader(metrica,f_min,f_max)
					matches, mean, median, stdev = findMatchs(metrica.name)
					allMatches+=list(map(lambda x:x[1],matches))
					toJsons.append([name,list(map(lambda x:x[1],matches)),mean,stdev,median])

					bands += [name]*len(matches)
					logMatches.append(matches)

				for toJson in toJsons:
					res += toJson[0] + " | " + metrica.name + " | " + str(toJson[2]) + " +- " + str(toJson[3]) + " | " + calculo(toJson[1]) + "/" + str(len(toJson[1])) + "\n"
				'''
				print(allMatches)
				print(bands)
				df = pd.DataFrame.from_dict({'band':bands, 'match':allMatches})

				fig, ax = plt.subplots()

				ax = sns.boxplot(x="band", y="match", data=df)
				ax = sns.swarmplot(x="band", y="match", data=df, color=".25")
				ax.axhline(6.5, color="red")

				plt.savefig('boxplots/'+metrica.name+".pdf")

				bandNames = ['theta','alpha','beta']
				with open('boxplots/' + metrica.name+ ".txt",'w') as f:
					i = 0
					for match in logMatches:
						f.write(bandNames[i]+'\n')
						for value in match:
							f.write(str(value)+"\n")
						i+=1
						f.write("\n")
'''
		# ax = plt.subplot(111)
		# ax.set_xlim(0,24)
		# ax.set_title('Match Score over different Frecuency Spectrums')
		# plt.errorbar(fs_min, means, stdevs, linestyle='None', marker='^')
		# path = IMGsDir + '/' + "bandas"
		# plt.savefig(path + '.png') # guardar graficos en .png
		# plt.clf() # clean buffer

		# result = []
		# for i in range(5):
		# 	setLoader(metrica,float(i*4))
		# 	matches, mean, median = findMatchs(metrica.name)
		# 	result.append(mean)
		# print result

		# if(not argv[1] or argv[1] != 'N'):
		# 	for match in matches:
		# 		matchString = 'Los resultados para ' + match[0] + ' son:\n'
		# 		for i in range(len(match[1])):
		# 			matchString += '\tResultado ' + str(i+1) + ': ' + str(match[1][i]) + '\n'
		# 		print matchString
	print(res)

if __name__ == "__main__":
	main(sys.argv[1:])