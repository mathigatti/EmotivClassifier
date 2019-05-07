import sys

from metrics import metricsDict
from repositorio import computeCorrelationMatrix

def main(argv):
	res = ''

	inputDir = argv[1]
	outputDir = argv[2]
	try:
		metric = metricsDict[argv[0]]
	except:
		print("Invalid Name:")
		print("Valid options are: 'spearman', 'correlation','pli','plv' y 'coh'")

	for band,f_min,f_max in [('theta',4,7.5),('alpha',8,14),('beta',15,25)]:
		computeCorrelationMatrix(metric,band,f_min,f_max,inputDir,outputDir)

if __name__ == "__main__":
	main(sys.argv[1:])