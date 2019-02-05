import sys

from metricas import *
from repositorio import *

valores_de_referencia = [('theta',4,7.5),('alpha',8,14),('beta',15,25)]

def main(argv):

	if len(argv) > 2:
		fmin = argv[1]
		fmax = argv[2]
	else:
		fmin = 8
		fmax = 14

	experimentType = 'ChicosOpenEmotiv'

	for metrica in [correlation,spearman]:
		setEEGLOader(experimentType, 'emotiv',metrica,fmin,fmax)

	for metricaName in ['plv','ppc','coh']:
		metrica = comodin(metricaName)
		setEEGLOader(experimentType, 'emotiv',metrica,fmin,fmax)

if __name__ == "__main__":
	main(sys.argv[1:])