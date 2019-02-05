
# Directories

BASE_PATH = '/home/mathi/Desktop/EmotivClassifier/'

IMGsDir = BASE_PATH + 'DatosProcesados/'
DatasetsDir = BASE_PATH + 'DatosCrudos/'

# Metrics

metricasMatrix = ["pli", "plv", "coh",'cohy','imcoh','ppc','pli2_unbiased','wpli2_debiased']

# Electrodes

biosemiLabels = ['C26', 'D6', 'C32', 'D11', 'D23', 'D31', 'A15', 'A28', 'B11', 'B26', 'B30', 'C10', 'C6', 'C13']
emotivLabels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

biosemi2emotiv = {0:8, 1:10, 2:9, 3:11, 4:12, 5:13, 6:0, 7:1, 8:2, 9:3, 10:4, 11:6, 12:5, 13:7}