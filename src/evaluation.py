from keras.models import load_model
import joblib
import numpy as np
from keras.utils import to_categorical
import sys
sys.path.append('../src/')
import vcf2onehot

def label2onehot(label: np.array) -> np.array:
    thresholds = [0.25, 0.75, 1.25, 1.75]

    categorical_labels = np.digitize(label, thresholds)

    one_hot_encoder = to_categorical(categorical_labels)
    
    return one_hot_encoder

def get_model():
	# model = load_model('../save_model/ModelCheckPoint/final_25-03-2024_02-32/model.088-0.0545-0.9746.h5')
	model = load_model('../save_model/FinalModel/final_25-03-2024_02-32/model.h5')
 
	return model

def get_data(vcf_file=None, seq_file=None, joblib_file=None):
	if joblib_file is not None:
		try:
			data = joblib.load(joblib_file)
			X = data['X']
			y = label2onehot(data['activate_score'])
			return X
		except FileNotFoundError:
			print(f"File {joblib_file} not found.")
			return None, None
	elif seq_file is not None:
		try:
			sample_seq = {}
			with open(seq_file, 'r') as f:
				for line in f:
					fields = line.split()
					seqs_hap1 = [int(x) for x in fields[1].split(',')]
					seqs_hap2 = [int(x) for x in fields[2].split(',')]
				
					sample_seq[fields[0]] = [seqs_hap1, seqs_hap2]
			print(sample_seq.keys())
			# seqs = content.split() 
	
			# seq_data = {seqs[0]: [[seqs[1]], [seqs[2]]]}
			# print(np.array(seq_data[seqs[0]]))
			return vcf2onehot.format_seqs(sample_seq)['X']
				
		except FileNotFoundError:
			print(f"File {seq_file} not found.")
	elif vcf_file is not None:
		try:
			seqs = vcf2onehot.build_seqs(vcf_file)
			return vcf2onehot.format_seqs(seqs)['X']
		except FileNotFoundError:
			print(f"File {seq_file} not found.")
		
def get_score(predictions: np.array):
    dict_class = {0: 0, 1: 0.5, 2: 1, 3: 1.5, 4: 2}
    
    mapped_results = [dict_class[pred] for pred in predictions]
    return mapped_results

def main():
    model = get_model()
    X = get_data(seq_file='../data/PRJEB19931.seq')
    pred = model.predict(X)
    
    predict_class = np.argmax(pred, axis=1)
    
    score = get_score(predict_class)
    
    print(score)
    
if __name__ == '__main__': 
    main()