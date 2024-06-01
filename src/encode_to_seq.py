import sys
import os
import numpy as np
import pandas as pd
import argparse

# This pre-processing was forked from the original author's published code
# Only main change is the encoding of diplotypes
# https://github.com/gregmcinnes/Hubble2D6/blob/master/bin

class Encode2Seq():
	def __init__(self, vcf, embedding_file, annotation_file, ref_seq, labels=None, verbose=False, label_cols=[0, 6]):
		self.vcf = vcf
		self.labels = labels
		self.embedding_file = embedding_file
		self.annotation_file = annotation_file
		self.ref_seq = ref_seq
		self.verbose = verbose
		self.label_cols = label_cols

		self.sample_names = None 
		self.X = None
		self.y = None

		self.embeddings = self.create_embedding_matrix()

		seqs = self.build_seqs()
		keys = list(seqs.keys())
		data = self.format_seqs(seqs)

		self.X = data["X"]
		self.y = data["y"]
		self.sample_names = np.array(data["sample_names"])

	def format_seqs(self, seq_data):
		sample_names = []
		seqs = []
		null_vector = 2048

		# Create a full array of encoded seqs for each haplotype per sample
		# Haplotypes per sample are seperated by 32-unit empty space
		for k in seq_data.keys():
			sample_names.append(k)
			seq = seq_data[k]
			null = [null_vector] * 32
			full_seq = seq[0].copy() + null + seq[1].copy()
			seqs.append(full_seq)

		# Turn vector of encodings into one-hot seq w/ annotations
		X_ind = np.array(seqs, dtype=int)
		X = self.indices2embeddings(X_ind)
		y = self.get_labels(sample_names)

		data = {
			"sample_names": sample_names, # Guaranteed to be order of X
			"X": X,
			"y": y
		}

		return data

	def get_labels(self, samples):
		if self.labels == None: return None

		y_df = pd.read_csv(self.labels, header=None, index_col=0, usecols=self.label_cols)
		try:
			y = y_df.loc[samples].values
		except KeyError:
			print("Mismatching labels in X and y files")
			exit(1)

		return y

	def create_embedding_matrix(self):
		headers = pd.read_csv(self.annotation_file, nrows=0)
		embedding_df = pd.read_csv(self.annotation_file, usecols=[h for h in headers.columns if h != 'key'], dtype=np.float32)
		return embedding_df.values

	def indices2embeddings(self, data):
		embeddings = np.apply_along_axis(self.embedding_mapper, axis=1, arr=data)
		return embeddings

	def embedding_mapper(self, x):
		return self.embeddings[x]

	# Heavily based on:
	# https://github.com/gregmcinnes/Hubble2D6/blob/master/bin/hubble.py#L77
	def build_seqs(self):
		samples = self.get_samples(self.vcf)
		sample_seqs = self.init_seq_data(samples)
		embeddings = self.precomputed_embeddings()
		previous = []
		total_variants, skipped = 0, 0

		with open(self.vcf) as f:
			for line in f:
				if line.startswith("#"):
					continue

				vcf_row = self.parse_vcf_line(line)
				total_variants += 1

				#! - Alt embedding is array of embeddings
				current_embeddings = self.get_embedding(embeddings, ref=vcf_row['ref_key'], alts=vcf_row['alt_key'])

				if current_embeddings is None:
					skipped += 1
					if self.verbose:
						print(vcf_row['ref_key'], vcf_row['alt_key'])

					# Skip variants with no encodings :(
					continue

				# For each row, build embeddings of each sample entry in row
				for i in range(len(samples)):
					s = samples[i]
					gt = vcf_row['diplotypes'][i]
					h1, h2 = [int(i) for i in gt.split('/')]

					h1_position_idx, h1_embedding_idx = current_embeddings[h1]['position_index'], current_embeddings[h1]['embedding_index']
					h2_position_idx, h2_embedding_idx = current_embeddings[h2]['position_index'], current_embeddings[h2]['embedding_index']

					sample_seqs[s][0][h1_position_idx] = h1_embedding_idx
					sample_seqs[s][1][h2_position_idx] = h2_embedding_idx

		if self.verbose:
			print("Skipped %d of %d total variants" % (skipped, total_variants))

		return sample_seqs

	def get_embedding(self, embeddings, ref, alts):
		keys = [ref] + alts
		try:
			if not all (k in embeddings for k in keys):
				raise LookupError(keys)

			return [embeddings[k] for k in keys]
		except LookupError as err:
			return None

	# https://github.com/gregmcinnes/Hubble2D6/blob/master/bin/hubble.py#L188
	def precomputed_embeddings(self):
		wd = sys.path[0]
		_file = os.path.join(wd, self.embedding_file)

		embeddings = {}
		with open(_file) as f:
			for line in f:
				fields = line.rstrip().split()
				key = "%s_%s" % (fields[1], fields[2])
				embeddings[key] = {"position": int(fields[1]),
													 "allele": fields[2],
													 "position_index": int(fields[3]),
													 "embedding_index": int(fields[4])}

		return embeddings

	def parse_vcf_line(self, line):
		fields = line.rstrip().split()
		row = {
			'position': int(fields[1]),
			'ref': fields[3],
			'alt': fields[4], 
			'ref_key': '%s_%s' % (fields[1], fields[3]),
			'alt_key': ['%s_%s' % (fields[1], f) for f in fields[4].split(',')],
			'diplotypes': fields[9:]
		}

		return row

	def get_reference_seq(self):
		ref = None
		# Convert embedding indices of reference gene into list
		with open(self.ref_seq) as f:
			ref = f.readline().rstrip().split()[1:][0].split(',')

		return ref

	def init_seq_data(self, samples):
		sample_seqs = {}
		reference = self.get_reference_seq()

		# There could be a more efficient method than storing two copies of the reference gene for each ht
		# Maybe only store changes?
		for s in samples:
			sample_seqs[s] = [reference.copy(), reference.copy()]

		return sample_seqs

	def get_samples(self, vcf):
		samples = []

		with open(vcf) as f:
			for line in f:
				if line.startswith("#CHROM"):
					samples = line.rstrip().split()[9:]
					break
			
		return samples

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-v', '--vcf', help="VCF file to convert")
	parser.add_argument('-l', '--labels', help="CSV file containing labels to samples")
	options = parser.parse_args()

	embedding = Encode2Seq(vcf=options.vcf, labels=options.labels, embedding_file='./data/embeddings.txt', annotation_file='./data/gvcf2seq.annotation_embeddings.csv', ref_seq='./data/ref.seq', verbose=True, label_cols=[0, 1, 2])
	print("Embeddings...")
	np.set_printoptions(threshold=np.inf)
	print(embedding.sample_names.shape)
	print(embedding.X.shape, embedding.y.shape)
