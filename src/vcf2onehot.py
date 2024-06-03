import numpy as np
import pandas as pd
import os
import sys
import glob
import joblib

class VCF2Onehot:
    def __init__(self, data_path = None, label_path = None) -> None:
        self.data_path = data_path
        self.label_path = label_path
    
    # Detect if a file is gzipped before opening
    def g_open(self, file):
        if file.endswith('gz'):
            import gzip
            return gzip.open(file)
        return open(file)

    def byte_decoder(self, a):
        return a.decode("utf-8")

    def get_vcf_subject_ids(self, vcf):
        ids = []
        with self.g_open(vcf) as f:
            for line in f:
                try:
                    line = self.byte_decoder(line)
                except:
                    line = line
                if line.startswith("#CHROM"):
                    fields = line.rstrip().split()
                    ids = fields[9:]
                    break
        return ids

    def reference_seq(self):
        wd = sys.path[0]
        file = os.path.join(wd, "../data/ref.seq")

        with open(file) as f:
            for line in f:
                fields = line.rstrip().split()
                seq = [int(x) for x in fields[1].split(',')]
        return seq

    def sample_seq_init(self, samples):
        ref_seq = self.reference_seq() # lay seq tham chieu; length = 7418
        sample_dict = {}
        
        for s in samples:
            sample_dict[s] = [ref_seq.copy(), ref_seq.copy()]
        return sample_dict

    def precomputed_embeddings(self):
        wd = sys.path[0]
        file = os.path.join(wd, "../data/embeddings.txt")

        embeddings = {}
        with open(file) as f:
            for line in f:
                fields = line.rstrip().split()
                key = "%s_%s" % (fields[1], fields[2])
                embeddings[key] = {"position": int(fields[1]),
                                    "allele": fields[2],
                                    "position_index": int(fields[3]),
                                    "embedding_index": int(fields[4])}
        return embeddings


    #todo get subjects and make a dictionary
    def parse_vcf_line(self, line, annovar=False):
        CHROM = 0 # Nhiễm sắc thể
        POS = 1 # Vị trí trên Gen 
        ID = 2 # ID cho mỗi biến thể thể gen (nếu có sẵn)
        REF = 3 # Gen tham chiếu đến
        ALT = 4 # Biến thể gen
        QUAL = 5 # Chất lượng biến đổi gen
        FILTER = 6 # Kiểm định sự biến đổi gen có đáng tin cậy hay không
        INFO = 7 # thông tin thêm về biến đổi gen
        FORMAT = 8 # định dạng của dữ liệu gen cho từng mẫu
        CALLS = 9 # tù 9 trở đi là gentype của các biến thể

        class VCFfields(object):
            def __init__(self, line):
                self.fields = line.rstrip().split() # tách các trường dũ liệu
                self.chrom = None
                self.pos = None
                self.id = None
                self.ref = None
                self.alt = None
                self.qual = None
                self.filter = None
                self.info = None
                self.format = None
                self.calls = []
                
                self.run()
                
                    
            def run(self):
                self.chrom = self.fields[CHROM]
                self.pos = int(self.fields[POS])
                self.id = self.fields[ID]
                self.ref = self.fields[REF].upper()
                self.alt = self.fields[ALT].upper().split(',')
                self.qual = self.fields[QUAL]
                self.filter = self.fields[FILTER]
                self.info = self.fields[INFO]
                self.format = self.fields[FORMAT]
                
                if len(self.fields) > 9:
                    self.calls = self.fields[CALLS:]

            def print_row(self, chr=True, summary=False):
                if summary is True:
                    print("%s\t%s\t%s\t%s\t%s" % (self.chrom, self.pos, self.id, self.ref, self.alt))
                else:
                    print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (self.chrom, self.pos, self.id, self.ref, self.alt,
                                                                self.qual, self.filter, self.info, self.format, self.calls))
            
        row = VCFfields(line)

        return row

    def get_embedding(self, embeddings, key):
        if key in embeddings:
            return embeddings[key]
        else:
            return None

    def build_seqs(self, vcf, debug=False):
        # Build a dictionary of seqs for each sample using the reference
        samples = self.get_vcf_subject_ids(vcf) # tên các biến thể
        sample_seqs = self.sample_seq_init(samples) #  tao dict voi: key=bien the; value=[ref_seq_1, ref_seq_2]       
        embeddings = self.precomputed_embeddings()

        # Read in file variant by variant
        with open(vcf) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                vcf_row = self.parse_vcf_line(line)
                
                # Get the embedding index for the current variant
                ref_key = "%s_%s" % (vcf_row.pos, vcf_row.ref)
                alt_key = ["%s_%s" % (vcf_row.pos, alt) for alt in vcf_row.alt]

                # return
                ref_embedding = self.get_embedding(embeddings, ref_key)
                alt_embedding = [self.get_embedding(embeddings, k) for k in alt_key]
                ref_alt_embedding =  [ref_embedding] + alt_embedding
                
                if ref_embedding is None or not all(alt_embedding):
                    if debug:
                        print("Variant does not have a precomputed embedding!")
                        vcf_row.print_row(summary=True)
                    continue

                # Update with the appropriate embedding index
                for i in range(len(samples)):
                    s = samples[i]
                    gt = vcf_row.calls[i]
                    
                    hap1, hap2 = [int(i) for i in gt.split('/')]
                    # print(hap1, hap2)
                    
                    position_index_hap1, embedding_index_hap1 = ref_alt_embedding[hap1]['position_index'], ref_alt_embedding[hap1]['embedding_index']
                    position_index_hap2, embedding_index_hap2 = ref_alt_embedding[hap2]['position_index'], ref_alt_embedding[hap2]['embedding_index']
                    
                    sample_seqs[s][0][position_index_hap1] = embedding_index_hap1
                    sample_seqs[s][1][position_index_hap2] = embedding_index_hap2
                    
        return sample_seqs

    # Load the variant embeddings from file
    def create_embedding_matrix(self):
        wd = sys.path[0]
        file = os.path.join(wd, "../data/gvcf2seq.annotation_embeddings.csv")
        embeddings_df = pd.read_csv(file)
        embedding_matrix = embeddings_df.loc[:, embeddings_df.columns != 'key'].values.astype(np.float64)
        return embedding_matrix


    # Transform the embedding indices to the actual embeddings
    def indices2embeddings(self, data):
        embeddings = np.apply_along_axis(self.embedding_mapper, axis=1, arr=data)
        return embeddings 

    def embedding_mapper(self, x):
        embeddings = self.create_embedding_matrix()
        return embeddings[x]

    def format_seqs(self, seq_data): # đầu vào là output của build_seqs(vcf)
        sample_names = []
        seqs = []

        # This value is the index for the null vector, i.e. no annotations at all
        null_vector = 2048 # vị trí có các giá trị cột đều bằng 0 trong file gvcf2seq_annotation_embedding.csv

        for k in seq_data.keys():
            sample_names.append(k) # ten các bien the
            seq = seq_data[k] # seq cua bien the tuong ung
            null = [null_vector] * 32
            full_seq = seq[0].copy() + null + seq[1].copy()
            seqs.append(full_seq)

        # Convert the indices to embeddings
        X_ind = np.array(seqs, dtype=int)
        X = self.indices2embeddings(X_ind)

        # Put the data in a dictionary and return
        data = {
            "X": X,
            "sample_names": np.array(sample_names)
        }

        return data
    
#     def save_data_processed(self):
#         batch_label = sorted(glob.glob(self.label_path))
#         batch_data = sorted(glob.glob(self.data_path))
        
#         n_batch = 12
#         n_batch_train = 10
        
#         print(f'create {n_batch} batch: {n_batch_train} batch train and {n_batch - n_batch_train} batch test')
        
#         for i in range(n_batch):
#             activate_score = []
#             # Generate the seq data
#             vcf = batch_data[i]
#             # print(vcf)
#             seqs = self.build_seqs(vcf)
#             # Create the data object
#             data = self.format_seqs(seqs)

#             with open(f'{batch_label[i]}') as f:
#                 for line in f:
#                     activate_score.append(line.split(',')[-2])
            
#             data["activate_score"] = np.array(activate_score, dtype=np.float64)
            
#             file_name = vcf.split('\\')[-1].split('.')[0]
            
#             if i < n_batch_train:
#                 joblib.dump(data, f'../data/pretrained_model/train/{file_name}.joblib')
#                 print(f"save train batch {i + 1}...")
#             else:
#                 joblib.dump(data, f'../data/pretrained_model/test/{file_name}.joblib')
#                 print(f'save test batch {(i - n_batch_train) + 1}...')
            
#         print("Done!")
        
# def main():
#     data_path = '../data/simulated_cyp2d6_diplotypes/*.vcf'
#     label_path = '../data/simulated_cyp2d6_diplotypes/*.csv'
    
#     data = VCF2Onehot(data_path=data_path, label_path=label_path)
#     data.save_data_processed()

# if __name__=='__main__':
#     main()


