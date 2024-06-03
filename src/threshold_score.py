import numpy as np
import pandas as pd
from keras.models import load_model
import glob
import joblib


class ThresholdScore:
    def __init__(self, train_path, folder_model) -> None:
        self.train_path = train_path
        self.folder_model = folder_model
        
        self.dataset_trainning = self.load_data(path_data=self.train_path)
        self.X = self.dataset_trainning[0][0]
        self.y = self.dataset_trainning[0][1]
        self.probs = self.predict_probs(self.X)
        
        self.save2csv(probs=self.probs, y=self.y)
    
    
    def load_data(self, path_data: str) -> tuple:
        X = None
        y = None
        data_loaded = joblib.load(path_data)
        X = data_loaded['X']
        y = data_loaded['y']
        
        all_stars = np.array([s.split('_')[1] for s in data_loaded['sample_names']]) # lấy ra star alen: 10, 1, 2, ...
        stars, idx = np.unique(all_stars, return_index=True)
        sample_mask = np.isin(all_stars, all_stars[idx]) # Đánh dấu những star allele của idx trong all_stars
        stars_001 = np.array([s for s in data_loaded['sample_names'][sample_mask] if s.split('_')[-1] == '001']) 
        mark_001 = np.isin(data_loaded['sample_names'], stars_001)
        X_001, y_001 = X[mark_001], y[mark_001]
        
        dataset = [(X, y), (X_001, y_001)]
        
        return dataset
    
    def predict_probs(self, X):
        models = glob.glob(self.folder_model + "*.model.h5")
        predicts = [] 
        for m in models: 
            model = load_model(m)       
            predict = model.predict(X)
            predicts.append(predict)
            
        return np.array(predicts).mean(axis=0)

    def save2csv(self, probs, y):
        match = np.concatenate((y, probs), axis=1)
        df = pd.DataFrame(match, columns=['origin_label_1', 'origin_label_2', 'No.Function', 'Normal.Function'])
        df['origin_label_1'] = df['origin_label_1'].apply(lambda x: 'Y' if x == 1.0 else 'N')
        df['origin_label_2'] = df['origin_label_2'].apply(lambda x: 'Y' if x == 1.0 else 'N')
        
        df.to_csv('../data/final_model/probs_getThreshold.csv', index=False)
        print('saved probs score to ../data/final_model/probs_getThreshold.csv')
        print('Done!')

def main():
    TRAIN_PATH = '../data/final_model/train/train_data.joblib'
    FOLDER_MODEL = '../model/FINAL_MODEL/03-06-2024_11-51/'
    
    score = ThresholdScore(train_path=TRAIN_PATH, folder_model=FOLDER_MODEL)
    
if __name__ == '__main__':
    main()