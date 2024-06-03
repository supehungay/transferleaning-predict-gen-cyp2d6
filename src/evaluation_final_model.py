from keras.models import load_model
import joblib
import numpy as np
import pandas as pd
import glob

class EvaluateFinalModel:
    
    def __init__(self, train_path, test_path, uncertain_path, folder_model) -> None:
        self.train_path = train_path
        self.test_path = test_path
        self.uncertain_path = uncertain_path
        self.folder_model = folder_model
        
        self.dataset_trainning = self.load_data(path_data=train_path)
        self.dataset_evalution = self.load_data(path_data=test_path)
        self.uncerted = self.load_data(uncertain_path, types='Uncerated')
        

    def load_data(self, path_data: str, types = None) -> tuple:
        X = None
        y = None
        data_loaded = joblib.load(path_data)
        
        if types == 'Uncerated':
            return data_loaded
            
            
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

    def predict(self, X):
        models = glob.glob(self.folder_model + "*.model.h5")
        predicts = [] 
        for m in models: 
            model = load_model(m)       
            predict = model.predict(X)
            predicts.append(predict)
            
        return np.array(predicts).mean(axis=0)

    def evaluation(self, pred_fuc, y_func):
        return np.sum(y_func == pred_fuc) / len(y_func)

    def get_functions(self, pred):
        cutpoint_1 = 0.743              
        cutpoint_2 = 0.77       

        cut1 = np.greater(pred[:, 0], [cutpoint_1])
        cut2 = np.greater(pred[:, 1], [cutpoint_2])
        functions = []
        for i in range(pred.shape[0]):
            if cut1[i] == True and cut2[i] == True:
                functions.append("Normal Function")
            elif cut1[i] == True and cut2[i] == False:
                functions.append("Decreased Function")
            else:
                functions.append("No Function")

        return np.array(functions)

    def pred_uncerted(self, X_uncerted):
        pred_label = self.get_functions(self.predict(X_uncerted))
        return pred_label
    
    def save_score_pred(self):
        pred_label_uncertain = self.pred_uncerted(self.uncerted['X'])
        
        data_frame_uncertain = pd.DataFrame({'Uncertain': np.array([f"*{uncurated.split('_')[1]}" for uncurated in self.uncerted['sample_names']]), 'Predict': pred_label_uncertain})
        # data_frame_uncertain.to_csv('../data/final_model/uncurated/predict.csv', index=False)
        alleles_function = pd.read_excel('../data/final_model/pcbi.1008399.s003.xlsx', usecols=[0, 1])
        
        for index, row in data_frame_uncertain.iterrows():
            condition = alleles_function['CYP2D6 Star Allele'] == row['Uncertain']
            alleles_function.loc[condition, 'Curated Function'] = row['Predict']

        function_to_score = alleles_function.copy()
        for index, row in function_to_score.iterrows():
            if row['Curated Function'] == 'Normal':
                function_to_score.iloc[index, 1] = 1
            elif row['Curated Function'] == 'Decreased Function':
                function_to_score.iloc[index, 1] = 0.5
            else:
                function_to_score.iloc[index, 1] = 0
        
        function_to_score = function_to_score.rename(columns={'Curated Function': 'Score'})
        
        gt_to_score = pd.read_csv('../data/final_model/CYP2D6_gt_to_score.txt', sep='\t')
        gt_to_score['reliability'] = '.'
        for index, row in gt_to_score.iterrows():
            if pd.isna(row['Activity Value']):
                condition = function_to_score['CYP2D6 Star Allele'] == row['Allele']
                score = function_to_score.loc[condition, 'Score'].values
                if len(score) > 0:
                    gt_to_score.iloc[index, 1] = score
                    gt_to_score.iloc[index, 3] = 'Predict by model'

        gt_to_score.to_csv('../data/final_model/uncerated/gt_to_score.txt', sep='\t', index=False, na_rep='N/A')
        
        print('saved "gt_to_score" ../data/final_model/uncerated/gt_to_score.txt')
        print('Done!')

def main():
    TRAIN_PATH = '../data/final_model/train/train_data.joblib'
    TEST_PATH = '../data/final_model/test/test_data.joblib'
    UNCERTAID_PATH = '../data/final_model/uncerated/uncerated.joblib'
    FOLDER_MODEL = '../model/FINAL_MODEL/03-06-2024_11-51/'
    
    evaluate = EvaluateFinalModel(train_path=TRAIN_PATH, test_path=TEST_PATH, uncertain_path=UNCERTAID_PATH, folder_model=FOLDER_MODEL)
    evaluate.save_score_pred()

if __name__ == '__main__': 
    main()