import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torch
from sklearn.preprocessing import normalize

class MovielensDataset(Dataset):
    def __init__(self, ratings, item_path,data_matrix, device):
        self.item_path = item_path
        self.audio_embeddings = pd.read_csv(item_path + "Compressed/audio.csv").to_numpy()
        self.meta_embeddings = pd.read_csv(item_path + "Meta/embeddings.csv").to_numpy()
        self.text_embeddings = pd.read_csv(item_path + "Text/embeddings.csv").to_numpy()
        print("Audio Embeddings Shape:", self.audio_embeddings.shape)
        print("Meta Embeddings Shape:", self.meta_embeddings.shape)
        print("Text Embeddings Shape:", self.text_embeddings.shape)
        self.ratings = pd.read_csv(ratings, sep='\t', 
                                   names=['user_id', 'movie_id', 'rating', 'unix_timestamp'],encoding='latin-1')
        print("Rating Embeddings Shape:", self.ratings.shape)
        self.indices = None
        self.device = device
        self.data = None
        self.n_users = None
        self.n_items = None
        self.data_matrix = data_matrix
        self.fill_ratings()
        self.embeddings()
    
    def fill_ratings(self, threshold=4):
        self.n_users = self.ratings.user_id.unique().shape[0]
        self.n_items = self.ratings.movie_id.unique().shape[0]
        
        self.data = np.zeros((943, 1682))
        for line in self.ratings.itertuples():
            self.data[line[1]-1, line[2]-1] = line[3]
        
        self.data_emp = np.where(np.logical_and(self.data > 3,
                            np.random.random_sample(self.data.shape) <= 0.2), 1, 0)
        self.indices = list(zip(*(np.where(self.data != 0))))
        
    def embeddings(self):
        print(self.video_embeddings.shape)
        self.audio_embeddings = np.nan_to_num(self.audio_embeddings, nan=0.0)
        self.audio_embeddings = normalize(self.audio_embeddings, axis = 0)
        self.user_embeddings = np.divide(np.dot(self.data_emp, self.meta_embeddings), 
                                         self.data_emp.sum(axis = 1)[:, None] + 0.001)
        #self.user_embeddings = data_matrix
        self.item_embeddings = self.data_matrix.T
        self.audio_embedding_size = self.audio_embeddings.shape[1]
        self.text_embedding_size = self.text_embeddings.shape[1]
        self.user_embedding_size = self.user_embeddings.shape[1]
        self.item_embedding_size = self.item_embeddings.shape[1]
        self.meta_embedding_size = self.meta_embeddings.shape[1]
        
    def __len__(self):
        return(len(self.indices))
    
    def __getitem__(self, idx):
        user = self.indices[idx][0]
        item = self.indices[idx][1]
        
        #xu = self.user_embeddings(torch.LongTensor([user])).squeeze().to(self.device)
        xu = torch.from_numpy(self.user_embeddings[user]).to(self.device)
        xa = torch.from_numpy(self.audio_embeddings[item]).to(self.device)
        xt = torch.from_numpy(self.text_embeddings[item]).to(self.device)
        xi = torch.from_numpy(self.item_embeddings[item]).to(self.device)
        xm = torch.from_numpy(self.meta_embeddings[item]).to(self.device)
        
        y = self.data[user][item]
        return(xu.float(), [xa.float(), xt.float(), xi.float(), xm.float()], int(y))
