#from models.clip_vit import CLIPViTEncoder
#from models.sentence_bert import SentenceBERTEncoder
from models.Babel import BabelNet
from utils.data_loader import MovielensDataset
import torch
import configs 
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report

def main():
    args = configs.get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    items = pd.read_csv("/scratch/zz4330/ml-100k/Text/items.csv")
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('/scratch/zz4330/ml-100k/Text/u1.base', sep='\t', names=r_cols,encoding='latin-1')
    print(ratings.head())
    data_matrix = np.zeros((943,1682))
    for line in ratings.itertuples():
        data_matrix[line[1]-1, line[2]-1] = line[3]
    data_matrix_emp = data_matrix.copy()
    data_matrix_emp[data_matrix < 4] = 0
    data_matrix_emp[data_matrix >= 4]= 1 
    train_indices = list(zip(*(np.where(data_matrix != 0))))

    #clip_encoder = CLIPViTEncoder()
    #bert_encoder = SentenceBERTEncoder()

    items_csv = "/scratch/zz4330/ml-100k/Text/items.csv"
    train_ratings = "/scratch/zz4330/ml-100k/u1.base"
    test_ratings = "/scratch/zz4330/ml-100k/u1.test"

    item_path = "/scratch/zz4330/ml-100k/"

    train_dataset = MovielensDataset(train_ratings,item_path,data_matrix,device)
    test_dataset = MovielensDataset(test_ratings,item_path,data_matrix,device)
    
    

    weight = np.array([np.count_nonzero(train_dataset.data == i) for i in range(1, 6)])
    weight = weight.max() / weight
    weight = torch.Tensor(weight).to(device)
    epochs = 100
    #recommender = MultimodalRecommender(clip_encoder, bert_encoder)
    trainloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
    testloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False)
    model = BabelNet(device, train_dataset,testloader,trainloader,weight,args.batch_size, args.row)
    model.to(device)
    train_loss, test_loss, train_f1, test_f1,train_ndcg, test_ndcg = model.fit(trainloader,testloader,epochs)
    print(train_loss, test_loss, train_f1, test_f1,train_ndcg,test_ndcg)
    torch.save(model, f"{args.save_dir}/model.pth")

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(train_loss[20:], label = "Train Loss", color = "orange")
    ax2.plot(test_loss[20:], label = "Test Loss")
    fig.legend([ax, ax2], labels = ["Train Loss", "Test Loss"], loc = "upper right")
    plt.show()
    plt.savefig(f"{args.save_dir}/loss.png")
    print("Recommendation results here")

if __name__ == "__main__":
    main()
