import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, classification_report
from utils.train_utils import weighted_mse_loss
from sklearn.metrics import ndcg_score

class BabelNet(nn.Module):
    def __init__(self, device, train_dataset,testloader,trainloader,weight,batch_size, row, channel = 84):
        super(BabelNet, self).__init__()
        self.device = device
        self.ROW = row
        self.BATCH = batch_size
        self.testloader = testloader
        self.trainloader = trainloader
        self.encoder_user = nn.Sequential(OrderedDict([
            ('linr1', nn.Linear(train_dataset.user_embedding_size, 1024)),
            ('relu1', nn.LeakyReLU()),
            ('linr2', nn.Linear(1024, channel)),
            ('relu2', nn.LeakyReLU()),
        ]))
        
        self.encoder_item = nn.Sequential(OrderedDict([
            ('linr1', nn.Linear(train_dataset.item_embedding_size, 256)),
            ('relu1', nn.LeakyReLU()),
            ('linr2', nn.Linear(256, 300)),
            ('relu2', nn.LeakyReLU()),
        ]))
        
        
        self.encoder_audio = nn.Sequential(OrderedDict([
            ('linr1', nn.Linear(train_dataset.audio_embedding_size, 600)),
            ('relu1', nn.LeakyReLU()),
            ('linr4', nn.Linear(600, 300)),
            ('relu4', nn.LeakyReLU()),
        ]))
        
        self.encoder_text = nn.Sequential(OrderedDict([
            ('linr1', nn.Linear(train_dataset.text_embedding_size, 256)),
            ('relu1', nn.LeakyReLU()),
            ('linr2', nn.Linear(256, 300)),
            ('relu2', nn.LeakyReLU()),
        ]))
        
        self.encoder_meta = nn.Sequential(OrderedDict([
            ('linr1', nn.Linear(train_dataset.meta_embedding_size, 1600)),
            ('relu1', nn.LeakyReLU()),
#             ('norm1', nn.BatchNorm1d(1600)),
            ('linr2', nn.Linear(1600, 300)),
            ('relu2', nn.LeakyReLU()),
#             ('linr3', nn.Linear(1000, 500)),
#             ('relu3', nn.LeakyReLU()),
#             ('linr4', nn.Linear(500, 300)),
#             ('relu4', nn.LeakyReLU()),
        ]))
        
        self.fusion = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 1, (20, 20), stride=(2, 2))), #(20,10)
            ('relu1', nn.LeakyReLU()),
            ('conv2', nn.Conv2d(1, 1, (5, 50), stride=(1, 1))), #(5,5)
            ('relu2', nn.LeakyReLU()),
        ]))
        
        
        self.babel_u = nn.Sequential(OrderedDict([
            ('linr1', nn.Linear(channel, 200)),
            ('relu1', nn.LeakyReLU()),
            ('linr2', nn.Linear(200, 256)),
            ('relu2', nn.LeakyReLU()),
            ('linr3', nn.Linear(256, 100)),
            ('relu3', nn.LeakyReLU()),
        ]))

        self.babel_i = nn.Sequential(OrderedDict([
            ('linr1', nn.Linear(channel, 200)),
            ('relu1', nn.LeakyReLU()),
            ('linr2', nn.Linear(200, 256)),
            ('relu2', nn.LeakyReLU()),
            ('linr3', nn.Linear(256, 100)),
            ('relu3', nn.LeakyReLU()),
        ]))

        self.ffn = nn.Sequential(OrderedDict([
            ('linr1', nn.Linear(300, 164)),
            ('actv1', nn.ReLU()),
            ('linr2', nn.Linear(164, 1)),
#             ('actv2', nn.ReLU()),
#             ('linr3', nn.Linear(50, 1)),
        ]))
        
        self.device = device
        self.encoder_user.apply(self.init_weights)
        self.encoder_item.apply(self.init_weights)
        self.encoder_audio.apply(self.init_weights)
        self.encoder_text.apply(self.init_weights)
        self.encoder_meta.apply(self.init_weights)
        self.babel_u.apply(self.init_weights)
        self.babel_i.apply(self.init_weights)
        self.ffn.apply(self.init_weights)
        self.weight = weight
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.01)
            
    def exp(self, q, k, v):
        z = torch.bmm(q, k.permute(0, 2, 1))
        z = F.normalize(z, p = 10, dim = 1)
        z = torch.softmax(z, 1)
        z = torch.bmm(z, v)
        return(z)
        
    def forward(self, x1, x2):
        # Modality-encoders
        
        outu = self.encoder_user(x1)
        outa = torch.split(self.encoder_audio(x2[1]), [100, 100, 100], 1)
        outt = torch.split(self.encoder_text(x2[2]), 100, 1)
        outm = torch.split(self.encoder_meta(x2[4]), [100, 100, 100], 1)
        ROW = self.ROW
        # Attention
        q_t = outt[0].unsqueeze(1).repeat(1, ROW, 1)
        k_t = outt[1].unsqueeze(1).repeat(1, ROW, 1)
        v_t = outt[2].unsqueeze(1).repeat(1, ROW, 1)

        q_a = outa[0].unsqueeze(1).repeat(1, ROW, 1)
        k_a = outa[1].unsqueeze(1).repeat(1, ROW, 1)
        v_a = outa[2].unsqueeze(1).repeat(1, ROW, 1)
     
        
        q_m = outm[0].unsqueeze(1).repeat(1, ROW, 1)
        k_m = outm[1].unsqueeze(1).repeat(1, ROW, 1)
        v_m = outm[2].unsqueeze(1).repeat(1, ROW, 1)
        #print("Q:",q_m.shape)
        # Self-Attention
        st = self.exp(q_t, k_t, v_t)
        sm = self.exp(q_m, k_m, v_m)
        sa = self.exp(q_a, k_a, v_a)

        #print("sm:",sm.shape)
        # Inter-Modal Attention
        ita = self.exp(q_a, k_t, v_a)
        itm = self.exp(q_t, k_m, v_m)
        ima = self.exp(q_a, k_m, v_m)
        iat = self.exp(q_t, k_a, v_t)
        
        # Forward
        ma = torch.mean(torch.stack([sa, ima]), 0)
        mt = torch.mean(torch.stack([st, itm]), 0)
        #print("ma: ",ma.shape)
        #print("itm: ",itm.shape)
        se = torch.mul(ma, mt)
        outi = torch.cat((se, sm), axis = 2)#.reshape(-1, ROW * 1200) #Principal Attention
        #outi = sm
        #print("outi: ",outi.shape)
        outi = self.fusion(outi.unsqueeze(1))
        #print("outi: ",outi.shape)
        #print("outu: ",outu.shape)
        out1 = self.babel_i(outu)
        #print("out1: ",out1.shape)
        #print("Shape of outi before reshape:", outi.shape)

        out2 = self.babel_u(outi.reshape(self.BATCH, -1))
        #print("out2: ",out2.shape)
        diff = torch.cat((out1, out2, outm[2]), axis=1)
        out = self.ffn(diff)
        return(out, out1, out2)
    
    def fit(self, trainloader, 
            testloader, epochs = 200):
        self.criterion_rate = weighted_mse_loss
        self.criterion_embd = nn.CosineEmbeddingLoss()
        self.optimizer = optim.Adam(self.parameters(), lr = 1e-4)
        
        train_loss, train_f1, test_loss, test_f1,train_ndcg, test_ndcg = [], [], [], [], [], []
        for epoch in range(epochs):
            running_loss = 0.0
            running_loss_1 = 0.0
            
            for i, data in tqdm(enumerate(trainloader)):
                self.train()
                x1, x2, y = data
                y_flt = y.type(torch.FloatTensor).to(self.device)
                y_lng = torch.div(y, 4, rounding_mode="floor").to(self.device)
                self.optimizer.zero_grad()
                reg, outu, outi = self.forward(x1, x2)
                loss_1, loss_ = self.criterion_rate(reg.squeeze(), y_flt,self.weight)
                #loss_2 = self.criterion_embd(outu, outi, y_lng * 2 - 1)
                loss = loss_1
                loss.backward()
                self.optimizer.step()

                running_loss_1 += torch.sqrt(loss_)
                running_loss += loss
            vl, vp, vr, vf, tp, tr, tf,ndcg_v,ndcg_t = self.evaluate(self.testloader,self.trainloader)
            print('Epoch-%d: Loss = %.3f\nTrain RMSE = %.3f||Train Precision = %.3f||Train Recall = %.3f||Train F1 = %.3f||Train NDCG = %.3f\nTest RMSE = %.3f || Test Precision = %.3f||Test F1 = %.3f|| Test Recall = %.3f||Test NDCG = %.3f'%
                  (epoch + 1, running_loss / i, running_loss_1 / i, 
                   tp, tr,tf,ndcg_t, vl, vp, vr,vf,ndcg_v,))
            train_loss.append((running_loss_1 / i).cpu().detach().numpy())
            test_loss.append(vl.cpu().detach().numpy())
            train_f1.append(tf)
            test_f1.append(vf)
            train_ndcg.append(ndcg_t)
            test_ndcg.append(ndcg_v)
        return(train_loss, test_loss, train_f1, test_f1,train_ndcg,test_ndcg)
            
    def evaluate(self, testloader,trainloader, k = 3):
        self.eval()
        ndcg_scores_t = []
        ndcg_scores_v = []
        with torch.no_grad():
            for data in testloader:
                x1, x2, y = data
                y_flt = y.type(torch.FloatTensor).to(self.device)
                y_lng = torch.div(y, 4, rounding_mode="floor").to(self.device)
                outputs = self.forward(x1, x2)[0].squeeze()
                pred = (outputs > k).float()
                
                # Calculate NDCG
                relevance = (y_flt > k).float().cpu().numpy() 
                predictions = outputs.cpu().numpy()
                ndcg = ndcg_score([relevance], [predictions], k=min(k, len(relevance)))
                ndcg_scores_v.append(ndcg)
                
                # Calculate other metrics
                vl = torch.sqrt(self.criterion_rate(outputs, y_flt, self.weight)[1])
                vp = precision_score(y_lng.cpu(), pred.cpu(), zero_division=0)
                vr = recall_score(y_lng.cpu(), pred.cpu(), zero_division=0)
                vf = f1_score(y_lng.cpu(), pred.cpu(), zero_division=0)

            for data in trainloader:
                x1, x2, y = data
                y_flt = y.type(torch.FloatTensor).to(self.device)
                y_lng = torch.div(y, 4, rounding_mode="floor").to(self.device)
                outputs = self.forward(x1, x2)[0].squeeze()
                pred = (outputs > k).float()
                
                # Calculate NDCG
                relevance = (y_flt > k).float().cpu().numpy()  # Define relevance based on a threshold
                predictions = outputs.cpu().numpy()
                ndcg = ndcg_score([relevance], [predictions], k=min(k, len(relevance)))
                ndcg_scores_t.append(ndcg)
                
                # Calculate other metrics
                tl = torch.sqrt(self.criterion_rate(outputs, y_flt, self.weight)[1])
                tp = precision_score(y_lng.cpu(), pred.cpu(), zero_division=0)
                tr = recall_score(y_lng.cpu(), pred.cpu(), zero_division=0)
                tf = f1_score(y_lng.cpu(), pred.cpu(), zero_division=0)
        avg_ndcg_t = sum(ndcg_scores_t) / len(ndcg_scores_t)
        avg_ndcg_v = sum(ndcg_scores_v) / len(ndcg_scores_v)
        return(vl, vp*100, vr*100, vf*100, tp*100, tr*100, tf*100,avg_ndcg_v,avg_ndcg_t)
