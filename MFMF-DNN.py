import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler 
import torch,re,os,time
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import  train_test_split
from sklearn.utils import shuffle
import scipy.io as scio
import torch.nn as nn
import gc, math 
from captum.attr import IntegratedGradients, Saliency
torch.__version__

torch.manual_seed(128)
np.random.seed(1024)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

def normalize_to_sum_1(data):
    total = sum(data)
    if total == 0:
        raise ValueError("Error")
    normalized_data = [x / total for x in data]
    return normalized_data

def normalization1(data):
    _range = 22.2 -2.2
    return (data - 2.2) / _range

def normalization2(data):
    _range = 1440 - 0
    return (data - 0) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def RMSE_MARD(pred_values,ref_values):    
    data_length = len(ref_values)
    pred_values = pred_values.reshape(-1)
    ref_values = ref_values.reshape(-1)
    total = 0
    for i in range (data_length):
        temp = (ref_values[i] - pred_values[i]) * (ref_values[i] - pred_values[i])
        total = total + temp

    smse_value = math.sqrt(total / data_length)
    print('RMSE: ', smse_value)
    
    total = 0
    for i in range(data_length):
        temp = abs((ref_values[i] - pred_values[i]) / ref_values[i])
        total = total + temp  
    mard_value = total / data_length
    print('MARD:  ', mard_value)
        
    return smse_value,mard_value

def clarke_error_grid(pred_values,ref_values):
    pred_values = pred_values.reshape(-1)
    ref_values = ref_values.reshape(-1)
    assert (len(ref_values) == len(pred_values)), "Unequal number of values (reference : {}) (prediction : {}).".format(len(ref_values), len(pred_values))
    if max(ref_values) > 400 or max(pred_values) > 400:
        print ("Input Warning: the maximum reference value {} or the maximum prediction value {} exceeds the normal physiological range of glucose (<400 mg/dl).".format(max(ref_values), max(pred_values)))
    if min(ref_values) < 0 or min(pred_values) < 0:
        print ("Input Warning: the minimum reference value {} or the minimum prediction value {} is less than 0 mg/dl.".format(min(ref_values),  min(pred_values)))

#    plt.figure(figsize=(6,6),dpi = 200)
#
#    plt.scatter(ref_values, pred_values, marker='o', color='black', s=8)
#    plt.xlabel("Reference Concentration (mg/dl)")
#    plt.ylabel("Prediction Concentration (mg/dl)")
#    plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
#    plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
#    plt.gca().set_facecolor('white')
#
#    #Set axes lengths
#    plt.gca().set_xlim([0, 400])
#    plt.gca().set_ylim([0, 400])
#    plt.gca().set_aspect((400)/(400))
#
#    #Plot zone lines
#    plt.plot([0,400], [0,400], ':', c='black')                      #Theoretical 45 regression line
#    plt.plot([0, 175/3], [70, 70], '-', c='black')
#    #plt.plot([175/3, 320], [70, 400], '-', c='black')
#    plt.plot([175/3, 400/1.2], [70, 400], '-', c='black')           #Replace 320 with 400/1.2 because 100*(400 - 400/1.2)/(400/1.2) =  20% error
#    plt.plot([70, 70], [84, 400],'-', c='black')
#    plt.plot([0, 70], [180, 180], '-', c='black')
#    plt.plot([70, 290],[180, 400],'-', c='black')
#    # plt.plot([70, 70], [0, 175/3], '-', c='black')
#    plt.plot([70, 70], [0, 56], '-', c='black')                     #Replace 175.3 with 56 because 100*abs(56-70)/70) = 20% error
#    # plt.plot([70, 400],[175/3, 320],'-', c='black')
#    plt.plot([70, 400], [56, 320],'-', c='black')
#    plt.plot([180, 180], [0, 70], '-', c='black')
#    plt.plot([180, 400], [70, 70], '-', c='black')
#    plt.plot([240, 240], [70, 180],'-', c='black')
#    plt.plot([240, 400], [180, 180], '-', c='black')
#    plt.plot([130, 180], [0, 70], '-', c='black')
#
#    #Add zone titles
#    plt.text(30, 15, "A", fontsize=15)
#    plt.text(370, 260, "B", fontsize=15)
#    plt.text(280, 370, "B", fontsize=15)
#    plt.text(160, 370, "C", fontsize=15)
#    plt.text(160, 15, "C", fontsize=15)
#    plt.text(30, 140, "D", fontsize=15)
#    plt.text(370, 120, "D", fontsize=15)
#    plt.text(30, 370, "E", fontsize=15)
#    plt.text(370, 15, "E", fontsize=15)

    #Statistics from the data
    zone = [0] * 5
    for i in range(len(ref_values)):
        if (ref_values[i] <= 70 and pred_values[i] <= 70) or (pred_values[i] <= 1.2*ref_values[i] and pred_values[i] >= 0.8*ref_values[i]):
            zone[0] += 1    #Zone A

        elif (ref_values[i] >= 180 and pred_values[i] <= 70) or (ref_values[i] <= 70 and pred_values[i] >= 180):
            zone[4] += 1    #Zone E

        elif ((ref_values[i] >= 70 and ref_values[i] <= 290) and pred_values[i] >= ref_values[i] + 110) or ((ref_values[i] >= 130 and ref_values[i] <= 180) and (pred_values[i] <= (7/5)*ref_values[i] - 182)):
            zone[2] += 1    #Zone C
        elif (ref_values[i] >= 240 and (pred_values[i] >= 70 and pred_values[i] <= 180)) or (ref_values[i] <= 175/3 and pred_values[i] <= 180 and pred_values[i] >= 70) or ((ref_values[i] >= 175/3 and ref_values[i] <= 70) and pred_values[i] >= (6/5)*ref_values[i]):
            zone[3] += 1    #Zone D
        else:
            zone[1] += 1    #Zone B

    A_score = zone[0]/(zone[0]+zone[1]+zone[2]+zone[3]+zone[4])
    B_score = zone[1]/(zone[0]+zone[1]+zone[2]+zone[3]+zone[4])
    C_score = zone[2]/(zone[0]+zone[1]+zone[2]+zone[3]+zone[4])
    D_score = zone[3]/(zone[0]+zone[1]+zone[2]+zone[3]+zone[4])
    E_score = zone[4]/(zone[0]+zone[1]+zone[2]+zone[3]+zone[4])
    score = np.matrix([A_score, B_score, C_score, D_score, E_score]) 
    print('Clarke:  A:{:.4f}, B:{:.4f}, C:{:.4f}, D:{:.4f}, E:{:.4f}'.format(A_score, B_score, C_score, D_score, E_score))
    return score

def BG_Range(bg_value):    
    data_length = len(bg_value)
    for i in range(data_length):
        if bg_value[i] > 22.2:
            bg_value[i] = 22.2
        elif bg_value[i] < 2.2:
            bg_value[i] = 2.2  
    bg_value = bg_value.reshape(-1)
    return bg_value



def captum_analysis(model, data_loader, device, batch_size=200):
    model.train()
    
    sample_data, _ = next(iter(data_loader))
    sample_data = sample_data[:batch_size].to(device)  
    
    # using Integrated Gradients
    ig = IntegratedGradients(model)
    attributions, _ = ig.attribute(sample_data, return_convergence_delta=True)    
    attrs_np = attributions.cpu().detach().numpy()    
    cgm_attributions = attrs_np[:, 0, :]
    plt.figure(figsize=(15, 10))
    cgm_attributions = abs(cgm_attributions)
    avg_attribution = np.mean(cgm_attributions, axis=0)
    norm_attribution = normalize_to_sum_1(avg_attribution)
    norm_attribution = np.array(norm_attribution)
    plt.imshow(norm_attribution.reshape(1, -1), 
                cmap='coolwarm',
                aspect='auto',
                interpolation='nearest')
    plt.yticks([], [])       
    plt.tight_layout()
    plt.show()
    
    return norm_attribution


class LSTM_Atten(nn.Module):
    def __init__(self,look_back,pre_len):
        super(LSTM_Atten, self).__init__()
        self.cnn1 = nn.Conv1d(in_channels = 1, out_channels = 4, stride = 1 , kernel_size=6, padding=0) 
        self.cnn2 = nn.Conv1d(in_channels = 1, out_channels = 4, stride = 1 , kernel_size=3, padding=0)
        self.cnn3 = nn.Conv1d(in_channels = 1, out_channels = 4, stride = 1 , kernel_size=9, padding=0)
        self.lstm = nn.LSTM(input_size=5, 
                            hidden_size=128, 
                            num_layers=1, 
                            batch_first=True, 
                            )
        self.lstm4 = nn.LSTM(input_size=2,
                            hidden_size=128, 
                            num_layers=1, 
                            batch_first=True,
                            )        
        self.lstmcell=nn.LSTMCell(input_size=128, hidden_size=128)
        self.drop=nn.Dropout(0.2) 
        self.fc1=nn.Linear(256,128)
        self.fc2=nn.Linear(128,1)
        self.fc3=nn.Linear(12,1)
        self.fc4=nn.Linear(6,1)
        self.look_back=look_back
        self.pre_len=pre_len
        self.Softmax=nn.Softmax(dim=1)
             
        
    def forward(self, x):
        ori = x.permute(0,2,1)                
        xx = x[:,0,:].unsqueeze(1)
        
        cov1 = self.cnn1(xx)
        cov1 = cov1.permute(0,2,1)
        x1 = x[:,1,5:72].unsqueeze(1)
        x1 = x1.permute(0,2,1)       
        COV1 = torch.cat([cov1,x1],2)       
        H1,(h1,c1) = self.lstm(COV1.float(),None) 
        h1 = h1.squeeze(0)
        c1 = c1.squeeze(0)        
        H_pre1 = torch.empty((h1.shape[0],self.pre_len,128*2)).to(DEVICE)
        for i in range(self.pre_len): 
            h_t1, c_t1 = self.lstmcell(h1, (h1, c1))  
            H1 = torch.cat((H1,h_t1.unsqueeze(1)),1)
            h_atten1 = self.Atten(H1) 
            H_pre1[:,i,:] = h_atten1  
            h1, c1 = h_t1, c_t1  
        r1 = self.fc2(self.fc1(H_pre1)).squeeze(2)

        cov2 = self.cnn2(xx)
        cov2 = cov2.permute(0,2,1)
        x2 = x[:,1,2:72].unsqueeze(1)
        x2 = x2.permute(0,2,1)       
        COV2 = torch.cat([cov2,x2],2)        
        H2,(h2,c2) = self.lstm(COV2.float(),None) 
        h2 = h2.squeeze(0)
        c2 = c2.squeeze(0)        
        H_pre2 = torch.empty((h2.shape[0],self.pre_len,128*2)).to(DEVICE)
        for i in range(self.pre_len): 
            h_t2, c_t2 = self.lstmcell(h2, (h2, c2))  
            H2 = torch.cat((H2,h_t2.unsqueeze(1)),1)
            h_atten2 = self.Atten(H2) 
            H_pre2[:,i,:] = h_atten2  
            h2, c2 = h_t2, c_t2  
        r2 = self.fc2(self.fc1(H_pre2)).squeeze(2)
    
        cov3 = self.cnn3(xx)
        cov3 = cov3.permute(0,2,1)
        x3 = x[:,1,8:72].unsqueeze(1)
        x3 = x3.permute(0,2,1)        
        COV3 = torch.cat([cov3,x3],2)
        H3,(h3,c3) = self.lstm(COV3.float(),None) 
        h3 = h3.squeeze(0)
        c3 = c3.squeeze(0)        
        H_pre3 = torch.empty((h3.shape[0],self.pre_len,128*2)).to(DEVICE)
        for i in range(self.pre_len): 
            h_t3, c_t3 = self.lstmcell(h3, (h3, c3))  
            H3 = torch.cat((H3,h_t3.unsqueeze(1)),1)
            h_atten3 = self.Atten(H3) 
            H_pre3[:,i,:] = h_atten3  
            h3, c3 = h_t3, c_t3  
        r3 = self.fc2(self.fc1(H_pre3)).squeeze(2) 
    

        H4,(h4,c4) = self.lstm4(ori.float(),None) 
        h4 = h4.squeeze(0)
        c4 = c4.squeeze(0)        
        H_pre4 = torch.empty((h4.shape[0],self.pre_len,128*2)).to(DEVICE)
        for i in range(self.pre_len): 
            h_t4, c_t4 = self.lstmcell(h4, (h4, c4))  
            H4 = torch.cat((H4,h_t4.unsqueeze(1)),1)
            h_atten4 = self.Atten(H4) 
            H_pre4[:,i,:] = h_atten4  
            h4, c4 = h_t4, c_t4  
        r4 = self.fc2(self.fc1(H_pre4)).squeeze(2)                 
        rr = torch.cat([r1,r2,r3,r4],1)
        RR = self.fc3(rr)
        return RR   
      
    
    def Atten(self,H):
        h = H[:,-1,:].unsqueeze(1) 
        H = H[:,-1-self.look_back:-1,:] 
        atten = torch.matmul(h,H.transpose(1,2)).transpose(1,2) 
        atten = self.Softmax(atten)
        atten_H = atten*H 
        atten_H = torch.sum(atten_H,dim=1).unsqueeze(1) 
        return torch.cat((atten_H,h),2).squeeze(1)




numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

#Enter the path of the dataset
path1 = '/data0/T1DM dataset/'

data_dir_list1 = sorted(os.listdir(path1),key=numericalSort) 
data_dir_s = shuffle(data_dir_list1, random_state = 8)

seq_lengh = 72
# when the prediction horizon(PH)=30 minutes, space_lengh = 6; when PH=60 minutes, space_lengh = 12
space_lengh = 6 
train_data_list = []
train_label_list = []
test_data_list = []
test_label_list = []
N = 0
for path in data_dir_s:
    if N < 422:      
      all_data = scio.loadmat(path1 + '/' + path)   
      dataset = all_data["CGM_data"]
      id_label = dataset[:,0]
      cgm_value = dataset[:,1:3]
      data_len = len(id_label)
      sign_list = []      
      for m in range(data_len):
          if id_label[m] == 0:
              sign_list.append(m)
      id_len = len(sign_list)  
      for n in range(id_len - 1):
          if (sign_list[n+1] - sign_list[n]) > (seq_lengh + space_lengh + 6):
              P1 = int(sign_list[n])
              P2 = int (sign_list[n + 1])
              temp_CGM = cgm_value[P1:P2,0]
              temp_CGM = BG_Range(temp_CGM)
              temp_CGM = normalization1(temp_CGM)
              cc = cgm_value[P1:P2,1]
              cc = normalization2(cc)
              Temp_CGM = np.hstack((temp_CGM.reshape(-1,1),cc.reshape(-1,1)))
              
              total_size = len(Temp_CGM)      
              for t in range(total_size - seq_lengh - space_lengh):
                  j = seq_lengh + t
                  k = seq_lengh + t + space_lengh - 1
                  train_data_list.append(Temp_CGM[t:j])
                  train_label_list.append(Temp_CGM[k,0]*20+2.2)             
      N = N + 1
    else:
      all_data = scio.loadmat(path1 + '/' + path)   
      dataset = all_data["CGM_data"]
      id_label = dataset[:,0]
      cgm_value = dataset[:,1:3]
      data_len = len(id_label)
      sign_list = []      
      for m in range(data_len):
          if id_label[m] == 0:
              sign_list.append(m)
      id_len = len(sign_list)  
      for n in range(id_len - 1):
          if (sign_list[n+1] - sign_list[n]) > (seq_lengh + space_lengh + 6):
              P1 = int(sign_list[n])
              P2 = int (sign_list[n + 1])
              temp_CGM = cgm_value[P1:P2,0]
              temp_CGM = BG_Range(temp_CGM)
              temp_CGM = normalization1(temp_CGM)
              cc = cgm_value[P1:P2,1]
              cc = normalization2(cc)
              Temp_CGM = np.hstack((temp_CGM.reshape(-1,1),cc.reshape(-1,1)))
              total_size = len(Temp_CGM)      
              for t in range(total_size - seq_lengh - space_lengh):
                  j = seq_lengh + t
                  k = seq_lengh + t + space_lengh - 1
                  test_data_list.append(Temp_CGM[t:j])
                  test_label_list.append(Temp_CGM[k,0]*20+2.2)  
      N = N + 1        

train_data1 = np.array(train_data_list)
train_label = np.array(train_label_list)
test_data = np.array(test_data_list)
test_label = np.array(test_label_list) 

train_data = train_data1.transpose(0,2,1)
Test_Data = test_data.transpose(0,2,1)
train_label = np.expand_dims(train_label,axis = 1)
Test_Label = np.expand_dims(test_label,axis = 1)


m1 = int(0.875 * len(train_label))
m2 = len(train_label)
Train_Data = train_data[0:m1]
Train_Label = train_label[0:m1]
Val_Data = train_data[m1:m2]
Val_Label = train_label[m1:m2]


Train_Data = torch.tensor(Train_Data,dtype=torch.float)
Val_Data = torch.tensor(Val_Data,dtype=torch.float)
Test_Data = torch.tensor(Test_Data,dtype=torch.float)
Train_Label = torch.tensor(Train_Label,dtype=torch.float)
Val_Label = torch.tensor(Val_Label,dtype=torch.float)
Test_Label = torch.tensor(Test_Label,dtype=torch.float)


BATCH_SIZE = 512
train_set = torch.utils.data.TensorDataset(Train_Data, Train_Label)
val_set = torch.utils.data.TensorDataset(Val_Data, Val_Label)
test_set = torch.utils.data.TensorDataset(Test_Data, Test_Label)

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False) 

look_back = 72
pre_len = 3



lstm = LSTM_Atten(look_back,pre_len).to(DEVICE)

criterion = nn.MSELoss().to(DEVICE)
LEARNING_RATE = 0.0001
optimizer = torch.optim.Adam(lstm.parameters(), lr=LEARNING_RATE)
TOTAL_EPOCHS = 1000

epoch_list = []
Loss_list = []
val_rmse_list = []
val_mard_list = []
test_rmse_list = []
test_mard_list = []
val_ClarkeA_list = []
val_ClarkeAB_list = []
test_ClarkeA_list = []
test_ClarkeAB_list = []

for epoch in range(TOTAL_EPOCHS):
    val_lstm_list = []
    val_true_list = []
    test_lstm_list = []
    test_true_list = []
    total_loss = 0
    p = 0
    print('epoch = ', int(epoch))
    lstm.train()   
    for x, (train_cgm, tr_labels) in enumerate(train_loader):
        train_cgm = train_cgm.to(DEVICE)
        tr_labels = tr_labels.to(DEVICE)
        optimizer.zero_grad()
        train_outputs = lstm(train_cgm) 
        train_loss = criterion(train_outputs,tr_labels)
        train_loss.backward()
        optimizer.step()     

    lstm.eval()         
    for y, (val_cgm, val_labels) in enumerate(val_loader): 
        p = p + 1
        val_cgm = val_cgm.to(DEVICE) 
        val_outputs = lstm(val_cgm)
        val_outputs = val_outputs.cpu()
        val_loss = criterion(val_outputs,val_labels)
        total_loss = total_loss + val_loss.item()
        val_outputs = val_outputs.detach().numpy()
        val_labels = val_labels.numpy()                            
        for w in range(len(val_outputs)):
              val_lstm_list.append(val_outputs[w,0])
              val_true_list.append(val_labels[w,0])
              
    aver_loss = total_loss / p
    print('aver_loss = ', aver_loss)
    val_lstm = np.array(val_lstm_list)
    val_true = np.array(val_true_list) 
    val_aver = BG_Range(val_lstm)*18
    val_true = BG_Range(val_true)*18
    val_rmse_mard =  RMSE_MARD(val_aver, val_true)
    val_Clarke = clarke_error_grid(val_aver, val_true)    
    Loss_list.append(aver_loss)
    val_rmse_list.append(val_rmse_mard[0])
    val_mard_list.append(val_rmse_mard[1])
    val_ClarkeA_list.append(val_Clarke[0,0])
    val_ClarkeAB_list.append(val_Clarke[0,1])    
    epoch_list.append(epoch)
    epoch_result = np.array(epoch_list)
    Loss_result = np.array(Loss_list)
    val_rmse_result = np.array(val_rmse_list)
    val_mard_result = np.array(val_mard_list)    
    val_ClarkeA = np.array(val_ClarkeA_list)
    val_ClarkeAB = np.array(val_ClarkeAB_list)
    Loss_result = Loss_result.reshape(-1,1)   
    val_rmse_result = val_rmse_result.reshape(-1,1)
    val_mard_result = val_mard_result.reshape(-1,1)
    epoch_result = epoch_result.reshape(-1,1)
    val_ClarkeA = val_ClarkeA.reshape(-1,1)
    val_ClarkeAB = val_ClarkeAB.reshape(-1,1)
                 
    for y, (test_cgm, te_labels) in enumerate(test_loader):
        test_cgm = test_cgm.to(DEVICE)    
        test_outputs = lstm(test_cgm)
        Test_outputs = test_outputs.cpu()
        Test_outputs = Test_outputs.detach().numpy()
        Te_labels = te_labels.numpy()                
             
        for w in range(len(Test_outputs)):
              test_lstm_list.append(Test_outputs[w,0])
              test_true_list.append(Te_labels[w,0])
             
    test_lstm = np.array(test_lstm_list)
    test_true = np.array(test_true_list)
    test_aver = BG_Range(test_lstm)*18 
    test_true = BG_Range(test_true)*18    
    test_rmse_mard =  RMSE_MARD(test_aver, test_true)
    test_Clarke = clarke_error_grid(test_aver, test_true)
    test_rmse_list.append(test_rmse_mard[0])
    test_mard_list.append(test_rmse_mard[1])
    test_ClarkeA_list.append(test_Clarke[0,0])
    test_ClarkeAB_list.append(test_Clarke[0,1])
   
    test_rmse_result = np.array(test_rmse_list)
    test_mard_result = np.array(test_mard_list)
    test_ClarkeA = np.array(test_ClarkeA_list)
    test_ClarkeAB = np.array(test_ClarkeAB_list)
      
    test_rmse_result = test_rmse_result.reshape(-1,1)
    test_mard_result = test_mard_result.reshape(-1,1)
    test_ClarkeA = test_ClarkeA.reshape(-1,1)
    test_ClarkeAB = test_ClarkeAB.reshape(-1,1)
    
    output_data = np.hstack((epoch_result, Loss_result,val_rmse_result, val_mard_result,val_ClarkeA,val_ClarkeAB,
                              test_rmse_result, test_mard_result,test_ClarkeA,test_ClarkeAB))
    
    np.savetxt('/data0/T1DM-30-result.csv', output_data, delimiter=',', fmt='%1.5f')
    
    attributions = captum_analysis(lstm, test_loader, DEVICE, batch_size=200)   
    

    
   