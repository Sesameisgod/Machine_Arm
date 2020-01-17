# 常用
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os
# 資料預處理用
import math
import pywt
# 讓每一種標籤的資料都平均一點
import random
# 資料分割用
from sklearn.model_selection import train_test_split
# 機器學習相關
import torch
import torch.utils.data as Data
import torch.nn as nn
import torchvision
import torch.nn.functional as F
# 正規化
from sklearn import preprocessing
# 訓練結果報告分析
from sklearn import metrics
# 儲存報告(使用當前時間作為 pickle 檔名)
import pickle
import time

# 讀取資料的模組
def data_reading(path):
    print('【讀取檔案】')
    data = []
    for file in os.listdir(path):
        data.append(sio.loadmat(path + '/' + file))
        plt.close()
    print('檔案讀取完畢！共讀取了', len(data), '個檔案的資料\n')
    return data
# 讀取受試者資料
def subject_choose(data, subject_ID_list):
    print('【讀取受試者資料】')
    # subject_emg[受試者編號][emg編號][訊號]
    # subject_emg[受試者編號][狀態]
    subject_emg = []
    subject_restimulus = []
    # 選擇受試者
    for subject_ID in subject_ID_list:
        print('受試者', subject_ID, '資料讀取')
        sub_data = list(filter(lambda x: np.squeeze(x['subject']) == subject_ID, data))[0]
        subject_emg.append(sub_data['emg'].T)
        subject_restimulus.append(sub_data['restimulus'])
    print('受試者資料讀取完畢！\n')
    return subject_emg, subject_restimulus

# 資料預處理
def wavelet_with_energy_sum(subject_emg, subject_restimulus, sliding_window_size, sliding_duration, wavelet, mode, maxlevel):
    print('【預處理-小波包能量】')
    x_data = []
    y_data = []
    count = 0
    for emg, restimulus in zip(subject_emg, subject_restimulus):
        count += 1
        print('第' + str(count) + '位受試者資料預處理')
        total_len = len(restimulus)
        # math.ceil -> 無條件進位
        sliding_times = math.ceil((total_len - sliding_window_size) / sliding_duration) + 1

        # 資料分割 + 特徵提取
        window_begin = 0
        for i in range(sliding_times):
        #   特徵提取
            feature_matrix = []
            for e in emg:
                emg_segment = e[window_begin:window_begin+sliding_window_size]
            #   使用多階小波包轉換
            #   小波包基底: db5
            #   層數: 4
                wp = pywt.WaveletPacket(data=emg_segment, wavelet=wavelet, mode=mode, maxlevel=maxlevel)
            #   對第四層每一節點做能量值計算
                wavelet_energy = []
                for j in [node.path for node in wp.get_level(wp.maxlevel, 'natural')]:
                    wavelet_energy.append(np.sum( (np.array(wp[j].data)) ** 2 ))
                feature_matrix.append(wavelet_energy)
            x_data.append(feature_matrix)
        #   標標籤
            restimulus_segment = restimulus[window_begin:window_begin+sliding_window_size]
        #   np.sqeeze()把矩陣內的單維向量的框框消掉
            counts = np.bincount(np.squeeze(restimulus_segment))
            #返回眾數(注意:此方法只有非負數列才可使用)
            label_action_ID = np.argmax(counts)
            y_data.append(label_action_ID)
            window_begin = window_begin + sliding_duration
    print('資料預處理完畢！共', len(x_data), '筆資料\n')
    print('資料標籤數量分布：', np.bincount(np.squeeze(y_data)))
#     讓 label 0 的資料減少一點，使資料分布平均
    x_filter_data = []
    y_filter_data = []
    for i in range(len(x_data)):
        if y_data[i] == 0:
            if random.randint(1, round(np.bincount(np.squeeze(y_data))[0]/np.bincount(np.squeeze(y_data))[1]) ) == 1:
                x_filter_data.append(x_data[i])
                y_filter_data.append(y_data[i])
        else:
            x_filter_data.append(x_data[i])
            y_filter_data.append(y_data[i])
    x_data = x_filter_data
    y_data = y_filter_data
    del x_filter_data
    del y_filter_data
    print('\n資料篩選完成')
    print('資料數量： x -> ', len(x_data), ', y ->', len(y_data))
    print('資料標籤分佈：', np.bincount(np.squeeze(y_data)),'\n')
    # 正規化
    x_data = list(preprocessing.scale(np.array(x_data).reshape(-1)).reshape(-1, 10, 16))
    for i in range(len(x_data)):
        x_data[i] = [x_data[i]]
    return x_data, y_data

def wavelet_parameters_only(subject_emg, subject_restimulus, sliding_window_size, sliding_duration, wavelet, mode, maxlevel):
    print('【預處理-小波包係數')
    x_data = []
    y_data = []
    count = 0
    for emg, restimulus in zip(subject_emg, subject_restimulus):
        count += 1
        print('第' + str(count) + '位受試者資料預處理')
        total_len = len(restimulus)
        # math.ceil -> 無條件進位
        sliding_times = math.ceil((total_len - sliding_window_size) / sliding_duration) + 1
        # 資料分割 + 特徵提取
        window_begin = 0
        for i in range(sliding_times):
        #   特徵提取
            feature_matrix = []
            for e in emg:
                emg_segment = e[window_begin:window_begin+sliding_window_size]
            #   使用多階小波包轉換
            #   小波包基底: db5
            #   層數: 4
                wp = pywt.WaveletPacket(data=emg_segment, wavelet=wavelet, mode=mode, maxlevel=maxlevel)
                wavelet_parameter = []
                for j in [node.path for node in wp.get_level(wp.maxlevel, 'natural')]:
                    wavelet_parameter.append(list(wp[j].data))
                feature_matrix.append(wavelet_parameter)
            x_data.append(feature_matrix)
            if(i % 100 ==0):
                print('feature')
                print(np.array(feature_matrix).shape)
                print('x_data')
                print(np.array(x_data).shape)
        #   標標籤
            restimulus_segment = restimulus[window_begin:window_begin+sliding_window_size]
        #   np.sqeeze()把矩陣內的單維向量的框框消掉
            counts = np.bincount(np.squeeze(restimulus_segment))
            #返回眾數(注意:此方法只有非負數列才可使用)
            label_action_ID = np.argmax(counts)
            y_data.append(label_action_ID)
            window_begin = window_begin + sliding_duration
            if(i % 100 ==0):
                print('m')
                print(np.array(x_data).shape)
    print('haha')
    print(np.array(x_data).shape)
    print(np.array(x_data[0][0]).shape)
    print('資料預處理完畢！共', len(x_data), '筆資料\n')
    print('資料標籤數量分布：', np.bincount(np.squeeze(y_data)))
#     讓 label 0 的資料減少一點，使資料分布平均
    x_filter_data = []
    y_filter_data = []
    for i in range(len(x_data)):
        if y_data[i] == 0:
            if random.randint(1, round(np.bincount(np.squeeze(y_data))[0]/np.bincount(np.squeeze(y_data))[1]) ) == 1:
                x_filter_data.append(x_data[i])
                y_filter_data.append(y_data[i])
        else:
            x_filter_data.append(x_data[i])
            y_filter_data.append(y_data[i])
    x_data = x_filter_data
    y_data = y_filter_data
    del x_filter_data
    del y_filter_data
    print('\n資料篩選完成')
    print('資料數量： x -> ', len(x_data), ', y ->', len(y_data))
    print('資料標籤分佈：', np.bincount(np.squeeze(y_data)),'\n')
    # 正規化
    print('正規化')
    print(np.array(x_data).shape)
    x_data = list(preprocessing.scale(np.array(x_data).reshape(-1)).reshape(-1, 10, 16, 20))
    return x_data, y_data

# 機器學習模組

# 資料集設定
def data_setting(x_data, y_data, BATCH_SIZE):
    print('\n【資料集設定】\n')
    # 先转换成 torch 能识别的 Dataset
    torch_dataset = Data.TensorDataset(torch.FloatTensor(x_data), torch.LongTensor(y_data))

    # 把 dataset 放入 DataLoader
    loader = Data.DataLoader(
        dataset=torch_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # 要不要打乱数据 (打乱比较好)
    )
    return loader
# CNN_energy
class CNN_energy(nn.Module):
    def __init__(self):
        super(CNN_energy, self).__init__()
        # nn.Sequential()可以快速搭建神經網路
        # 卷積運算所使用的mask一般稱為kernel map，這邊為5x5的map
        # stride決定kernel map一次要走幾格
        # 上面用5x5的kernel map去跑28x28的圖片，卷積完會只剩下26x26，故加兩層
        # zero-padding 讓圖片大小相同
        self.conv1 = nn.Sequential( # input shape(channel=1, height=28, weight=28)
            nn.Conv2d(
                in_channels = 1, # 輸入信號的通道
                out_channels = 4, # 卷積產生的通道
                kernel_size = (3, 5), # 卷積核的尺寸
                stride = 1, # 卷積步長
                padding = (1,2) # 輸入的每一條邊補充0的層數
            ),  # output shape(4, 10, 16)
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            # input shape(4, 10, 16)
            nn.Conv2d(4, 8, (3,5), 1, (1,2)), # output shape(8, 10, 16)
            nn.ReLU()
        )
        self.hidden = nn.Linear(8*10*16, 800)
        self.hidden2 = nn.Linear(800, 800)
        self.hidden3 = nn.Linear(800, 800)
        self.out = nn.Linear(800, 13) # fully connected layer
     
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # 展平多維卷積圖層 (batch_size, 32*10*16)
        x = self.hidden(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        output = F.softmax(self.out(x))
        return output
# CNN_parameter
class CNN_parameter(nn.Module):
    def __init__(self):
        super(CNN_parameter, self).__init__()
        # nn.Sequential()可以快速搭建神經網路
        # 卷積運算所使用的mask一般稱為kernel map，這邊為5x5的map
        # stride決定kernel map一次要走幾格
        # 上面用5x5的kernel map去跑28x28的圖片，卷積完會只剩下26x26，故加兩層
        # zero-padding 讓圖片大小相同
        self.conv1 = nn.Sequential( # input shape(channel=1, height=28, weight=28)
            nn.Conv2d(
                in_channels = 10, # 輸入信號的通道
                out_channels = 20, # 卷積產生的通道
                kernel_size = (3, 5), # 卷積核的尺寸
                stride = 1, # 卷積步長
                padding = (1,2) # 輸入的每一條邊補充0的層數
            ),  # output shape(20, 16, 20)
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            # input shape(4, 10, 16)
            nn.Conv2d(20, 40, (3,5), 1, (1,2)), # output shape(40, 16, 20)
            nn.ReLU()
        )
        self.hidden = nn.Linear(40*16*20, 800)
        self.hidden2 = nn.Linear(800, 800)
        self.hidden3 = nn.Linear(800, 800)
        self.out = nn.Linear(800, 13) # fully connected layer
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # 展平多維卷積圖層 (batch_size, 32*10*16)
        x = self.hidden(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        output = F.softmax(self.out(x))
        return output
# 訓練模型
def module_training(loader, EPOCH, LR, module_class):
    print('【訓練模型】')
    module = module_class()
    print(module)
    # optimize all cnn parameters
    optimizer = torch.optim.Adam(module.parameters(), lr=LR)
    # the target label is not one-hotted
    # pytorch 的 CrossEntropyLoss 會自動把張量轉為 one hot形式
    loss_func = nn.CrossEntropyLoss()

    # training and testing
    for epoch in range(EPOCH):
        print('第', epoch, '次訓練')
        # enumerate : 枚舉可列舉對象．ex.
        # A = [a, b, c]
        # list(enumerate(A)) = [(0,a), (1,b), (2,c)]
        for step, (b_x, b_y) in enumerate(loader):
            output = module(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step % 50 == 0):
                print('step:' + str(step))
                print('loss:' + str(loss))
    print('訓練完成！！\n')
    return module

# 訓練結果分析
def testing_report(module, testing_x_data, testing_y_data):
    print('【訓練結果報告】')
    test_output = module(torch.FloatTensor(testing_x_data))
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    # 混淆矩陣
    confusion_matrix = metrics.confusion_matrix(testing_y_data,pred_y)
    print('混淆矩陣')
    print('true/predict')
    print(confusion_matrix)
    classification_report = metrics.classification_report(testing_y_data,pred_y)
    print('\n<============================>\n\n分類報告')
    print(classification_report)

    print('<============================>\n\n分類準確率')
    correct = 0
    for i in range(len(testing_y_data)):
        if testing_y_data[i] == pred_y[i]:
            correct += 1
    print('測試資料數：', len(testing_y_data), ', 預測正確數：', correct, '準確率：', (correct/len(testing_y_data)))
    return confusion_matrix, classification_report, (correct/len(testing_y_data))

feature_extract_method = {'wavelet_with_energy_sum':wavelet_with_energy_sum, 'wavelet_parameters_only':wavelet_parameters_only}
CNN_module = {'wavelet_with_energy_sum':CNN_energy, 'wavelet_parameters_only':CNN_parameter}

# ====================== 開始測試 ==========================
data = data_reading('S_All_A1_E1')
subject = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
#subject = [1,2,3]
feature_extract = 'wavelet_parameters_only'
sliding_window_size = 200
sliding_window_movement = 100
batch_size = 2048
learning_rate = 0.0005
epoch_list = [10, 20 ,30, 40, 50, 60, 70 ,80, 90, 100, 110, 120, 130, 140, 150]
subject_emg, subject_restimulus = subject_choose(data, subject)
x_data, y_data = feature_extract_method[feature_extract](subject_emg, subject_restimulus, sliding_window_size, sliding_window_movement, 'db5', 'symmetric', 4)
# 將資料拆分成訓練與測試集
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3, random_state = 0)
print('\n資料拆分完成')
print('訓練資料: x -> ', len(x_train), ', y -> ', len(y_train))
print('測試資料: x -> ', len(x_test), ', y -> ', len(y_test))
loader = data_setting(x_train, y_train, batch_size)
for epoch in epoch_list:
    module = module_training(loader, epoch, learning_rate, CNN_module[feature_extract])
    print('\n訓練資料結果分析\n')
    confusion_matrix_train, classification_report_train, accuracy_train = testing_report(module, x_train, y_train)
    print('\n測試資料結果分析\n')
    confusion_matrix_test, classification_report_test, accuracy_test = testing_report(module, x_test, y_test)
    test_report_dict = {'epoch':epoch,
                        'batch_size':batch_size, 
                        'module':module, 'sliding_window_size':sliding_window_size, 
                        'sliding_window_movement':sliding_window_movement, 
                        'feature_extract':feature_extract, 
                        'learning_rate':learning_rate, 
                        'subject':subject, 
                        'confusion_matrix_train':confusion_matrix_train, 
                        'classification_report_train':classification_report_train,
                        'accuracy_train': accuracy_train, 
                        'confusion_matrix_test':confusion_matrix_test, 
                        'classification_report_test':classification_report_test,
                        'accuracy_test': accuracy_test}
    #     path = 'training_report/DB1_training_report_' + time.strftime("%Y%m%d_%H%M%S", time.localtime())
    path = 'training_report/DB1_training_report_' + time.strftime("%Y%m%d_%H%M%S", time.localtime()) + '.pickle'
    file = open(path, 'wb')
    pickle.dump(test_report_dict, file)
    file.close()
    print('\n訓練完畢! 結果報告已匯入至 ' + path)