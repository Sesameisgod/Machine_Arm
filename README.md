# Machine_Arm
this is a  research and develop for EMG based machine arm<br>

# Ninapro_Data_Reading.ipynb
Ninapro Database 的資料讀取練習與分析<br>
使用的資料庫為 Ninapro DB1中的S1_A1_E1<br>
說明：五根手指的捲曲舒張動作（動作圖：https://www.nature.com/articles/sdata201453/figures/2)<br>
動作：總共有12個動作，每個動作會做10次<br>

數據集標籤：<br>
	• emg : 量測到的10個emg訊號
	• glove : 使用CyberGlove II量測到的22個手部角度變化（節點對應圖：http://ninapro.hevs.ch/node/2)<br>
	• Subject: 受試者編號<br>
	• Exercise: 受試者所執行的實驗編號<br>
	• Stimulus: 受試者所執行的動作(i.e. 動作1 到 動作12)<br>
	• Restimulus: 受試者所執行的動作(i.e. 動作1 到 動作12)，並調整動作時間間隔使其完全匹配動作(優化過的)。<br>
	• Repetition: 受試者重複該動作幾次了<br>
  	• Rerepetition: 受試者重複該動作幾次了(優化過後的版本)<br>

# data_preprocessing.ipynb
Ninapro Database 的資料遇處理練習<br>

# EMG訊號分析與分類預測 alpha 1
	• 目標: 辨識手指的十二個動作(簡單的單拇指彎曲)
	• Sliding window size: 260 samples
	• Sliding duration: 80 samples
	• EMG 特徵提取: 使用[1]中的提取方法，一個sliding window 內的EMG訊號做小波包轉換(WT)，使用db5作為小波包的基底函數，分解層數為四層，window內的原始訊號會被分為24個不同的頻帶(這部分不清楚，要再查)，並可以得到2^4=16個能量值，能量值計算如下:
		E(j,i) = sigma(ρs(n,j,k)^2)		
	  其中E(j, i)為第 j 層第 I 個節點的能量值，ρs(n,j,k)為小波包分解的係數，因總共有12個sEMG通道( i.e. EMG sensor有十二個)，故特徵矩陣為16x12(能量x信號通道)，並將矩陣內的元素全部進行歸一化為0~1的值。
	• Label 定義: 一個 Sliding window 底下最相關的stimulus值(1~12表示動作1~動作12，0表示未做動作)，也就是說判斷Sliding window裡面的stimulus值最多的為何(ex. 動作1有100個sample point，動作2有50個，則判定Label為動作1)
	• 模型架構: 兩層捲基層、三層全連接層(最後一層加softmax函數)
	• Loss function: Cross Entropy
	• 優化器: AdamOptimizer
