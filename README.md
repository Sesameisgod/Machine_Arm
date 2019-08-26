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
