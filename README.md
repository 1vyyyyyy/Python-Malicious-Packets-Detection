### 1.所有代码和文件说明

(1)multi_models.py: 用4种方法训练模型并保存至models文件夹.

(2)flow_analysis.py: 用models文件夹中的模型对prediction.csv中的数据进行预测，预测结果保存至"各模型预测结果".

(3)min_wireshark.py: 基于tkinter和scapy实现的模仿抓包工具wireshark的抓包平台，将所抓取的pcap包送至CICFlowMeter转换为.csv文件后，即可使用flow_analysis.py进行类型判断.

(4)train.csv: 训练集

(5)prediction.csv: 待预测的流量包的特征数据

(6)真实值与各模型预测结果.csv: 在multi_models.py中生成，用于表征训练好的模型的效果.

(7)各模型预测结果.csv: 在flow_analysis.py中生成，展示了4种模型对prediction.csv的类型预测结果.

(8)Model Comparison-Dataset(train)-f1-score.jpg: 在multi_models.py中生成,将模型效果可视化.

### 2.一些运行说明

(1)使用min_wireshark.py时，需要将32行的"INTERFACE"修改为自己电脑的网卡名称.

(2)若要重新训练模型，最好将models中的模型删除，将"真实值与各模型预测结果.csv"、"各模型预测结果.csv"、"Model Comparison-Dataset(train)-f1-score.jpg"删除，再依次运行multi_models.py、flow_analysis.py.

(3)若要使用新的流量包进行预测，可以先运行min_wireshark.py，获取pcap文件，再使用CICFlowMeter将pcap文件转换为.csv文件，再运行flow_analysis.py.

