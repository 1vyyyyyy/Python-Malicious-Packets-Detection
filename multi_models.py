import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# 读取数据
data = pd.read_csv('train.csv')
print(f'columns: {data.columns}, length: {len(data.columns)}')
# 打乱数据
data = data.sample(frac=1)

# 查看数据
data.head()

# 查看特征数据分布
print(data.info())

# 去除空值
data = data.dropna(axis=0)

# 去除重复值
data = data.drop_duplicates()

# 选择特征
# features = ['Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
#             'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max',
#             'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max',
#             'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',  'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total',
#             'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
#             'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
#             'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length',
#             'Max Packet Length', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
#             'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count',
#             'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
#             'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
#             'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets',
#             'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward',
#             'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Std',
#             'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']

features = ['Dst Port', 'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets', 'Total Length of Fwd Packet',
            'Total Length of Bwd Packet', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
            'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
            'Bwd Packet Length Std',  'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
            'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
            'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
            'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length',
            'Fwd Packets/s', 'Bwd Packets/s', 'Packet Length Min', 'Packet Length Max', 'Packet Length Mean',
            'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
            'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWR Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
            'Average Packet Size', 'Fwd Segment Size Avg', 'Bwd Segment Size Avg', 'Fwd Bytes/Bulk Avg',
            'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg', 'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg',
            'Bwd Bulk Rate Avg', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
            'FWD Init Win Bytes', 'Bwd Init Win Bytes', 'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean',
            'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']

# 选择y变量
target = 'Label'

# 选择有足够训练数据的label标签
data = data[data[target].isin(data[target].value_counts()[data[target].value_counts() > 0].index)].copy().reset_index(
    drop=True)
print(f"labels count: \n{data[target].value_counts()}")

# 将label标签转化成数值类型
from sklearn.preprocessing import LabelEncoder

lbEn = LabelEncoder()
data[target] = lbEn.fit_transform(data[target])

# 特征工程

# 选择X,y
X = data[features]
y = data[target]

# 归一化处理
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)

# 测试集30%训练集70%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.30, random_state=30)

# 模型

import numpy as np
import time


# 定义模型性能评估方法
def evaluation(y_test, y_pred, model_type='regression'):
    """
    传入真实值y_test和预测值y_pred，评估模型效果
    model_type = regression评估回归模型
    model_type = classification评估分类模型
    """
    metrics = {}
    # if classification model
    if model_type == 'classification':
        from sklearn.metrics import classification_report
        from sklearn.metrics import accuracy_score  # 正确率
        from sklearn.metrics import precision_score  # 精准率
        from sklearn.metrics import recall_score  # 召回率
        from sklearn.metrics import f1_score  # 调和平均值F1
        # metrics['cls_report'] = classification_report(y_test, y_pred)
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['precision'] = precision_score(y_test, y_pred, average='macro')
        metrics['recall'] = recall_score(y_test, y_pred, average='macro')
        metrics['f1-score'] = f1_score(y_test, y_pred, average='macro')

    # if regression model
    elif model_type == 'regression':
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import r2_score
        metrics['MSE'] = mean_squared_error(y_test, y_pred)
        metrics['RMSE'] = mean_squared_error(y_test, y_pred) ** 0.5
        metrics['MAE'] = mean_absolute_error(y_test, y_pred)
        metrics['R2'] = r2_score(y_test, y_pred)
    else:
        raise Exception('model_type should be classification or regression!')
    return metrics


def KNN_cls_model(X_train, y_train, X_test, y_test):
    # KNN模型
    time_start = time.time()
    from sklearn.neighbors import KNeighborsClassifier
    KNN = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski')
    KNN.fit(X_train, y_train)
    time_end = time.time()
    print(f"training time: {(time_end - time_start):.4f}s")
    y_pred_KNN = KNN.predict(X_test)
    # 性能评估
    metrics = evaluation(y_test, y_pred_KNN, model_type='classification')
    return KNN, metrics


# classification model
def DT_cls_model(X_train, y_train, X_test, y_test):
    time_start = time.time()
    # 建立决策树模型
    from sklearn.tree import DecisionTreeClassifier
    DT = DecisionTreeClassifier(max_depth=5, random_state=42)
    DT.fit(X_train, y_train)
    time_end = time.time()
    print(f"training time: {(time_end - time_start):.4f}s")
    y_pred_DT = DT.predict(X_test)
    # 性能评估
    metrics = evaluation(y_test, y_pred_DT, model_type='classification')
    return DT, metrics


# RF模型
def RF_cls_model(X_train, y_train, X_test, y_test):
    time_start = time.time()
    from sklearn.ensemble import RandomForestClassifier
    RF = RandomForestClassifier(n_estimators=10)
    RF.fit(X_train, y_train)
    time_end = time.time()
    print(f"training time: {(time_end - time_start):.4f}s")
    # 预测模型
    # 测试集预测
    y_pred_RF = RF.predict(X_test)
    # 性能评估
    metrics = evaluation(y_test, y_pred_RF, model_type='classification')
    return RF, metrics


# Xgboost模型
def Xgboost_cls_model(X_train, y_train, X_test, y_test):
    time_start = time.time()
    from xgboost import XGBClassifier
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    time_end = time.time()
    print(f"training time: {(time_end - time_start):.4f}s")
    # 预测模型
    # 测试集预测
    y_pred_RF = xgb.predict(X_test)
    # 性能评估
    metrics = evaluation(y_test, y_pred_RF, model_type='classification')
    return xgb, metrics


# 选择要训练的分类模型
cls_models = [DT_cls_model, RF_cls_model, Xgboost_cls_model, KNN_cls_model]
# cls_models = [DT_cls_model, RF_cls_model, KNN_cls_model]

# 模型类型为分类模型
model_type = 'classification'
eval_dic = {}
models = []

# 依次训练
for model in cls_models:
    print(f"【{model.__name__}】")
    model, metrics = model(X_train, y_train, X_test, y_test)
    models.append(model)
    # 得到评估结果
    eval_dic[type(model).__name__] = metrics.copy()
    print(f"finished {type(model).__name__} model training...")
    print("evaluation:")
    print(metrics)
    print('\n')

import joblib

for model in models:
    joblib.dump(model, f'./models/{type(model).__name__}.pkl')

import matplotlib.pyplot as plt


def plot_metric(metric_dic, metric='RMSE', dataset_name='None'):
    """
    metric_dic looks like: {'model_name':{'metric_name': value}}
    """
    # 画图，画出各个模型的指标对比
    x = []
    y = []
    # 提取数据
    for model_name in metric_dic.keys():
        x.append(model_name)
        y.append(metric_dic.get(model_name).get(metric))
    plt.figure(figsize=(10, 7))
    # 画柱形图
    plt.bar(x, y)
    for i, j in zip(range(len(x)), y):
        plt.text(i, j, '{:.4}'.format(j), va='bottom', ha='center')
    # 设置标题坐标轴名称
    plt.title(f"Model Comparison - Dataset({dataset_name}) - {metric}", fontsize=15)
    plt.xlabel("Model Name")
    plt.ylabel(metric)
    plt.xticks(rotation=0)
    plt.ylim(np.min(y) * 0.8, np.max(y) * 1.05)
    # 保存图片
    plt.savefig(f"Model Comparison - Dataset({dataset_name}) - {metric}.jpg", dpi=200)
    plt.show()


plot_metric(eval_dic, metric='f1-score', dataset_name='train')

# 使用各模型进行预测
df_pred = pd.DataFrame([lbEn.inverse_transform(model.predict(X_test)) for model in models],
                       index=[type(model).__name__ + "_Predict" for model in models]).T

df_pred['True Value'] = lbEn.inverse_transform(y_test)
# 保存结果
df_pred.to_csv("真实值与各模型预测结果.csv", index=None)

df_pred.sample(20)

