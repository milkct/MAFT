# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 假设你有一个CSV文件，包含两列：'Year' 和 'Expenditure'
data = pd.read_csv('army.csv')

# 将年份转换为一个数值特征，这是为了线性回归模型
# 例如，如果年份是2010，转换为10；如果年份是2020，转换为20
data['Year'] = data['Year'] - data['Year'].min()

# 将数据分为特征(X)和目标变量(y)
X = data['Year'].values.reshape(-1, 1)
y = data['Expenditure'].values

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测测试集的结果
y_pred = model.predict(X_test)

# 计算模型的均方误差
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# 使用模型预测未来一年的支出
next_year = np.array([[data['Year'].max() + 1]])
predicted_expenditure = model.predict(next_year)
print('2024年的军费为:', str(predicted_expenditure)+'亿元')

# 可视化历史数据和预测线
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red', linewidth=2)
plt.scatter(next_year, predicted_expenditure, color='green')
plt.show()
