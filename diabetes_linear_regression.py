from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#加载糖尿病数据集
diabetes = datasets.load_diabetes()
x = diabetes.data
y = diabetes.target

#将数据集拆分为训练集和测试集
#拆分的20%是训练集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#创建一个多元线性回归算法对象
lr = LinearRegression()

#使用训练集训练模型
lr.fit(x_train, y_train)

#使用测试集进行预测
y_pred_train = lr.predict(x_train)
y_pred_test = lr.predict(x_test)

#打印模型的均方差
print('均方误差: %.2f'%mean_squared_error(y_train, y_pred_train))  #ctrl＋D直接复制粘贴
print('均方误差: %.2f'%mean_squared_error(y_test, y_pred_test))