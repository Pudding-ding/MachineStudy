from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target
# print(y)

#将数据集拆分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)#random_state=42切分的数据不一样

# 创建一个逻辑回归对象,这里逻辑回归会根据我们的数据决定是用二分类还是多分类
# 逻辑回归到底是把多分类转化成了多个二分类，还是说使用的是softmax回归
lr = LogisticRegression(max_iter=1000)
# lr = LogisticRegression(multi_class='multinomial')   # softmax回归做多分类
# lr = LogisticRegression(multi_class='ovr')   # 分类转行成了多个二分类

#使用训练集训练模型
lr.fit(x_train, y_train)

#使用测试集进行预测
y_pred = lr.predict(x_test)

#打印模型的准确率
print('准确率: %.2f'% accuracy_score(y_test, y_pred))
