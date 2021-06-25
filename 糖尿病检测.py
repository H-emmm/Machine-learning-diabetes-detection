import pandas as pd
import numpy as np
import math
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score
from keras.models import Sequential
from keras.layers import Dense


def draw_scatter(dataframe, col):
    """
    绘制DataFrame型数据对应散点图分布

    dataframe:导入的DataFrame型数据
    col:对应属性的数组
    """
    d_list = np.array(dataframe)
    plt.figure(1)
    for i in range(len(col)):
        plt.subplot(1, len(col), i+1)  # 绘制在同一张图内，图内一共有1×len(col)（即每行图片和每列图片）张图
        plt.scatter(range(d_list.shape[0]), d_list[:, i], s=10)
        plt.title(col[i])
    plt.show()


class Diabetes(object):
    """
    糖尿病检测——LDA,支持向量机,神经网络
    """

    def __init__(self):
        self.diabetes = pd.read_csv("pima-indians-diabete.csv")  # 读取数据

    def pretreatment_missing(self):
        """
        数据预处理_异常值缺失处理
        """
        col_total = np.array(self.diabetes.columns)  # 将属性转为数组
        draw_scatter(self.diabetes[col_total], col_total)  # 绘图，分析异常缺失数据
        a = self.diabetes[col_total]
        col = ['Glucose', 'BloodPressure', 'SkinThickness', "Insulin", 'BMI', 'Age']  # 异常为0的数据
        draw_scatter(self.diabetes[col], col)  # 单独绘制，便于对比

        imr = SimpleImputer(missing_values=0, strategy="mean")  # 初始化 利用平均数填充异常值 异常值为0 自动对列进行计算
        di_list = np.array(self.diabetes[col])  # 将存在异常值的属性的DataFrame格式转为数组
        new_di_list = imr.fit_transform(di_list)  # fit训练数据，transform将训练好的数据拟合到原来的数据中

        self.diabetes[col] = new_di_list  # 将数组返回
        draw_scatter(self.diabetes[col], col)  # 绘制处理后的散点图，对比

    def pretreatment_separation(self):
        """
        数据预处理_数据分集

        self.train_x:训练集
        self.train_y:
        elf.validation_x:验证集
        self.test_x:
        self.validation_y:测试集
        self.test_y:
        """
        col_total = np.array(self.diabetes.columns)
        self.train_x, x, self.train_y, y = train_test_split(self.diabetes[col_total[:8]], self.diabetes[col_total[-1]],
                                                            test_size=0.4, shuffle=True)  # 分训练集和（测试集，验证集）
        self.validation_x, self.test_x, self.validation_y, self.test_y = train_test_split(x, y, test_size=0.5, shuffle=True)
        # 分测试集和验证集

    def pretreatment_standardization(self):
        """
        数据预处理_标准化
        """

        sc = StandardScaler()
        self.train_x = sc.fit_transform(self.train_x)
        self.test_x = sc.transform(self.test_x)
        self.validation_x = sc.transform(self.validation_x)

    def lda(self):
        """
        LDA预测
        """
        lda = LDA(n_components=1)
        train_x, train_y, test_x, test_y = self.train_x, self.train_y, self.test_x, self.test_y
        train_x_lda = lda.fit_transform(train_x, train_y)  # 训练模型
        test_x_lda = lda.transform(test_x)  # 拟合数据

        lr = LogisticRegression()  # 逻辑回归预测
        lr = lr.fit(train_x_lda, train_y)
        pre = lr.predict(test_x_lda)
        print("LDA正确率为:", accuracy_score(test_y, pre))
        print("\n")

    def svm(self):
        """
        支持向量机预测
        """
        train_x, train_y, test_x, test_y, validation_x, validation_y = \
            self.train_x, self.train_y, self.test_x, self.test_y, self.validation_x, self.validation_y
        parameter = {"kernel": ("linear", "rbf"), "C": np.arange(0.1, 1, 0.1), "gamma": np.arange(0.001, 0.01, 0.001)}
        svm = SVC(kernel="linear")
        g = GridSearchCV(svm, parameter)  # 网格搜索，寻找最佳超参数
        g.fit(train_x, train_y)
        print("C", g.best_estimator_.C)
        print("kernel", g.best_estimator_.kernel)
        print("gamma", g.best_estimator_.gamma)
        pre = g.predict(test_x)
        print("支持向量机正确率为:", accuracy_score(test_y, pre))
        print("\n")

    def ann(self, n_input, n_out):
        """
        神经网络预测
        """
        train_x, train_y, test_x, test_y, validation_x, validation_y = \
            self.train_x, self.train_y, self.test_x, self.test_y, self.validation_x, self.validation_y
        acc = []
        acc_1 = []
        n_hidden_best = 0
        for j in range(2):
            for i in range(1, 11):
                model = Sequential()
                n_hidden = int(math.pow((n_input + n_out), 0.5) + i)
                if n_hidden_best:
                    n_hidden = n_hidden_best
                epochs = 100  # 迭代次数
                batch_size = 50  # 小批量训练
                model.add(Dense(units=n_hidden, activation='sigmoid', input_shape=(n_input,)))  # 隐含层
                model.add(Dense(units=n_out, activation='sigmoid'))  # 输出层
                model.compile(loss='mean_squared_error', optimizer="sgd", metrics=['accuracy'])  # 配置模型

                history = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs,
                                    validation_data=(validation_x, validation_y))  # 训练
                if n_hidden_best:
                    acc_best = accuracy_score(test_y, model.predict_classes(test_x))
                    print("最佳隐含层节点数为:", n_hidden_best)
                    print("正确率为:", acc_best)
                    break
                acc.append(history.history["val_acc"][-1])
                acc_1.append(accuracy_score(test_y, model.predict_classes(test_x)))
            n_hidden_best = np.argmax(acc)+4


c = Diabetes()
c.pretreatment_missing()
c.pretreatment_separation()
c.pretreatment_standardization()
c.lda()
c.svm()
c.ann(8, 1)
