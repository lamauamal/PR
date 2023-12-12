"utf-8"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

class Bayes :
    def __init__(self, path):
        self.datapath = path
        # 预设的先验概率
        self.P_f = 0.5 #0.5, 0.75
        self.P_m = 0.5 #0.5, 0.25
        # 定义损失函数L
        self.l12 = 0.3  # 男错分为女承担的风险
        self.l21 = 0.7  # 女错分为男承担的风险

    def fit(self):
        df_f = np.loadtxt(r'./dataset/FEMALE.TXT')
        df_m = np.loadtxt(r'./dataset/MALE.TXT')
        data_f = pd.DataFrame(df_f, columns = ['h', 'w'])
        data_m = pd.DataFrame(df_m, columns=['h', 'w'])
        len_f = len(data_f)
        len_m = len(data_m)
        h_f, w_f = list(data_f['h']), list(data_f['w'])
        h_m, w_m = list(data_m['h']), list(data_m['w'])

        #u, o2, E
        E_f, E_m = np.mat([[0,0],[0,0]])
        self.u_fh, self.u_fw = np.mean(h_f), np.mean(w_f)
        self.u_mh, self.u_mw = np.mean(h_m), np.mean(w_m)
        self.o2_fh, self.o2_fw = np.var(h_f), np.var(w_f)
        self.o2_mh, self.o2_mw = np.var(h_m), np.var(w_m)
        self.u_f = np.mat([[self.u_fh], [self.u_fw]])
        self.u_m = np.mat([[self.u_mh], [self.u_mw]])
        #E
        for i in range(len_f):
            x = np.mat([[df_f[i][0]], [df_f[i][1]]])
            E_f = (x - self.u_f) * (x - self.u_f).T + E_f
        self.E_f = E_f / len_f
        for i in range(len_m) :
            x = np.mat([[df_m[i][0]], [df_m[i][1]]])
            E_m = (x - self.u_m) * (x - self.u_m).T + E_m
        self.E_m = E_m / len_m

    # 判别函数 ：
    def g_1(self, x, u, o2, P):  # 最小错误率单变量
        return -0.5*(x-u)**2/o2 + np.log(P)- 0.5*np.log(o2)

    def g_2(self, x, u, E2, P) : #最小错误率二元变量
        return -0.5*(x-u).T*np.linalg.inv(E2)*(x-u)-0.5*np.log(np.linalg.det(E2)) + np.log(P)

    def l_1(self, l, x, u, o2, P): #最小风险单变量，多了损失系数l,
        return -0.5*(x-u)**2/o2 + np.log(P)- 0.5*np.log(o2) + np.log(l)

    def l_2(self, l, x, u, E2, P) : #最小风险二元变量
        return -0.5*(x-u).T*np.linalg.inv(E2)*(x-u)-0.5*np.log(np.linalg.det(E2)) + np.log(P)+np.log(l)

    def predict(self):
        data = np.loadtxt(self.datapath)
        columns = ['height', 'weight', 'gender']
        data = pd.DataFrame(data, columns = columns)
        self.data_label = list(data['gender'])
        data_label = self.data_label
        data.drop('gender', axis=1)
        self.data_h = list(data['height'])
        self.data_w = list(data['weight'])
        result = []
        #二元变量
        for i, j in zip(data['height'], data['weight']):
            x = np.mat([[i], [j]])
            gender_f = self.g_2(x, self.u_f, self.E_f, self.P_f)
            gender_m = self.g_2( x, self.u_m, self.E_m, self.P_m)
            if gender_f > gender_m:
                result.append(2)
            else:
                result.append(1)

        #单变量：h
        # for i in data['height']:
        #     gender_f = self.l_1(self.l12, i, self.u_fh, self.o2_fh, self.P_f)
        #     gender_m = self.l_1(self.l21, i, self.u_mh, self.o2_mh, self.P_m)
        #     if gender_f > gender_m:
        #         result.append(2)
        #     else:
        #         result.append(1)

        #单变量：w
        for i in data['weight']:
            gender_f = self.l_1(self.l12, i, self.u_fw, self.o2_fw, self.P_f)
            gender_m = self.l_1(self.l21, i, self.u_mw, self.o2_mw, self.P_m)
            if gender_f > gender_m:
                result.append(2)
            else:
                result.append(1)

        self.result = result
        y_f, y_m = 0, 0
        for pre, label in zip(result, data_label) :
            if pre == label :
                if label == 2 :
                    y_f = y_f + 1
                else:
                    y_m = y_m + 1
        count_f = data_label.count(2)
        count_m = data_label.count(1)
        acc_f = y_f/count_f #女性的预测准确率
        acc_m = y_m/count_m #男性的预测准确率
        err_f = 1 - acc_f #女性预测错误率
        err_m = 1 - acc_m #男性预测错误率
        acc = (y_m+y_f)/(len(data_label))
        err = (count_m+count_f-y_f-y_m)/(count_m+count_f)
        return acc_f,acc_m,acc,err_f,err_m,err

    def pl(self):
        plt.figure(figsize=(15, 8), dpi = 80 )
        w_m, w_f, w_fm, w_mf, m, f, fm, mf = [], [], [], [], [], [], [], []
        for h, w, label, pre in zip(self.data_h, self.data_w, self.data_label, self.result) :
            if pre == label : #决策正确
                if label == 1 :
                    m.append(h) #男性
                    w_m.append(w)
                else:
                    f.append(h) #女性
                    w_f.append(w)
            else: #决策错误
                if label == 2 and pre == 1 :
                    fm.append(h) #女错分为男
                    w_fm.append(w)
                else:
                    mf.append(h) #男错分为女
                    w_mf.append(w)
        #二元变量
        #y_m, y_f, y_fm, y_mf = w_m, w_f, w_fm, w_mf
        #单变量：h
        # y_m = np.zeros(len(m))
        # y_f = np.zeros(len(f))
        # y_fm = np.zeros(len(fm))
        # y_mf = np.zeros(len(mf))
        #单变量：w
        y_m = w_m[:]
        y_f = w_f[:]
        y_fm = w_fm[:]
        y_mf = w_mf[:]
        m, f, fm, mf = np.zeros(len(y_m)), np.zeros(len(y_f)), np.zeros(len(y_fm)), np.zeros(len(y_mf))

        plt.scatter(m, y_m, color='#0000FF', label='male', s=5)
        plt.scatter(f, y_f, color='#FF0000', label='female', s=5)
        plt.scatter(fm, y_fm, color='#FFA500', label='error:female to male', s=5)
        plt.scatter(mf, y_mf, color='#008000', label='erro:male to female', s=5)
        plt.title("Bayesclassfication_gender")
        plt.xlabel('height')
        plt.ylabel('weight')
        plt.legend()
        plt.show()



if __name__ == "__main__":
    classfier = Bayes(r'./dataset/test1.txt')
    classfier.fit()
    acc_f, acc_m, acc,err_f, err_m, err = classfier.predict()
    print("acc_f:{}\nacc_m:{}\nacc:{}\n"
          "err_f:{}\nerr_m:{}\nerr:{}".
          format(acc_f, acc_m, acc,err_f, err_m, err))
    classfier.pl()


