'utf-8'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')


class K_means :
    def __init__(self, path):
        self.datapath = path
        #设置K，误差，次数
        self.K = 3
        self.ess = 0
        self.ep = 0

    def fit(self) :
        data = pd.DataFrame(np.loadtxt(self.datapath))
        C = self.Pre_C(data) #聚类中心
        data = self.ToClass(data,C)
        preC = C
        C = self.Lable_C(data)
        self.FitAgain(preC,C,data)

    def FitAgain(self,preC,C,data):
        flag = 0
        for i in range(self.K) :
            t = C[i] - preC[i]
            t = np.sqrt(t[0]**2 + t[1]**2)
            if t>self.ess :
                flag = 1
                break
        if flag == 1 : #不满足误差条件，需要继续迭代
            self.ep += 1
            print('第{}次迭代后的聚类中心：'.format(self.ep))
            for i in range(self.K) :
                print(C[i])
            preC = C
            data = self.ToClass(data,C)
            C = self.Lable_C(data)
            self.FitAgain(preC, C, data)
        if flag == 0 : #满足误差条件，结束迭代
            print('共迭代了{}次，最终聚为{}类的聚类中心：'.format(self.ep,self.K))
            for i in range(self.K) :
                print(C[i])
            #结果可视化
            plt.figure(figsize=(15, 8), dpi=80)
            data_x = list(data[0])
            data_y = list(data[1])
            lables = list(data['lable'])
            for k in range(self.K) :
                kx = []
                ky = []
                for i in range(len(lables)):
                    if lables[i] == k :
                        kx.append(data_x[i])
                        ky.append(data_y[i])
                plt.scatter(kx,ky,label=k,s=7)
            # plt.scatter(C[0][0],C[0][1],marker='x',s=100)
            # plt.scatter(C[1][0],C[1][1], marker='x', s=100)
            # plt.scatter(C[2][0],C[2][1], marker='x', s=100)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.title(" Kmeansclassfication_VOC")
            plt.show()

    #二维数据的一次聚类
    def ToClass(self,data,C) :
        data_x = list(data[0])
        data_y = list(data[1])
        lable_list = []
        for x, y in zip(data_x, data_y):
            lable = None
            dis_pare = float('inf')
            for i in range(len(C)):
                t = np.array([x, y]) - C[i]
                dis_2 = t[0] ** 2 + t[1] ** 2
                if dis_2 < dis_pare:
                    dis_pare = dis_2
                    lable = i
            lable_list.append(lable)
        data['lable'] = lable_list
        return data

    #初始聚类中心
    def Pre_C(self,data) :
        self.ep = 0
        C = data.sample(n=self.K, replace=False, axis=0)
        C_list = []
        print('随机选取的初始聚类中心：')
        for x,y in zip(list(C[0]),list(C[1])) :
            C_list.append(np.array([x,y]))
            print('[{} {}]'.format(x,y))
        return C_list

    def Lable_C(self,data_havelable):
        data_x = list(data_havelable[0])
        data_y = list(data_havelable[1])
        lables = list(data_havelable['lable'])
        C = []
        Count = []
        for i in range(self.K) :
            C.append(np.array([0.,0.]))
            Count.append(lables.count(i))
        for x,y,lable in zip(data_x,data_y,lables) :
            C[lable] += np.array([x,y])
        for i in range(self.K):
            C[i] = C[i]/Count[i]
        return C


if __name__ == "__main__" :
    classfier = K_means(r'.\data\classes.txt')
    classfier.fit()




