'utf-8'

import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import numpy as np
import pandas as pd

df_f = np.loadtxt(r'./dataset/FEMALE.TXT')
df_m = np.loadtxt(r'./dataset/MALE.TXT')
df_f = pd.DataFrame(df_f,columns=['h','w'])
df_m = pd.DataFrame(df_m,columns=['h','w'])
fh, fw = list(df_f['h']), list(df_f['w'])
mh, mw = list(df_m['h']), list(df_m['w'])
#训练样本数据分析
#散点图
fig = plt.figure(figsize=(15, 8), dpi=80)
p1 = fig.add_subplot(131)
plt.scatter(fh, fw, c='r', s=15, label='female')
plt.scatter(mh, mw, c='b', s=15, label='male')
plt.title("train_data")
plt.xlabel('height')
plt.ylabel('weight')
plt.legend()
#直方图
p2 = fig.add_subplot(132)
plt.hist([fh,mh], color=["r","b"], label=['female','male'])
plt.title("height")
plt.legend()
p3 = fig.add_subplot(133)
plt.hist([fw,mw], color=["r","b"], label=['female','male'])
plt.title("weight")
plt.legend()

plt.show()



