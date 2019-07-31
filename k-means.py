import iris_data
import numpy as np
import pandas as pd
import copy

f = open('C:/联创/机器学习/数据集/iris.csv')
content = pd.read_csv(f)
df = pd.DataFrame(content)

# 转为可处理数据
f1 = df['Sepal.Length'].values
f2 = df['Sepal.Width'].values
f3 = df['Petal.Length'].values
f4 = df['Petal.Width'].values
X = np.array(list(zip(f1, f2, f3, f4)))

# 定义距离函数
def L0 (point1,point2):
    L = np.sqrt(np.sum((point1 - point2)* (point1 - point2)))
    return L
k = 3
# 随机获取坐标
r_f1 = np.random.randint(np.min(X[...,0]), np.max(X[...,0]), size=k)
r_f2 = np.random.randint(np.min(X[...,1]), np.max(X[...,1]), size=k)
r_f3 = np.random.randint(np.min(X[...,2]), np.max(X[...,2]), size=k)
r_f4 = np.random.randint(np.min(X[...,3]), np.max(X[...,3]), size=k)
C = np.array(list(zip(r_f1, r_f2, r_f3, r_f4)), dtype=np.float32) 


# 保存中心点改变的坐标
C_ex = np.zeros(C.shape)
print(C)
# 用于保存数据所属中心点
c = np.zeros(len(X))
# 迭代标识位，通过计算新旧中心点的距离
iteration_flag = dist(C, C_ex, 1)
x = 1
#循环，得出中心点
while iteration_flag.any() and x <= 100:
	for i in range(len(X)):
		distances = dist(X[i], C, 1)
		cluster = np.argmin(distances) 
		clusters[i] = cluster
     C_ex = copy.deepcopy(C)

	 for x in range(k):
		points = [X[j] for j in range(len(X)) if clusters[j] == x]
		if len(points) != 0:
			C[x] = np.mean(points, axis=0)

# 计算新的距离
	print ('循环第%d次' % x)
	x += 1
	iteration_flag = dist(C, C_ex, 1)
	print("新中心点与旧点的距离：", iteration_flag)
