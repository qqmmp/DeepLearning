import pickle
import numpy
import os
from PIL import Image
import matplotlib.pyplot as pyplot

#导入一个包
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        #x = dict.keys()
        #y = dict.values()
        #x = numpy.array(x)
        #y = numpy.array(y)
        #print(dict[b'data'])
        x = dict[b'labels']#标签数组
        y = dict[b'data']#数据数组
        #y = y.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    return x,y

#导入所有文件
def unpickle_all(ROOT):
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        x,y = unpickle(f)
        xs.append(x)
        ys.append(y)
    xtr = numpy.concatenate(xs)#标签数组
    ytr = numpy.concatenate(ys)#数据数组
    del x,y
    xte,yte = unpickle(os.path.join(ROOT, 'test_batch'))#测试用的数据及其标签数组
    return xtr,ytr,xte,yte

#曼哈顿距离
def distance(x,y):
    return sum(abs(x-y))

#欧式距离
def distance_o(x,y):
    return numpy.sqrt(((x-y)**2))

#第一种方法：用曼哈顿距离或者欧式距离计算近邻
def distance_all(xtr, ytr, xte, yte, num1):#num1表示进行测试的图片数量
    mindis = 1000000
    for i in range(0, num1):#检索测试集
        for j in range(0, 5000):#在训练集中找最合适的
            dis = distance_o(ytr[j], yte[i])
            if dis < mindis:
                mindis = dis
                minlable = xtr[j]
                minsta = j
    return mindis, minlable,minsta

#多类SVM损失（传入一张图片，和w）
def hingle_loss(data,w,i):
    Li_all = 0
    scores = w.dot(data)
    for j in range(10):
        if j != xtr[i]:
            Li = max(0, scores[j] - scores[xtr[i]])
            Li_all += Li
    Li_all = (numpy.sum(w ** 2) + Li_all) / 10
    return (Li_all)

#基础优化方式
def optimize1(data, w, num):#传入参数：一张图片的数据，w，该图片的位置（用于确定图片类型）
    n = 0
    while n < 5:#收敛状态应该调整n
        for i in range(10):
            for j in range(3072):
                loss1 = hingle_loss(data, w, num)
                w[i][j] += 0.0001
                loss2 = hingle_loss(data, w, num)
                dw = (loss2 - loss1)/0.0001
                w[i][j] += -0.5*dw#步长应该变化
        n += 1
    return w

if __name__ == '__main__':
    path = r'E:\Code\PYTHON\DeepLearning\cifar-10-batches-py'
    file = 'data_batch_1'
    label = {1:'airplane',2:'automobile',3:'bird',4:'cat',5:'deer',6:'dog',7:'frog',8:'horse',9:'ship',10:'truck'}
    #xtr, ytr, xte, yte = unpickle_all(path)
    xtr, ytr = unpickle(file)
    min_batch = ytr[0:500]
    w = numpy.random.random([10, 3072])
    data = min_batch[0]
    x = optimize1(data , w, 0)
    n=0
    for j in range(500):
        for i in range(10):
            scores = x.dot(min_batch[j])
            if scores[i]>=scores[0]:
                minla = i
        if minla==xtr[j]:
            n+=1


    print(n/5,'%')



