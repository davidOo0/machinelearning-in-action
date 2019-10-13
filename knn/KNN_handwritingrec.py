from numpy import *
import os
import operator
import matplotlib.pyplot as plt
import matplotlib


def creatdataset():
    group = array([[1,0],[1,0],[0,0],[0,1]])
    labels = ['A','A','B','B']
    return group,labels
##*****************************************##    
##分类函数
# 输入参数 ：
#           inX 用于分类的输入向量
#           dataset 训练样本集
#           label标签向量，数目与训练样本的行数一样
#           k 用于选择最近邻居数目
def clasify0(inX,dataset,label,k):
    ##使用欧氏距离计算距离
    datasetSize = dataset.shape[0] ##计算行数 shape[0]计算第一维
     #numpy中shape[0]返回数组的行数，shape[1]返回列数
    diffmat = tile(inX,(datasetSize,1))-dataset##将一个数组重复n次   x-x1
    #将intX在横向重复dataSetSize次，纵向重复1次
    #例如intX=([1,2])--->([[1,2],[1,2],[1,2],[1,2]])便于后面计算
    sqDiffmat= diffmat**2   ##(x-x1)^2

    #计算距离
    sqDistances=sqDiffmat.sum(axis=1)##axis=1 按行相加 axis=0按列相加 ——(x-x1)^2 +(y-y1)^2.....欧式距离
    distances= sqDistances**0.5
    #排序
    sortedDistindicies = distances.argsort() #排序 ，返回的是索引值
    classCount = {}##字典类型
    #提出前K个
    for i in range(k):
        votelabel = label[sortedDistindicies[i]]
        classCount[votelabel]=classCount.get(votelabel,0)+1 #计算各类型出现最多次的
    #计算类别 降序排序字典
    #key=operator.itemgetter(1)根据字典的值进行排序
    #key=operator.itemgetter(0)根据字典的键进行排序
    #reverse降序排序字典
    sortedClassCount = sorted(classCount.items() ,key = operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    array0line = fr.readlines()
    number0lines = len(array0line)#readines 获取行数
    returnmat = zeros((number0lines,3))#生成一个返回矩阵 行：行数 列 ：3
    classlabelvector = []  #返回标签向量
    index = 0
    for line in array0line:
        line = line.strip()##删除头尾空格
        listfromline = line.split('\t') #按行划分数据
        returnmat[index,:] = listfromline[0:3] ##提取前三列
        classlabelvector.append(int(listfromline[-1])) ##-1最后一列元素
        index += 1
    return returnmat,classlabelvector

def autonorm(dataset):
    ##归一化数据集
    minvals = dataset.min(0)#每列的最小值
    maxvals = dataset.max(0)#每列最大值
    ranges = maxvals - minvals
    m = dataset.shape[0]
    normdataset = dataset - tile(minvals,(m,1))
    normdataset = normdataset/tile(ranges,(m,1))
    return normdataset,ranges,minvals

def datingclasstest():
    #错误率测试
    horatio = 0.31
    datingdatamat,datinglabels = file2matrix('C:\\Users\\WZF\\Desktop\\test\\机器学习实战\\datingTestSet2.txt')
    normmat,ranges,minvals =autonorm(datingdatamat)
    m=normmat.shape[0]
    numtestvecs = int(m*horatio)
    errorcount = 0.0
    for i in range(numtestvecs):
        classfierresult = clasify0(normmat[i,:],normmat[numtestvecs:m,:],datinglabels[numtestvecs:m],3)
        print("分类结果 %d, 真实结果 %d" %(classfierresult,datinglabels[i]))
        if(classfierresult != datinglabels[i]):
            errorcount+=1.0
    print("错误率 %f"%(errorcount/float(numtestvecs)))


def img2vector(filename):
    #32*32的图像矩阵转换成1*1024的向量
    
    returnvect = zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        linestr = fr.readline()
        for j in range(32):
            returnvect[0,(32*i+j)] = int (linestr[j])
    return returnvect
def handwritingclasstest():
    hwlabel = []
    trainingfilelist = os.listdir('C:\\Users\\WZF\\Desktop\\test\\机器学习实战\\K-临近算法\\trainingDigits')
    m=len(trainingfilelist)
    trainingmat = zeros((m,1024))
    for i in range(m):
        filenamestr = trainingfilelist[i]
        filestr = filenamestr.split('.')[0]
        classnumstr=int(filenamestr.split('_')[0])
        hwlabel.append(classnumstr)
        trainingmat[i,:] = img2vector('C:\\Users\\WZF\\Desktop\\test\\机器学习实战\\K-临近算法\\trainingDigits\\%s'%(filenamestr))
    testfilelist = os.listdir('C:\\Users\\WZF\\Desktop\\test\\机器学习实战\\K-临近算法\\testDigits')
    errorcount = 0.0 
    mtest = len(testfilelist)
    for i in range(mtest):
        filenamestr = testfilelist[i]
        filestr = filenamestr.split('.')[0]
        classnumstr=int(filenamestr.split('_')[0])
        vectorundertest= img2vector('C:\\Users\\WZF\\Desktop\\test\\机器学习实战\\K-临近算法\\testDigits/%s'%filenamestr)
        classifierresult = clasify0(vectorundertest,trainingmat,hwlabel,3)
        print("分类结果：%d 实际结果 %d"%(classifierresult,classnumstr))
        if(classifierresult != classnumstr):
            errorcount+=1.0
    print("错误率 %f"%(errorcount/float(mtest)))