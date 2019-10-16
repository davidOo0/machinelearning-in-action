from math import log
import operator
import matplotlib.pyplot as plt
#全局变量
decisionNode = dict(boxstyle = "sawtooth",fc = "0.8")
leafNode = dict(boxstyle = "round4",fc = "0.8")
arrow_args = dict(arrowstyle = "<-")
#**************************
#创造测试数据集
#**************************
def creatdataset():
    dataset = [[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'maybe']]
    labels = ['no surfacing','flippers','fish']
    return dataset,labels

#***************************
#计算信息熵 
#***************************
def calshannoEnt(dataset):
    numentries = len(dataset)
    labelcounts= {}
    for featvec in dataset:
        currentlabel = featvec[-1]

        if currentlabel not in labelcounts.keys():
            labelcounts[currentlabel] = 0
        labelcounts[currentlabel] += 1
    shannoent = 0.0
    for key in labelcounts:
        prob = float(labelcounts[key])/numentries ##计算概率
        shannoent -= prob*log(prob,2)
    return shannoent 
#***********************************
#划分数据集 根据信息增益
#抽取符合要求的特征
#***********************************
def splitdataset(dataset,axis,value):
    retdataset = []
    for featvec in dataset:
        if featvec[axis] == value:
            reducefectvec = featvec[:axis]
            reducefectvec.extend(featvec[axis+1:])
            retdataset.append(reducefectvec)##append 和extend 的区别！！！
    return retdataset
#**************************************
#选择最好的划分方式
#遍历各种方式计算信息熵，取最大的
#**************************************
def choosebestfeaturetosplit(dataset):
    numfeatures = len(dataset[0])-1   ##计算属性个数
    baseent = calshannoEnt(dataset)  ##计算基础信息熵
    bestinfogain = 0.0        ##初始信息增益为0
    bestfeature = -1        ##初始最佳分类属性为-1
    for  i in range(numfeatures):
        featlist = [example[i] for example in dataset] #!!!取dataset 列
        uniquevals = set(featlist)    ##创建一个无重复的list（这个分类节点有多少分支）
        newentropy = 0.0 
        for value in uniquevals:
            subdataset = splitdataset(dataset,i,value)
            prob = len(subdataset)/float(len(dataset))
            newentropy += prob * calshannoEnt(subdataset)
        infogain = baseent - newentropy##计算信息增益的公式
        if(infogain > bestinfogain):
            bestinfogain = infogain
            bestfeature= i     ##z找到最大信息增益的分类属性 返回下标
    return bestfeature
#*******************************************************    
#叶子节点 如果存在多个类别，选择多的类别作为叶子节点
def majoritycnt(classlist):
    classcount={}
    for vote in  classlist:
        if(vote not in  classcount.keys):
            classcount[vote]= 0.0
        classcount += 1 
    sortedclasscount = sorted(classcount.items(),key=operator.itemgetter(1),reversed = True)#排序  #reverse=true 倒序排列 itemgetter（1）获取对象第一维的值
    return sortedclasscount[0][0]#返回最多的类的名称                                                       #dic.item()把字典键值变为元组数组

#-*******************************************************
#创建树
#*******************************************************
def creattree(datasheet,labels):
    classlist = [example[-1] for example in datasheet]
    if classlist.count(classlist[0]) == len(classlist) :
        return classlist[0]#只剩下一个种类时
    if( len(datasheet[0]) == 1):
#只剩下一个属性时
        return majoritycnt(classlist)
    bestfeat = choosebestfeaturetosplit(datasheet)
    bestlabels = labels[bestfeat]
    mytree = {bestlabels:{}} ##循环的一个字典
    del (labels[bestfeat])
    featvalue= [example[bestfeat] for example in datasheet]
    uniquevals = set (featvalue)
    for values in uniquevals:
        sublabels = labels[:]
        mytree[bestlabels][values] =creattree(splitdataset(datasheet,bestfeat,values),sublabels)
    return mytree 


##******************************
#使用matplotlib 绘制决策树
#******************************
def getnumberleafs(mytree):
    numleafs = 0
    firststr = list(mytree.keys())[0]
    print(firststr)
    seconddict = mytree[firststr]
    for key in seconddict.keys():
        if (type(seconddict[key]).__name__ =='dict'):
            numleafs += getnumberleafs(seconddict[key])
        else :
            numleafs += 1
    return numleafs
def gettreedepth(mytree):
    maxdepth = 0
    firststr = list(mytree.keys())[0]
    
    seconddict = mytree[firststr]
    for key in seconddict.keys():
        if (type(seconddict[key]).__name__=='dict'):
            thisdepth = 1+ gettreedepth(seconddict[key])
        else :
            thisdepth = 1
        if(thisdepth>maxdepth):
            maxdepth=thisdepth
        return maxdepth
def retrievetree(i):
    listoftree = [{'no surfacig':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}},{'no surfacing':{0:'no',1:{'flippers':{0:{'head':{0:'no',1:'yes'}},1:'no'}}}}]
    return listoftree[i]
def plotNode(nodetxt,centerpt,parentpt,nodetype):
    creatplot.ax1.annotate(nodetxt,xy = parentpt,xycoords='axes fraction',xytext = centerpt ,textcoords = 'axes fraction',va = "center",ha = "center",bbox =nodetype, arrowprops= arrow_args)
'''
def creatplot():
    fig = plt.figure(1,facecolor="white")
    fig.clf()
    creatplot.ax1 = plt.subplot(111,frameon = False)
    plotNode(U'决策节点',(0.5,0.1),(0.5,0.1),decisionNode)
    plotNode(U'叶节点',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()
'''
def plotmidtext(cntrpt,parentpt,txtstring):
    xmid= (parentpt[0]-cntrpt[0])/2.0 +cntrpt[0]
    ymid= (parentpt[1]-cntrpt[1])/2.0 +cntrpt[1]
    creatplot.ax1.text(xmid,ymid,txtstring)
def plottree(mytree,parentpt,nodetxt):
    numleafs  = getnumberleafs(mytree)
    depth = gettreedepth(mytree)
    firststr = list(mytree.keys())[0]
    cntrpt = (plottree.x0ff + (1.0+float(numleafs))/2.0/plottree.totalw,plottree.y0ff)
    plotmidtext(cntrpt,parentpt,nodetxt)
    plotNode(firststr,cntrpt,parentpt,decisionNode)
    seconddict = mytree[firststr]
    plottree.y0ff = plottree.y0ff-1.0/plottree.totald
    for key in seconddict.keys():
        if type(seconddict[key]).__name__=='dict':
            plottree(seconddict[key],cntrpt,str(key))
        else:
            plottree.x0ff = plottree.x0ff + 1.0/plottree.totalw
            plotNode(seconddict[key],(plottree.x0ff,plottree.y0ff),cntrpt,leafNode)
            plotmidtext((plottree.x0ff,plottree.y0ff),cntrpt,str(key))
    plottree.y0ff = plottree.y0ff+1.0/plottree.totald


def creatplot (intree):
    #绘制的主函数   
    fig = plt.figure(1,facecolor='white')
    fig.clf()  #清除axes
    axprop = dict(xticks= [], yticks= [])
    creatplot.ax1 = plt.subplot(111,frameon = False,**axprop)#给函数绑定成员
    plottree.totalw = float(getnumberleafs(intree))
    plottree.totald = float(gettreedepth(intree))
    plottree.x0ff = -0.6/plottree.totalw
    plottree.y0ff=1.2   
    plottree(intree,(0.5,1.0),'')#坐标
    plt.show()

#***************************************************************************************


def classify(inputtree,featlabel,testvec):
    classlabel=''
    firststr = list(inputtree.keys())[0]
    seconddict = inputtree[firststr]
    featindex = featlabel.index(firststr)
    for key in seconddict.keys():
        if testvec[featindex] == key:
            if type(seconddict[key]).__name__=='dict':
                classlabel = classify(seconddict[key],featlabel,testvec)
            else:
                classlabel = seconddict[key]
    return classlabel
##*********************************************************************
#将学习好的树存下来
#*********************************************************************
def storetree(inputtree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dumps(inputtree,fw)
    fww.close()
def grabtree (filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
fr=open('C:\\Users\\WZF\\Desktop\\test\\机器学习实战\\决策树\\lenses.txt')   
lenses = [inst.strip().split('\t') for inst in fr.readlines()] 
lenseslabels=['ages','prescript','astigmatic','tearrate']
lensestree = creattree(lenses,lenseslabels)
creatplot(lensestree)
