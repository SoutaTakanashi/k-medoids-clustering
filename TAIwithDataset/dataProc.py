import  pandas as pd, numpy as np
def processIris(path):# To process and extract data from dssp and stride. Returns: dataframe , list
    with open(path, 'r') as f:
        data = f.readlines()
    chart=[]
    for l in data:
        l=l.strip('\n')
        chart.append(l.split(','))
    df = pd.DataFrame(chart,columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
    res=[]
    labels=[]
    for instances in chart:
        res.append(instances[:len(instances)-1])
        labels.append(instances[-1])
    res=np.array([ [float(attr) for attr in inst] for inst in res])
    return res,labels,df

def processBank(path):# To process and extract data from dssp and stride. Returns: dataframe , list
    with open(path, 'r') as f:
        data = f.readlines()
    chart=[]
    for l in data:
        l=l.strip('\n')
        chart.append(l.split(','))
    df = pd.DataFrame(chart,columns=['variance', 'skewness', 'curtosis', 'entropy', 'class'])
    res=[]
    labels=[]
    for instances in chart:
        res.append(instances[:len(instances)-1])
        labels.append(instances[-1])
    res=np.array([ [float(attr) for attr in inst] for inst in res])
    return res,labels,df

def getLabel(l,cenIndex,Label): #Convert the number of cluster to a certain Label
    #We see what is the medoid's label(string) actually.
    dict=[l[i] for i in cenIndex]# The actual label(string) for a medoid.
    res = []
    labelList=[]
    for label in Label: #Actually it is not used then.
        for i in range(len(dict)):
            if label==dict[i]:
                res.append(i)
                """
        if label == dict[0]:
            res.append(0)
        elif label == dict[1]:
            res.append(1)
        else:
            res.append(2)"""
    for i in range(len(l)):# It means for all points
        labelList.append(dict[Label[i]])# Label is the prediction labels(number)
    return res,labelList

def accuracy(true_labels,label_pred,cenIndex):

    label_ori_num,predStrings=getLabel(true_labels,cenIndex,label_pred)
    print("Original Labels: ",true_labels)
    print("Prediction: ",predStrings)
    cnt=0
    for i in range (len(predStrings)):
        if true_labels[i]==predStrings[i]:
            cnt+=1
    print(f"The accuracy of classifying iris is {100*cnt/len(label_pred)}%")
    return cnt/len(label_pred)