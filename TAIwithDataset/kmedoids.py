import numpy as np
import math
import random
import copy
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
def ecl_dist(x, y): #x and y should be numpy arrays
    return np.sqrt(sum(np.square(x - y)))

def nearest(curr, centList):#return the nearest medoid pt's index in centList and its distance
    res=math.inf
    ind=0
    for i in range (len(centList)):
        if ecl_dist(curr,centList[i])<res:
            res=ecl_dist(curr,centList[i])
            ind=i
    return res,ind

class kmedoids(object):
    def __init__(self, data):
        self.belongto = []  #Record the point belong to which centroid(index) labels
        self.dataSet = data
        self.centList=None
        self.clusMap=None
        self.iter=0
    def initCluster(self,dataSet,k):#Init the clusters.
        def initCentroid(num):
            index = random.sample(range(1, len(self.dataSet)), num)
            return self.dataSet[index, :]
        self.centList=initCentroid(k)
        cluster=[[cent]for cent in self.centList]
        for point in dataSet:
            res,ind=nearest(point,self.centList)
            self.belongto.append(ind)
            cluster[ind].append(point)
        self.clusMap=cluster

    def chooseCluster(self,newCentroids,dataSet):#Assign all points to medoids.
        cluster=[[cent] for cent in newCentroids]
        belong=[]
        for point in dataSet:
            self.iter+=1
            res,ind=nearest(point,newCentroids)
            belong.append(ind)
            cluster[ind].append(point)
        return cluster,belong
    def selectPoint(self, medoids, clusters):#Find a point not medoid
        return [pts for PTS in clusters for pts in PTS if pts not in medoids]

    def algo_PAM(self,k):#Partition Around Medoids
        self.initCluster(self.dataSet,k)
        optimalCost = 0

        for i in range(k):
            optimalCost +=sum([ecl_dist(point,self.centList[i]) for point in self.clusMap[i]])
        improve=True
        while improve:#When no improvements could be made, stop.
            improve=False
            reducedSet=self.selectPoint(self.centList,self.clusMap)
            for i in range(k):
                for point in reducedSet:
                    tmp_centList = copy.deepcopy(self.centList)
                    tmp_centList[i] = point
                    tmp_clusMap,tmp_label = self.chooseCluster(tmp_centList, self.dataSet)
                    currCost = 0
                    for j in range(k):
                        currCost += sum([ecl_dist(point, tmp_centList[j]) for point in tmp_clusMap[j]])
                        self.iter+=1
                    if currCost < optimalCost:#If there exist a potential change to optimal the cost, we have to do all the steps again.
                        #Or otherwise we could stop.
                        improve = True
                        optimalCost = currCost
                        self.centList = copy.deepcopy(tmp_centList)
                        self.clusMap = tmp_clusMap
                        self.belongto=tmp_label
        return optimalCost
def draw_scatter2D(x,y,centroids,labels,title):# Function used to plot. Specially for "example dataset".
    pyplot.scatter(x, y, c=labels)
    xValues = []
    yValues = []
    for centroid in centroids:
        xValues.append(centroid[0])
        yValues.append(centroid[1])
    pyplot.scatter(xValues, yValues, c="r", marker="p")

    pyplot.title(title)
    pyplot.show()

def draw_scatter(x, y, z, centroids, labels, title):# Function used to plot.
    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c=labels)
    xValues = []
    yValues = []
    zValues = []
    for centroid in centroids:
        xValues.append(centroid[0])
        yValues.append(centroid[1])
        zValues.append(centroid[2])
    ax.scatter(xValues, yValues, zValues, c="r", marker="p")
    # axis(Order: Z, Y, X)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    pyplot.title(title)
    pyplot.show()
def exampleKMED(k):
    dataset = np.loadtxt('exampledataset.txt')
    example=kmedoids(dataset)
    distances_sum=example.algo_PAM(k)
    Map=example.clusMap
    Label=example.belongto
    print(f"[EXAMPLE]number of clusters: {len(example.centList)}")
    print(f"[EXAMPLE]Times of iteration: {example.iter}")
    centroids=example.centList
    draw_scatter2D(dataset[:, 0], dataset[:, 1], centroids, Label,f"KMedoids: Example k={k}")

class kMeans(object):
    def __init__(self,data:np.array,kValue):
        self.cluster=[]
        self.centroids=None
        self.k=kValue
        self.dataSet=data
        self.iter=0
    def initCentroids(self):
        dimension=self.dataSet.shape[1]#For Iris dimension=4
        #print(self.dataSet)
        self.centroids=np.zeros((self.k,dimension))
        # Then we should select several positions as initial centroids.
        # Range of values in different dimensions
        maxiValues=[[]for i in range(dimension)]
        miniValues=[[]for i in range(dimension)]
        tmp=np.zeros((self.k,dimension))
        for i in range (dimension):
            maxiValues[i]=np.max(self.dataSet[:, i])
            miniValues[i]=np.min(self.dataSet[:, i])
            for j in range(self.k):
                tmp[j, i] = maxiValues[i] + (miniValues[i] - maxiValues[i]) * np.random.rand()
                self.iter += 1
        self.centroids=tmp

    def classify(self,centroids):#To calculate all points' distance to all centroids
        distancesAllCents = []

        for centPts in centroids:
            disancesSingle = []  #Stores the euclidean dist. from all points to a single centroid.
            for instances in self.dataSet:
                distance = ecl_dist(instances, centPts)
                disancesSingle.append(distance)
                self.iter += 1
            distancesAllCents.append(disancesSingle)

        clsy = np.argmin(distancesAllCents, axis=0)#Index of minimum distance centr.
        return clsy

        # Compare two results
    def clsy_change(self, new_clsy, clsy):
        changed = False
        for i in range(len(clsy)):
            self.iter += 1
            if clsy[i] != new_clsy[i]:
                changed = True
                break
        return changed

    def get_distances_sse(self, crowds, clsys):
        sse = 0.0  # 保存所有样本点到所有聚类中心的欧式距离，其维度为(k,n)
        for i in range(len(self.dataSet)):
            # sse += get_euclidean_distance(train_data[i], crowds[clsys[i]])
            sse += float(ecl_dist(self.dataSet[i], crowds[clsys[i]]))
        return sse

    def kmeanAlgo(self):
        def getNewCent():
            mapping=[[]for _ in range(self.k)]
            dimension = self.dataSet.shape[1]
            tmp = np.zeros((self.k,dimension))
            for i in range(len(self.dataSet)):
                group=self.cluster[i]
                mapping[group].append(self.dataSet[i])
                self.iter+=1
            for i in range(self.k):
                a=averLists(mapping[i])#Returns a np array.
                tmp[i]=a
                self.iter += 1
            return tmp,mapping
        self.initCentroids()
        self.cluster=self.classify(self.centroids)
        flag=True
        while flag:
            newCentroids,mapping=getNewCent()
            newClusters=self.classify(newCentroids)
            self.iter+=1
            if not self.clsy_change(newClusters,self.cluster):
                flag=False
            else:
                self.centroids=newCentroids
                self.cluster=newClusters


        return self.get_distances_sse(self.centroids,self.cluster),mapping

def averLists(originList):
    res=np.zeros(4)
    for lines in originList:
        res+=np.array(lines)
    return res/len(originList)