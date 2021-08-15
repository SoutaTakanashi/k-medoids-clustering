import dataProc
import kmedoids
from matplotlib import pyplot as plt
def KHunt(path):
    chart, true_labels, df = dataProc.processIris(path)
    example = kmedoids.kmedoids(chart)
    x=[]
    y=[]
    for k in range(1,int(len(chart)/100)):
        x.append(k)
        sumDistance=example.algo_PAM(k)
        y.append(sumDistance)
    plot(x,y)

def plot(x,y):
    plt.ylabel("SSE")
    plt.xlabel("number of clusters (k)")
    plt.plot(x,y)
    plt.show()