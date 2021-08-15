import kmedoids
import dataProc
import KSelect
IRIS='iris.data'
BankNote='data_banknote_authentication.txt'


def workFlow(dataSet,dataFrame=True,name="Iris"):
    if name=="Iris":
        chart,true_labels,df=dataProc.processIris(dataSet)
    else:
        chart, true_labels, df = dataProc.processBank(dataSet)
    if dataFrame==True:
        print(df)
    x=[x[0]for x in chart]
    y=[y[1]for y in chart]
    z=[z[2]for z in chart]
    q=[q[3]for q in chart]

    example=kmedoids.kmedoids(chart)
    if name=="Iris":
        example.algo_PAM(3)
    else:
        example.algo_PAM(2)
    Label = example.belongto
    medoids=example.centList
    medoids=medoids.tolist()
    listChart=chart.tolist()
    cenIndex=[listChart.index(m)for m in medoids]# The index of the medoids in entire data set.
    #print(Label)
    if name == "Iris":
        print(f"[IRIS]number of clusters: {len(example.centList)}")
        print(f"[IRIS]Times of iteration: {example.iter}")
    else:
        print(f"[BankNote]number of clusters: {len(example.centList)}")
        print(f"[BankNote]Times of iteration: {example.iter}")

    dataProc.accuracy(true_labels,Label,cenIndex)

    def draw(attr1,attr2,attr3):
        kmedoids.draw_scatter(attr1, attr2, attr3, medoids, Label,f"{name}")

    draw(x, q, z)
"""Please run these lines seperately."""
"""different part of experiment"""
#1.The example data set(40).
#kmedoids.exampleKMED(10)
#2.Iris data set(150).
workFlow(IRIS,False,"Iris")
#3.Banknote data set(more than 1000).
#workFlow(BankNote,False,"BankNote")
#KSelect.KHunt(BankNote)
def kmeansDemo():
    chart, true_labels, df = dataProc.processIris(BankNote)
    x = [x[0] for x in chart]
    y = [y[1] for y in chart]
    z = [z[2] for z in chart]
    q = [q[3] for q in chart]

    example = kmedoids.kMeans(chart, 2)

    sse, allCluster = example.kmeanAlgo()
    labelMean = list(example.cluster)
    listChart = chart.tolist()
    kmedoids.draw_scatter(x, y, z, example.centroids, labelMean, "KMeans")
    print(f"number of clusters:{example.k}")
    print(f"times of iterations:{example.iter}")

"""Part:compare with k-means"""
#kmeansDemo()

"""#Data collected from the output of first dataset. Times of iteration under different k value.
x=[i for i in range(2,11)]
y=[6384,9546,19008,23625,28152,32571,36864,41013,45000]
plt.plot(x,y,marker='o')
plt.title("Size of data set:40")
plt.xlabel("Number of clusters")
plt.ylabel("Times of iteration (Median)")
plt.show()
"""
