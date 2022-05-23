from sklearn.cluster import KMeans
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

def assement_result(labels,embeddings,k):
    origin_cluster = labels
    a = 0
    sum = 0
    while a < 10:
        clf = KMeans(k)
        y_pred = clf.fit_predict(embeddings)

        c = y_pred.T
        epriment_cluster = c ;
        NMI = metrics.normalized_mutual_info_score(origin_cluster, epriment_cluster)
        sum = sum + NMI
        a = a + 1
    average_NMI = sum / 10
    return average_NMI

