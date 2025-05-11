import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, minmax_scale, MinMaxScaler

df = pd.read_csv('data_refined.csv')  ## already cleaned
print(df)

xbreast = df.drop(columns=['diagnosis'])
ybreast = df['diagnosis']

dfins = pd.read_csv('insurance.csv').dropna()
print(dfins.info())
label = LabelEncoder()
onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')

dfins['sex'] = label.fit_transform(dfins['sex'])
dfins['smoker'] = label.fit_transform(dfins['smoker'])

onehot_encode = onehot.fit_transform(dfins['region'].values.reshape(-1, 1))

xins = pd.concat([dfins.drop(columns=['region', 'charges']), onehot_encode], axis=1)
yins = dfins['charges']

print('K-Main Method:')

from sklearn.cluster import KMeans

distorsions = []
distorsions2 = []
thex = range(1, 100)

for i in thex:
    k = KMeans(n_clusters=i, tol=0.01, random_state=0)
    k.fit(xbreast)
    distorsions.append(k.inertia_)

    kins = KMeans(n_clusters=i, tol=0.01, random_state=0)
    kins.fit(xins)
    distorsions2.append(kins.inertia_)

plt.figure()
plt.plot(thex, distorsions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distorsions')
plt.title('Elbow Method - breast cancer')

plt.figure()
plt.plot(thex, distorsions2, 'ro-')
plt.xlabel('k')
plt.ylabel('Distorsions')
plt.title('Elbow Method - insurance')
plt.show()

print('From the elbow plots, the best K for both dataset is 10')
print('')

print('---------------------------------------------------------------------')
print('Main Shift Method:')
from sklearn.cluster import estimate_bandwidth

bandwidth = estimate_bandwidth(xbreast, quantile=0.15)
bandwidth2 = estimate_bandwidth(xins, quantile=0.15)
print('the optimum bandwidth for breast cancer data set is: ' + str(bandwidth))
print('the optimum bandwidth for insurance data set is: ' + str(bandwidth2))
from sklearn.cluster import MeanShift

m_breast = MeanShift(bandwidth=bandwidth).fit(xbreast)
m_insurance = MeanShift(bandwidth=bandwidth2).fit(xins)

clusters_meanshift = m_breast.predict(xbreast)
clusters_meanshift2 = m_insurance.predict(xins)

print("Unique CLusters for Beast Cancer Set is:", np.unique(clusters_meanshift))
print("Number of Unique Clustersfor Beast Cancer Set is:", len(np.unique(clusters_meanshift)))
print("Unique CLusters for Insurance Set is:", np.unique(clusters_meanshift2))
print("Number of Unique Clustersfor Insurenace Set is:", len(np.unique(clusters_meanshift2)))
print('')

print('---------------------------------------------------------------------')
print(
    'Yes, DBSCAN Can be used to cluster the data sets in this project. Here is an example of DBSCAN on Breast Cancer data set:')

from sklearn.cluster import DBSCAN

l = len(xbreast.transpose())

dbscan = DBSCAN(eps=0.25, min_samples=3)
clusters = dbscan.fit_predict(xbreast)

print("Unique CLusters for Beast Cancer Set (DBSCAN) is:", np.unique(clusters))
print("Number of Unique Clustersfor Beast Cancer Set (DBSCAN) is:", len(np.unique(clusters)))
