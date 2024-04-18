import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt

# Carregar dados
data_df = pd.read_csv('data.csv')
labels_df = pd.read_csv('labels.csv')

# Verificar valores nulos
#print(data.isnull().sum())
# Imputação com a média
#data_imputed = data.fillna(data.mean())

# Remover coluna de identificação(sample_x) e normalizar os dados
data_numeric = data_df.drop(columns=data_df.columns[0])
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data_numeric)

# plotar PCA
# pca = PCA().fit(data_normalized)
# plt.figure()
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('Number of Components')
# plt.ylabel('Variance (%)') # for each component
# plt.title('Explained Variance')
# plt.show()

# Aplicar PCA para reduzir a dimensionalidade
pca = PCA(n_components=2)  # Reduz para 70 componentes
data_pca = pca.fit_transform(data_normalized)

# Aplicar modelos de clustering
# K-Means
kmeans = KMeans(n_clusters=4)
labels_kmeans = kmeans.fit_predict(data_pca)

# Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=4)
labels_hierarchical = hierarchical.fit_predict(data_pca)

# DBSCAN
#dbscan = DBSCAN(eps=700, min_samples=5, algorithm='ball_tree', metric='minkowski', leaf_size=90, p=2)
dbscan = DBSCAN(eps=70, min_samples=50)
labels_dbscan = dbscan.fit_predict(data_pca)

# Função para plotar os resultados do clustering com centróides
def plot_clusters(data, labels, centroids=None, title="Clustering"):
    plt.figure(figsize=(10, 7))
    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Pontos pretos serão usados para representar ruído (pontos sem cluster atribuído no DBSCAN).
            col = 'k'

        class_member_mask = (labels == k)
        
        plt.scatter(data[class_member_mask, 0], data[class_member_mask, 1], s=50, c=col, label=f'Cluster {k}' if k != -1 else 'Noise')
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

# Plotar resultados para K-Means com centróides
plot_clusters(data_pca, labels_kmeans, kmeans.cluster_centers_, 'K-Means Clustering')
# Plotar resultados para Hierarchical Clustering sem centróides
plot_clusters(data_pca, labels_hierarchical, title='Hierarchical Clustering')
# Plotar resultados para DBSCAN sem centróides
plot_clusters(data_pca, labels_dbscan, title='DBSCAN Clustering')

# Imprimir quantos clusters foram gerados e os centróides para K-Means
print("Número de clusters em K-Means:", np.unique(labels_kmeans).size)
print("Centróides de K-Means:\n", kmeans.cluster_centers_)
print("Número de clusters em Hierarchical:", np.unique(labels_hierarchical).size)
print("Número de clusters em DBSCAN:", np.unique(labels_dbscan).size)
