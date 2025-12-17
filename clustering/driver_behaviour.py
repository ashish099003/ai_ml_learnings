import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def driver_plot(df):
    plt.figure(figsize=(8,6))
    sns.scatterplot(df, x='mean_dist_day', y= 'mean_over_speed_perc', s=60, color='blue', edgecolor='black')
    plt.title('Driver Behavior Scatter Plot')
    plt.xlabel('Mean Distance per Day (km)')
    plt.ylabel('Mean Over-Speeding (%)')
    plt.grid(True)
    plt.show()

def get_k_cluster(scaled_data):

    # Elbow Method to determine the optimal number of clusters
    wcss = []
    silhouette_scores = []
    K_range = range(2, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(K_range, wcss, marker='o')
    plt.title('Elbow Method - Optimal K')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('WCSS')
    plt.grid(True)

    # Plot Silhouette Scores
    plt.subplot(1, 2, 2)
    plt.plot(K_range, silhouette_scores, marker='o', color='green')
    plt.title('Silhouette Score Analysis')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Return best K based on max silhouette score
    optimal_k = K_range[silhouette_scores.index(max(silhouette_scores))]
    return optimal_k

def apply_kmeans(df, scaled_data, scaler):
    # Apply KMeans with 4 clusters
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    df['Cluster'] = clusters

    # Plot the clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='mean_dist_day', y='mean_over_speed_perc', hue='Cluster', palette='tab10', s=100)
    plt.scatter(scaler.inverse_transform(kmeans.cluster_centers_)[:, 0],
                scaler.inverse_transform(kmeans.cluster_centers_)[:, 1],
                s=300, c='black', marker='X', label='Centroids')
    plt.title('K-Means Clustering with K=4')
    plt.xlabel('Mean Distance per Day')
    plt.ylabel('Mean Over Speed Percentage')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def driver_clustering():
    print(10)
    url = "https://raw.githubusercontent.com/JangirSumit/kmeans-clustering/refs/heads/master/driver-data.csv"
    df = pd.read_csv(url)
    # driver_plot(df)
    # print(df.head())
    # print(df.describe())
    X = df[['mean_dist_day', 'mean_over_speed_perc']]
    print(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # get_k_cluster(X_scaled)
    apply_kmeans(df,X_scaled,scaler)



if __name__=='__main__':
    driver_clustering()