import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


df = pd.read_csv(r"C:\Users\Bhuvaneswari\Downloads\Python\app_user_behavior_dataset.csv")

st.title("App user behavior")

# FEATURE SELECTION
features = [
    'sessions_per_week',
    'avg_session_duration_min',
    'daily_active_minutes',
    'feature_clicks_per_session',
    'engagement_score'
]

X = df[features]

# SCALING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_) 

plt.plot(K, inertia, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
st.pyplot(plt)  


# K-MEANS CLUSTERING

k = st.slider("Select Number of Clusters", 1, 2, 3)

kmeans = KMeans(n_clusters=k, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# PCA FOR VISUALIZATION
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# PLOT CLUSTERS
st.subheader("Cluster Visualization (PCA)")

fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=df['cluster'])

ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_title("User Clusters")

st.pyplot(fig)

# CLUSTER DISTRIBUTION
st.subheader("Cluster Distribution")
st.bar_chart(df['cluster'].value_counts())

# CLUSTER SUMMARY
st.subheader("Cluster Behavior Summary")
cluster_summary = df.groupby('cluster').mean(numeric_only=True)
st.write(cluster_summary)