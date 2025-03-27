# K-Means Clustering on Wine Dataset

Welcome to the **K-Means Clustering on Wine Dataset** project! This repository demonstrates how to use the **K-Means clustering** algorithm to group wines based on their chemical properties. The dataset used for this model is the **Wine dataset** from `sklearn.datasets`.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation Instructions](#installation-instructions)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Evaluation](#model-evaluation)
- [Visualizations](#visualizations)
- [Conclusion](#conclusion)
- [License](#license)

## Project Overview

In this project, we implement the **K-Means clustering** algorithm to classify wines into distinct clusters based on their chemical properties such as alcohol content, malic acid, and color intensity. The **Elbow Method** and **Silhouette Score** are used to identify the optimal number of clusters. We also visualize the clustering results after applying **PCA (Principal Component Analysis)** for dimensionality reduction.

## Dataset

The **Wine dataset** from `sklearn.datasets` contains 178 samples of wines, each described by 13 features such as:
- **Alcohol content**
- **Malic acid**
- **Color intensity**
- **Hue**
- **OD280/OD315 of diluted wines**

The dataset is used for unsupervised learning to identify natural clusters in the data.

## Installation Instructions

To run this project locally, you'll need to set up your environment by installing the required dependencies.

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/k-means-wine-clustering.git
   ```

2. Navigate to the project directory:
   ```bash
   cd k-means-wine-clustering
   ```

3. Install the necessary Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

   This will install the following libraries:
   - pandas
   - numpy
   - scikit-learn
   - matplotlib
   - seaborn

## Usage

After setting up your environment, you can execute the Python script to train the **K-Means clustering** model, evaluate its performance, and visualize the results.

1. Run the script:
   ```bash
   python k_means_wine_clustering.py
   ```

   The script will output:
   - **Elbow plot** for optimal `k` using WCSS (Within-Cluster Sum of Squares).
   - **Silhouette scores** for each value of `k`.
   - **Cluster visualization** after dimensionality reduction using PCA.

## Project Structure

The project directory contains the following files:

- **k_means_wine_clustering.py**: Python script that loads the dataset, trains the K-Means model, evaluates performance, and visualizes results.
- **requirements.txt**: A list of required dependencies.
- **elbow_plot.png**: A plot showing the Elbow Method for optimal `k`.
- **cluster_visualization.png**: A plot showing the wine dataset clusters after applying PCA.

## Model Evaluation

The K-Means model’s performance is evaluated using:

1. **Elbow Method**: Helps determine the optimal number of clusters by plotting WCSS and looking for the "elbow" point.
2. **Silhouette Score**: Measures how well the samples are clustered. A higher score indicates better clustering.

## Visualizations

1. **Elbow Plot**:
   - The Elbow plot visualizes the WCSS for different values of `k` and helps identify the optimal number of clusters.



2. **Cluster Visualization**:
   - The cluster visualization shows the wine dataset clusters after dimensionality reduction using PCA.

   

## Conclusion

The **K-Means clustering** algorithm successfully grouped the Wine dataset into clusters, and the **Elbow Method** helped determine the optimal number of clusters (`k = 3`). The **Silhouette Score** indicates that the clusters have moderate cohesion. 

To improve the analysis further, you could:
- Explore other clustering algorithms such as **DBSCAN** or **Agglomerative Clustering**.
- Experiment with different dimensionality reduction techniques.
- Compare the clustering results with target labels (if available).

Feel free to explore and adapt this code for your own projects or tutorials. Contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
