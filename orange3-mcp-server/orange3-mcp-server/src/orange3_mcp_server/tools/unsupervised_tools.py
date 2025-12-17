import os
import json
import logging
import tempfile
import pickle
import numpy as np
from Orange.data import Table, Domain
from Orange.distance import Euclidean, Mahalanobis, Cosine, Jaccard, SpearmanR, PearsonR
from Orange.projection import PCA, MDS, TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from Orange.clustering import KMeans, HierarchicalClustering, DBSCAN, Louvain
from Orange.projection.manifold import MDS as ManifoldMDS  # For manifold variants
from scipy.spatial.distance import cdist  # For custom distances
from scipy.stats import pearsonr, spearmanr  # For correlations fallback
from sklearn.neighbors import LocalOutlierFactor  # Replacement for Orange's LOF

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def initialize_weights(data, x_dim, y_dim):
    """Initialize SOM weights randomly."""
    n_features = data.shape[1]
    return np.random.random((x_dim, y_dim, n_features))

def find_winner(weights, sample, x_dim, y_dim):
    """Find the best matching unit (BMU) for a sample."""
    bmu_idx = np.argmin(np.sum((weights - sample) ** 2, axis=2))
    return np.unravel_index(bmu_idx, (x_dim, y_dim))

def update_weights(weights, sample, bmu_pos, learning_rate, sigma, x_dim, y_dim, t):
    """Update weights based on neighborhood function."""
    bmu_x, bmu_y = bmu_pos
    for i in range(x_dim):
        for j in range(y_dim):
            dist_to_bmu = np.sqrt((i - bmu_x) ** 2 + (j - bmu_y) ** 2)
            influence = np.exp(-dist_to_bmu ** 2 / (2 * sigma ** 2))
            weights[i, j] += learning_rate * influence * (sample - weights[i, j])
    # Decay learning rate and sigma
    return weights, learning_rate * 0.95, sigma * 0.95

async def distance_file(file_path: str) -> str:
    """Load a distance matrix from a file (using NumPy as fallback).
    
    Args:
        file_path: Path to the distance file (e.g., .npy or .pkl).
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return json.dumps({"error": f"File not found: {file_path}"})
        
        if file_path.endswith('.npy'):
            dist_matrix = np.load(file_path)
        elif file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                dist_matrix = pickle.load(f)
        else:
            raise ValueError("Unsupported file format. Use .npy or .pkl.")
        
        logger.debug(f"Loaded distance matrix: shape {dist_matrix.shape}")
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as temp:
            pickle.dump(dist_matrix, temp)
            matrix_path = temp.name
        return json.dumps({
            "matrix_path": matrix_path,
            "shape": list(dist_matrix.shape),
            "message": "Distance matrix loaded from file successfully"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to load distance file: {str(e)}")
        return json.dumps({"error": f"Failed to load distance file: {str(e)}"})

async def distance_matrix(data_path: str, metric: str = 'euclidean') -> str:
    """Compute a distance matrix from data.
    
    Args:
        data_path: Input data file path.
        metric: Distance metric ('euclidean', 'mahalanobis', 'cosine', 'jaccard', 'spearman', 'pearson').
    """
    try:
        data = Table(data_path)
        dist_class = {
            'euclidean': Euclidean,
            'mahalanobis': Mahalanobis,
            'cosine': Cosine,
            'jaccard': Jaccard,
            'spearman': SpearmanR,
            'pearson': PearsonR
        }.get(metric, Euclidean)
        dist_matrix = dist_class(data)
        logger.debug(f"Computed distance matrix: shape {dist_matrix.shape}")
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as temp:
            pickle.dump(dist_matrix, temp)
            matrix_path = temp.name
        return json.dumps({
            "matrix_path": matrix_path,
            "shape": list(dist_matrix.shape),
            "message": f"Distance matrix computed with {metric} metric successfully"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to compute distance matrix: {str(e)}")
        return json.dumps({"error": f"Failed to compute distance matrix: {str(e)}"})

async def tsne(data_path: str, n_components: int = 2, perplexity: float = 30.0, n_iter: int = 1000, metric: str = 'euclidean') -> str:
    """Perform t-SNE dimensionality reduction.
    
    Args:
        data_path: Input data file path.
        n_components: Number of dimensions (usually 2 or 3).
        perplexity: Perplexity parameter.
        n_iter: Number of iterations.
        metric: Distance metric ('euclidean', etc.).
    """
    try:
        data = Table(data_path)
        tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, metric=metric)
        projected = tsne(data)
        logger.debug(f"t-SNE projected: {len(projected)} rows")
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as temp:
            pickle.dump(projected, temp)
            projected_path = temp.name
        return json.dumps({
            "projected_path": projected_path,
            "rows": len(projected),
            "message": "t-SNE reduction completed successfully"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to perform t-SNE: {str(e)}")
        return json.dumps({"error": f"Failed to perform t-SNE: {str(e)}"})

async def correlations(data_path: str, method: str = 'pearson') -> str:
    """Compute correlations between variables using NumPy/SciPy.
    
    Args:
        data_path: Input data file path.
        method: Correlation method ('pearson', 'spearman').
    """
    try:
        data = Table(data_path)
        X = data.X  # Numerical data matrix
        if method.lower() == 'pearson':
            corr_matrix = np.corrcoef(X.T)  # Transpose to get feature correlations
        elif method.lower() == 'spearman':
            corr_matrix, _ = spearmanr(X.T)
        else:
            raise ValueError(f"Unsupported correlation method: {method}")
        corr_matrix = np.nan_to_num(corr_matrix)  # Handle NaN values
        logger.debug(f"Computed correlations: shape {corr_matrix.shape}")
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as temp:
            pickle.dump(corr_matrix, temp)
            corr_path = temp.name
        return json.dumps({
            "corr_path": corr_path,
            "shape": list(corr_matrix.shape),
            "message": f"Correlations computed with {method} method successfully"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to compute correlations: {str(e)}")
        return json.dumps({"error": f"Failed to compute correlations: {str(e)}"})

async def distance_map(data_path: str, metric: str = 'euclidean') -> str:
    """Compute and map distances (visualization via matrix).
    
    Args:
        data_path: Input data file path.
        metric: Distance metric ('euclidean', etc.).
    """
    try:
        data = Table(data_path)
        dist_matrix = Euclidean(data) if metric == 'euclidean' else Mahalanobis(data)  # Example, extend for others
        logger.debug(f"Distance map matrix: shape {dist_matrix.shape}")
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as temp:
            pickle.dump(dist_matrix, temp)
            map_path = temp.name
        return json.dumps({
            "map_path": map_path,
            "shape": list(dist_matrix.shape),
            "message": f"Distance map computed with {metric} successfully"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to compute distance map: {str(e)}")
        return json.dumps({"error": f"Failed to compute distance map: {str(e)}"})

async def hierarchical_clustering(data_path: str, linkage: str = 'ward', n_clusters: int = None, distance_metric: str = 'euclidean') -> str:
    """Perform Hierarchical Clustering.
    
    Args:
        data_path: Input data file path.
        linkage: Linkage method ('ward', 'single', 'complete', 'average').
        n_clusters: Number of clusters to cut (optional).
        distance_metric: Distance metric ('euclidean', etc.).
    """
    try:
        data = Table(data_path)
        dist = Euclidean(data) if distance_metric == 'euclidean' else Mahalanobis(data)
        hc = HierarchicalClustering(linkage=linkage)
        model = hc(dist)
        if n_clusters:
            clusters = model.get_clusters(n_clusters)
        else:
            clusters = model.labels_
        logger.debug(f"Hierarchical clustering: {len(np.unique(clusters))} clusters")
        return json.dumps({
            "clusters": clusters.tolist(),
            "message": f"Hierarchical clustering completed with linkage {linkage}"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to perform hierarchical clustering: {str(e)}")
        return json.dumps({"error": f"Failed to perform hierarchical clustering: {str(e)}"})

async def kmeans(data_path: str, n_clusters: int = 3, init: str = 'k-means++', max_iter: int = 300, random_state: int = None) -> str:
    """Perform k-Means clustering.
    
    Args:
        data_path: Input data file path.
        n_clusters: Number of clusters.
        init: Initialization method ('k-means++', 'random').
        max_iter: Maximum iterations.
        random_state: Random seed for reproducibility.
    """
    try:
        data = Table(data_path)
        kmeans = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, random_state=random_state)
        model = kmeans(data)
        clusters = model(data)
        logger.debug(f"k-Means clustering: {len(np.unique(clusters))} clusters")
        return json.dumps({
            "clusters": clusters.tolist(),
            "message": f"k-Means clustering completed with {n_clusters} clusters"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to perform k-Means: {str(e)}")
        return json.dumps({"error": f"Failed to perform k-Means: {str(e)}"})

async def louvain_clustering(data_path: str, resolution: float = 1.0, **kwargs) -> str:
    """Perform Louvain clustering (requires graph input, assume data as adjacency).
    
    Args:
        data_path: Input data file path (assume adjacency matrix).
        resolution: Resolution parameter for modularity.
        **kwargs: Additional params.
    """
    try:
        data = Table(data_path)
        # Assume data.X is adjacency matrix
        graph = data.X  # Convert to graph if needed
        louvain = Louvain(resolution=resolution, **kwargs)
        clusters = louvain(graph)
        logger.debug(f"Louvain clustering: {len(np.unique(clusters))} communities")
        return json.dumps({
            "clusters": clusters.tolist(),
            "message": "Louvain clustering completed successfully"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to perform Louvain clustering: {str(e)}")
        return json.dumps({"error": f"Failed to perform Louvain clustering: {str(e)}"})

async def dbscan(data_path: str, eps: float = 0.5, min_samples: int = 5, metric: str = 'euclidean') -> str:
    """Perform DBSCAN clustering.
    
    Args:
        data_path: Input data file path.
        eps: Epsilon radius.
        min_samples: Minimum samples for core point.
        metric: Distance metric.
    """
    try:
        data = Table(data_path)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        clusters = dbscan(data)
        logger.debug(f"DBSCAN clustering: {len(np.unique(clusters))} clusters")
        return json.dumps({
            "clusters": clusters.tolist(),
            "message": f"DBSCAN clustering completed with eps {eps}"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to perform DBSCAN: {str(e)}")
        return json.dumps({"error": f"Failed to perform DBSCAN: {str(e)}"})

async def manifold_learning(data_path: str, method: str = 'isomap', n_components: int = 2, n_neighbors: int = 5, **kwargs) -> str:
    """Perform Manifold Learning (Isomap, LLE, etc.).
    
    Args:
        data_path: Input data file path.
        method: Manifold method ('isomap', 'lle', 'spectral').
        n_components: Number of components.
        n_neighbors: Number of neighbors for local methods.
        **kwargs: Additional params like eigen_solver.
    """
    try:
        data = Table(data_path)
        manifold_class = {
            'isomap': Isomap,
            'lle': LocallyLinearEmbedding,
            'spectral': SpectralEmbedding
        }.get(method, Isomap)
        manifold = manifold_class(n_components=n_components, n_neighbors=n_neighbors, **kwargs)
        projected = manifold(data)
        logger.debug(f"Manifold learning ({method}): {len(projected)} rows")
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as temp:
            pickle.dump(projected, temp)
            projected_path = temp.name
        return json.dumps({
            "projected_path": projected_path,
            "rows": len(projected),
            "message": f"Manifold learning with {method} completed successfully"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to perform manifold learning: {str(e)}")
        return json.dumps({"error": f"Failed to perform manifold learning: {str(e)}"})

async def outliers(data_path: str, contamination: float = 'auto', n_neighbors: int = 20, metric: str = 'euclidean') -> str:
    """Detect outliers using Local Outlier Factor.
    
    Args:
        data_path: Input data file path.
        contamination: Proportion of outliers ('auto' or float 0-0.5).
        n_neighbors: Number of neighbors.
        metric: Distance metric.
    """
    try:
        data = Table(data_path)
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, metric=metric)
        scores = lof.fit_predict(data.X)  # Returns -1 for outliers, 1 for inliers
        outliers = np.where(scores == -1)[0].tolist()  # Indices of outliers
        logger.debug(f"Detected {len(outliers)} outliers")
        return json.dumps({
            "outliers": outliers,
            "scores": scores.tolist(),
            "message": "Outliers detected successfully"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to detect outliers: {str(e)}")
        return json.dumps({"error": f"Failed to detect outliers: {str(e)}"})

async def pca(data_path: str, n_components: int = 2, whiten: bool = False, svd_solver: str = 'auto') -> str:
    """Perform PCA dimensionality reduction.
    
    Args:
        data_path: Input data file path.
        n_components: Number of components.
        whiten: If True, whiten the data.
        svd_solver: Solver ('auto', 'full', 'arpack', 'randomized').
    """
    try:
        data = Table(data_path)
        pca = PCA(n_components=n_components, whiten=whiten, svd_solver=svd_solver)
        projected = pca(data)
        logger.debug(f"PCA projected: {len(projected)} rows")
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as temp:
            pickle.dump(projected, temp)
            projected_path = temp.name
        return json.dumps({
            "projected_path": projected_path,
            "rows": len(projected),
            "message": "PCA reduction completed successfully"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to perform PCA: {str(e)}")
        return json.dumps({"error": f"Failed to perform PCA: {str(e)}"})

async def neighbors(data_path: str, n_neighbors: int = 5, metric: str = 'euclidean') -> str:
    """Compute nearest neighbors distances.
    
    Args:
        data_path: Input data file path.
        n_neighbors: Number of neighbors.
        metric: Distance metric.
    """
    try:
        data = Table(data_path)
        dist_matrix = cdist(data.X, data.X, metric=metric)
        neighbors = np.argsort(dist_matrix, axis=1)[:, 1:n_neighbors+1]
        logger.debug(f"Computed neighbors for {len(data)} instances")
        return json.dumps({
            "neighbors": neighbors.tolist(),
            "message": f"Nearest {n_neighbors} neighbors computed with {metric}"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to compute neighbors: {str(e)}")
        return json.dumps({"error": f"Failed to compute neighbors: {str(e)}"})

async def correspondence_analysis(data_path: str, n_components: int = 2) -> str:
    """Perform Correspondence Analysis.
    
    Args:
        data_path: Input data file path (contingency table expected).
        n_components: Number of components.
    """
    try:
        data = Table(data_path)
        # CA is not directly in Orange, use SVD on contingency for approximation
        cont = data.X  # Assume contingency table
        row_sums = cont.sum(axis=1, keepdims=True)
        col_sums = cont.sum(axis=0, keepdims=True)
        expected = row_sums @ col_sums / cont.sum()
        residuals = (cont - expected) / np.sqrt(expected)
        u, s, vt = np.linalg.svd(residuals, full_matrices=False)
        projected = u[:, :n_components] * s[:n_components]
        logger.debug(f"Correspondence Analysis projected: shape {projected.shape}")
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as temp:
            pickle.dump(projected, temp)
            projected_path = temp.name
        return json.dumps({
            "projected_path": projected_path,
            "shape": list(projected.shape),
            "message": "Correspondence Analysis completed successfully"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to perform Correspondence Analysis: {str(e)}")
        return json.dumps({"error": f"Failed to perform Correspondence Analysis: {str(e)}"})

async def distances(data_path: str, metric: str = 'euclidean') -> str:
    """Compute pairwise distances.
    
    Args:
        data_path: Input data file path.
        metric: Distance metric.
    """
    try:
        data = Table(data_path)
        dist_matrix = Euclidean(data) if metric == 'euclidean' else Mahalanobis(data)
        logger.debug(f"Computed distances: shape {dist_matrix.shape}")
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as temp:
            pickle.dump(dist_matrix, temp)
            dist_path = temp.name
        return json.dumps({
            "dist_path": dist_path,
            "shape": list(dist_matrix.shape),
            "message": f"Pairwise distances computed with {metric} successfully"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to compute distances: {str(e)}")
        return json.dumps({"error": f"Failed to compute distances: {str(e)}"})

async def distance_transformation(data_path: str, transformation: str = 'normalize', **kwargs) -> str:
    """Transform distance matrix (e.g., normalize).
    
    Args:
        data_path: Input data file path (distance matrix as .pkl).
        transformation: Type ('normalize', 'square', etc.).
        **kwargs: Additional params.
    """
    try:
        with open(data_path, 'rb') as f:
            dist_matrix = pickle.load(f)
        if transformation == 'normalize':
            dist_matrix = dist_matrix / np.max(dist_matrix)
        elif transformation == 'square':
            dist_matrix = dist_matrix ** 2
        # Add more transformations as needed
        logger.debug(f"Transformed distance matrix: shape {dist_matrix.shape}")
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as temp:
            pickle.dump(dist_matrix, temp)
            transformed_path = temp.name
        return json.dumps({
            "transformed_path": transformed_path,
            "shape": list(dist_matrix.shape),
            "message": f"Distance transformation ({transformation}) completed successfully"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to transform distances: {str(e)}")
        return json.dumps({"error": f"Failed to transform distances: {str(e)}"})

async def mds(data_path: str, n_components: int = 2, metric: bool = True, n_init: int = 4, max_iter: int = 300) -> str:
    """Perform MDS dimensionality reduction.
    
    Args:
        data_path: Input data file path.
        n_components: Number of components.
        metric: If True, metric MDS; else non-metric.
        n_init: Number of initializations.
        max_iter: Maximum iterations.
    """
    try:
        data = Table(data_path)
        mds = MDS(n_components=n_components, metric=metric, n_init=n_init, max_iter=max_iter)
        projected = mds(data)
        logger.debug(f"MDS projected: {len(projected)} rows")
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as temp:
            pickle.dump(projected, temp)
            projected_path = temp.name
        return json.dumps({
            "projected_path": projected_path,
            "rows": len(projected),
            "message": "MDS reduction completed successfully"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to perform MDS: {str(e)}")
        return json.dumps({"error": f"Failed to perform MDS: {str(e)}"})

async def save_distance_matrix(matrix_path: str, output_path: str) -> str:
    """Save a distance matrix to file.
    
    Args:
        matrix_path: Path to computed distance matrix (pickled).
        output_path: Output file path (e.g., .npy or .pkl).
    """
    try:
        with open(matrix_path, 'rb') as f:
            dist_matrix = pickle.load(f)
        if output_path.endswith('.npy'):
            np.save(output_path, dist_matrix)
        elif output_path.endswith('.pkl'):
            with open(output_path, 'wb') as f_out:
                pickle.dump(dist_matrix, f_out)
        else:
            raise ValueError("Unsupported file format. Use .npy or .pkl.")
        logger.debug(f"Saved distance matrix to {output_path}")
        return json.dumps({
            "message": f"Distance matrix saved to {output_path} successfully"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to save distance matrix: {str(e)}")
        return json.dumps({"error": f"Failed to save distance matrix: {str(e)}"})

async def self_organizing_map(data_path: str, x_dim: int = 10, y_dim: int = 10, learning_rate: float = 0.5, sigma: float = 1.0, n_iterations: int = 100) -> str:
    """Perform Self-Organizing Map clustering using NumPy.
    
    Args:
        data_path: Input data file path.
        x_dim: X dimension of the map.
        y_dim: Y dimension of the map.
        learning_rate: Initial learning rate.
        sigma: Initial neighborhood sigma.
        n_iterations: Number of training iterations.
    """
    try:
        data = Table(data_path)
        data_matrix = data.X
        weights = initialize_weights(data_matrix, x_dim, y_dim)
        
        for t in range(n_iterations):
            for sample in data_matrix:
                bmu_pos = find_winner(weights, sample, x_dim, y_dim)
                weights, learning_rate, sigma = update_weights(weights, sample, bmu_pos, learning_rate, sigma, x_dim, y_dim, t)
        
        # Assign clusters based on BMU
        clusters = np.array([find_winner(weights, sample, x_dim, y_dim)[0] * y_dim + find_winner(weights, sample, x_dim, y_dim)[1] for sample in data_matrix])
        logger.debug(f"SOM clustering: map size {x_dim}x{y_dim}")
        return json.dumps({
            "clusters": clusters.tolist(),
            "message": "Self-Organizing Map completed successfully"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to perform Self-Organizing Map: {str(e)}")
        return json.dumps({"error": f"Failed to perform Self-Organizing Map: {str(e)}"})