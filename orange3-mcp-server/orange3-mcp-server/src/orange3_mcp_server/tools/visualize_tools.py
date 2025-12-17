import os
import json
import logging
import numpy as np
import pandas as pd
import pickle
from Orange.data import Table
from Orange.projection import FreeViz, PCA
from sklearn.metrics import silhouette_score  # 使用 sklearn 替代 Orange silhouette_score
from scipy.spatial.distance import pdist, squareform  # 為 heat_map 使用

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def tree_viewer(model_path: str) -> str:
    """View a decision tree model as text representation.
    
    Args:
        model_path: Path to the pickled tree model.
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        # 簡化為文本表示 (假設 model 有 to_string 或 str 方法)
        tree_str = str(model) if hasattr(model, 'str') else "Tree representation not available in text form."
        return json.dumps({"tree_text": tree_str, "message": "Tree viewed successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to view tree: {str(e)}")
        return json.dumps({"error": f"Failed to view tree: {str(e)}"})

async def box_plot(data_path: str, var: str, group_var: str = None) -> str:
    """Generate a box plot description (placeholder for GUI).
    
    Args:
        data_path: Input data file path.
        var: Variable to plot.
        group_var: Optional grouping variable.
    """
    try:
        data = Table(data_path)
        logger.debug(f"Generating box plot for {var} in {data_path}")
        return json.dumps({
            "var": var,
            "group_var": group_var,
            "message": "Box plot data ready (GUI visualization required)"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to generate box plot: {str(e)}")
        return json.dumps({"error": f"Failed to generate box plot: {str(e)}"})

async def violin_plot(data_path: str, var: str, group_var: str = None) -> str:
    """Generate a violin plot description.
    
    Args:
        data_path: Input data file path.
        var: Variable to plot.
        group_var: Optional grouping variable.
    """
    try:
        data = Table(data_path)
        logger.debug(f"Generating violin plot for {var} in {data_path}")
        return json.dumps({
            "var": var,
            "group_var": group_var,
            "message": "Violin plot data ready (GUI visualization required)"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to generate violin plot: {str(e)}")
        return json.dumps({"error": f"Failed to generate violin plot: {str(e)}"})

async def distributions(data_path: str, var: str, bins: int = 10) -> str:
    """Generate distribution data.
    
    Args:
        data_path: Input data file path.
        var: Variable to plot.
        bins: Number of bins.
    """
    try:
        data = Table(data_path)
        values = data[:, var].X.flatten()
        logger.debug(f"Generating distribution for {var} in {data_path}")
        return json.dumps({
            "var": var,
            "bins": bins,
            "values_range": [float(np.min(values)), float(np.max(values))],
            "message": "Distribution data ready"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to generate distribution: {str(e)}")
        return json.dumps({"error": f"Failed to generate distribution: {str(e)}"})

async def scatter_plot(data_path: str, x_var: str, y_var: str, color_var: str = None) -> str:
    """Generate scatter plot data.
    
    Args:
        data_path: Input data file path.
        x_var: X-axis variable.
        y_var: Y-axis variable.
        color_var: Optional color variable.
    """
    try:
        data = Table(data_path)
        x = data[:, x_var].X.flatten()
        y = data[:, y_var].X.flatten()
        logger.debug(f"Generating scatter plot {x_var} vs {y_var} in {data_path}")
        return json.dumps({
            "x_var": x_var,
            "y_var": y_var,
            "color_var": color_var,
            "x_range": [float(np.min(x)), float(np.max(x))],
            "y_range": [float(np.min(y)), float(np.max(y))],
            "message": "Scatter plot data ready"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to generate scatter plot: {str(e)}")
        return json.dumps({"error": f"Failed to generate scatter plot: {str(e)}"})

async def line_plot(data_path: str, x_var: str, y_var: str) -> str:
    """Generate line plot data.
    
    Args:
        data_path: Input data file path.
        x_var: X-axis variable.
        y_var: Y-axis variable.
    """
    try:
        data = Table(data_path)
        x = data[:, x_var].X.flatten()
        y = data[:, y_var].X.flatten()
        logger.debug(f"Generating line plot {x_var} vs {y_var} in {data_path}")
        return json.dumps({
            "x_var": x_var,
            "y_var": y_var,
            "x_range": [float(np.min(x)), float(np.max(x))],
            "y_range": [float(np.min(y)), float(np.max(y))],
            "message": "Line plot data ready"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to generate line plot: {str(e)}")
        return json.dumps({"error": f"Failed to generate line plot: {str(e)}"})

async def bar_plot(data_path: str, var: str, group_var: str = None) -> str:
    """Generate bar plot data.
    
    Args:
        data_path: Input data file path.
        var: Variable to plot.
        group_var: Optional grouping variable.
    """
    try:
        data = Table(data_path)
        df = pd.DataFrame(data)
        if group_var:
            grouped = df.groupby(group_var)[var].mean()
        else:
            grouped = df[var].value_counts()
        logger.debug(f"Generating bar plot for {var} in {data_path}")
        return json.dumps({
            "var": var,
            "group_var": group_var,
            "grouped_data": grouped.to_dict(),
            "message": "Bar plot data ready"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to generate bar plot: {str(e)}")
        return json.dumps({"error": f"Failed to generate bar plot: {str(e)}"})

async def sieve_diagram(data_path: str, var1: str, var2: str) -> str:
    """Generate sieve diagram data.
    
    Args:
        data_path: Input data file path.
        var1: First variable.
        var2: Second variable.
    """
    try:
        data = Table(data_path)
        df = pd.DataFrame(data)
        contingency = pd.crosstab(df[var1], df[var2])
        logger.debug(f"Generating sieve diagram for {var1} and {var2} in {data_path}")
        return json.dumps({
            "var1": var1,
            "var2": var2,
            "contingency": contingency.to_dict(),
            "message": "Sieve diagram data ready"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to generate sieve diagram: {str(e)}")
        return json.dumps({"error": f"Failed to generate sieve diagram: {str(e)}"})

async def mosaic_display(data_path: str, var1: str, var2: str) -> str:
    """Generate mosaic display data.
    
    Args:
        data_path: Input data file path.
        var1: First variable.
        var2: Second variable.
    """
    try:
        data = Table(data_path)
        df = pd.DataFrame(data)
        contingency = pd.crosstab(df[var1], df[var2], normalize='all')
        logger.debug(f"Generating mosaic display for {var1} and {var2} in {data_path}")
        return json.dumps({
            "var1": var1,
            "var2": var2,
            "contingency": contingency.to_dict(),
            "message": "Mosaic display data ready"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to generate mosaic display: {str(e)}")
        return json.dumps({"error": f"Failed to generate mosaic display: {str(e)}"})

async def freeviz(data_path: str, n_components: int = 2) -> str:
    """Generate a FreeViz plot.
    
    Args:
        data_path: Input data file path.
        n_components: Number of components.
    """
    try:
        data = Table(data_path)
        freeviz = FreeViz(n_components=n_components)
        projection = freeviz(data)
        logger.debug(f"FreeViz projected: {len(projection)} rows")
        return json.dumps({
            "rows": len(projection),
            "message": "FreeViz plot generated successfully"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to generate FreeViz: {str(e)}")
        return json.dumps({"error": f"Failed to generate FreeViz: {str(e)}"})

async def linear_projection(data_path: str, method: str = 'PCA', n_components: int = 2) -> str:
    """Generate a linear projection using PCA as default.
    
    Args:
        data_path: Input data file path.
        method: Projection method ('PCA', etc.).
        n_components: Number of components.
    """
    try:
        data = Table(data_path)
        if method.lower() == 'pca':
            projection = PCA(n_components=n_components)(data)
        else:
            raise ValueError(f"Unsupported method: {method}. Only 'PCA' is supported.")
        logger.debug(f"Linear projection {method} projected: {len(projection)} rows")
        return json.dumps({
            "rows": len(projection),
            "method": method,
            "message": "Linear projection generated successfully"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to generate linear projection: {str(e)}")
        return json.dumps({"error": f"Failed to generate linear projection: {str(e)}"})

async def radviz(data_path: str) -> str:
    """Generate a Radviz plot (placeholder, as not directly supported in API).
    
    Args:
        data_path: Input data file path.
    """
    try:
        logger.warning("Radviz is not supported in current Orange version. Placeholder response.")
        return json.dumps({
            "message": "Radviz plot not available in current version"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to generate Radviz: {str(e)}")
        return json.dumps({"error": f"Failed to generate Radviz: {str(e)}"})

async def heat_map(data_path: str, distance_measure: str = 'euclidean') -> str:
    """Generate a heat map data.
    
    Args:
        data_path: Input data file path.
        distance_measure: Distance measure to use.
    """
    try:
        data = Table(data_path)
        if distance_measure == 'euclidean':
            dist_matrix = squareform(pdist(data.X, metric='euclidean'))
        else:
            dist_matrix = squareform(pdist(data.X, metric=distance_measure))
        logger.debug(f"Generating heat map for {data_path} with {distance_measure}")
        return json.dumps({
            "distance_measure": distance_measure,
            "dist_matrix_shape": list(dist_matrix.shape),
            "message": "Heat map data ready"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to generate heat map: {str(e)}")
        return json.dumps({"error": f"Failed to generate heat map: {str(e)}"})

async def silhouette_plot(data_path: str, cluster_labels: list) -> str:
    """Generate a silhouette plot.
    
    Args:
        data_path: Input data file path.
        cluster_labels: List of cluster labels.
    """
    try:
        data = Table(data_path)
        score = silhouette_score(data.X, np.array(cluster_labels))
        logger.debug(f"Silhouette score: {score}")
        return json.dumps({
            "score": float(score),
            "message": "Silhouette plot generated successfully"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to generate silhouette plot: {str(e)}")
        return json.dumps({"error": f"Failed to generate silhouette plot: {str(e)}"})

async def cn2_rule_viewer(model_path: str) -> str:
    """View CN2 rules.
    
    Args:
        model_path: Path to the trained CN2 model file.
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        rules = str(model.rule_list) if hasattr(model, 'rule_list') else "Rules not available"
        logger.debug(f"Viewing CN2 rules from {model_path}")
        return json.dumps({
            "rules": rules,
            "message": "CN2 rules viewed successfully"
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to view CN2 rules: {str(e)}")
        return json.dumps({"error": f"Failed to view CN2 rules: {str(e)}"})