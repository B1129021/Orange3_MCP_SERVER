import os
import sys
import logging
os.environ.setdefault('FASTMCP_LOG_LEVEL', 'INFO')
from fastmcp import FastMCP

# Import from each category
from .tools.data_tools import (
    load_file, load_csv_file, load_datasets, load_data_table,
    get_data_info, rank_features, edit_domain, get_feature_statistics, save_data
)
from .tools.transform_tools import (
    data_sampler, select_columns, select_rows, transpose, merge_data, concatenate,
    select_by_data_index, unique, aggregate_columns, group_by_tool, pivot_table,
    apply_domain, preprocess, impute, continuize, discretize, randomize,
    purge_domain, melt, formula, create_class, create_instance, python_script
)
from .tools.visualize_tools import (
    tree_viewer, box_plot, violin_plot, distributions, scatter_plot, line_plot,
    bar_plot, sieve_diagram, mosaic_display, freeviz, linear_projection, radviz,
    heat_map, silhouette_plot, cn2_rule_viewer
)
from .tools.model_tools import (
    train_calibrated, train_knn, train_tree,
    train_random_forest, train_gradient_boosting, train_svm, train_linear_regression,
    train_logistic_regression, train_naive_bayes, train_adaboost, train_pls,
    train_curve_fit, train_neural_network, train_sgd, train_stacking,
    save_model, load_model
)
from .tools.evaluate_tools import (
    evaluate_cross_validation, evaluate_leave_one_out, evaluate_test_on_training,
    evaluate_test_on_test_data, evaluate_accuracy, evaluate_rmse, evaluate_mae,
    evaluate_r2, evaluate_f1
)
from .tools.unsupervised_tools import (
    distance_file, distance_matrix, tsne, correlations, distance_map,
    hierarchical_clustering, kmeans, louvain_clustering, dbscan,
    manifold_learning, outliers, pca, neighbors, correspondence_analysis,
    distances, distance_transformation, mds, save_distance_matrix,
    self_organizing_map
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

mcp = FastMCP("Orange3 MCP Server")

def register_tools():
    """Register all Orange3 tools to the MCP server."""
    
    # Data tools
    @mcp.tool()
    async def load_file_tool(file_path: str, file_type: str = 'auto'):
        logger.info(f"Loading file from {file_path} with type {file_type}")
        return await load_file(file_path, file_type)
    
    @mcp.tool()
    async def load_csv_file_tool(file_path: str, delimiter: str = ',', encoding: str = 'utf-8'):
        logger.info(f"Loading CSV from {file_path} with delimiter {delimiter} and encoding {encoding}")
        return await load_csv_file(file_path, delimiter, encoding)
    
    @mcp.tool()
    async def load_datasets_tool(dataset_name: str = 'iris'):
        logger.info(f"Loading dataset {dataset_name}")
        return await load_datasets(dataset_name)
    
    @mcp.tool()
    async def load_data_table_tool(data_path: str):
        logger.info(f"Loading data table from {data_path}")
        return await load_data_table(data_path)
    
    @mcp.tool()
    async def get_data_info_tool(data_path: str):
        logger.info(f"Getting data info for {data_path}")
        return await get_data_info(data_path)
    
    @mcp.tool()
    async def rank_features_tool(data_path: str, method: str = 'gain_ratio', k: int = 5):
        logger.info(f"Ranking features for {data_path} with method {method} and k {k}")
        return await rank_features(data_path, method, k)
    
    @mcp.tool()
    async def edit_domain_tool(data_path: str, new_domain: dict):
        logger.info(f"Editing domain for {data_path}")
        return await edit_domain(data_path, new_domain)
    
    @mcp.tool()
    async def get_feature_statistics_tool(data_path: str, features: list = None):
        logger.info(f"Getting feature statistics for {data_path}")
        return await get_feature_statistics(data_path, features)
    
    @mcp.tool()
    async def save_data_tool(data_path: str, output_path: str, format: str = 'tab'):
        logger.info(f"Saving data from {data_path} to {output_path} in {format} format")
        return await save_data(data_path, output_path, format)
    
    # Transform tools
    @mcp.tool()
    async def data_sampler_tool(data_path: str, proportion: float = 0.1, stratified: bool = True, replace: bool = False):
        logger.info(f"Sampling data from {data_path} with proportion {proportion}")
        return await data_sampler(data_path, proportion, stratified, replace)
    
    @mcp.tool()
    async def select_columns_tool(data_path: str, columns: list):
        logger.info(f"Selecting columns {columns} from {data_path}")
        return await select_columns(data_path, columns)
    
    @mcp.tool()
    async def select_rows_tool(data_path: str, row_indices: list):
        logger.info(f"Selecting rows {row_indices} from {data_path}")
        return await select_rows(data_path, row_indices)
    
    @mcp.tool()
    async def transpose_tool(data_path: str):
        logger.info(f"Transposing data from {data_path}")
        return await transpose(data_path)
    
    @mcp.tool()
    async def merge_data_tool(data_path1: str, data_path2: str, on: list = None, how: str = 'inner'):
        logger.info(f"Merging data from {data_path1} and {data_path2}")
        return await merge_data(data_path1, data_path2, on, how)
    
    @mcp.tool()
    async def concatenate_tool(data_path1: str, data_path2: str, axis: int = 0):
        logger.info(f"Concatenating data from {data_path1} and {data_path2}")
        return await concatenate(data_path1, data_path2, axis)
    
    @mcp.tool()
    async def select_by_data_index_tool(data_path: str, index: int):
        logger.info(f"Selecting by data index from {data_path}")
        return await select_by_data_index(data_path, index)
    
    @mcp.tool()
    async def unique_tool(data_path: str, columns: list):
        logger.info(f"Finding unique values in {data_path}")
        return await unique(data_path, columns)
    
    @mcp.tool()
    async def aggregate_columns_tool(data_path: str, aggregation: str = 'mean'):
        logger.info(f"Aggregating columns in {data_path} with {aggregation}")
        return await aggregate_columns(data_path, aggregation)
    
    @mcp.tool()
    async def group_by_tool(data_path: str, group_by: list):
        logger.info(f"Grouping by {group_by} in {data_path}")
        return await group_by_tool(data_path, group_by)
    
    @mcp.tool()
    async def pivot_table_tool(data_path: str, values: str, index: str, columns: str):
        logger.info(f"Creating pivot table for {data_path}")
        return await pivot_table(data_path, values, index, columns)
    
    @mcp.tool()
    async def apply_domain_tool(data_path: str, new_domain: dict):
        logger.info(f"Applying domain to {data_path}")
        return await apply_domain(data_path, new_domain)
    
    @mcp.tool()
    async def preprocess_tool(data_path: str, method: str = 'normalize'):
        logger.info(f"Preprocessing data in {data_path} with {method}")
        return await preprocess(data_path, method)
    
    @mcp.tool()
    async def impute_tool(data_path: str, method: str = 'mean'):
        logger.info(f"Imputing missing values in {data_path} with {method}")
        return await impute(data_path, method)
    
    @mcp.tool()
    async def continuize_tool(data_path: str):
        logger.info(f"Continuizing data in {data_path}")
        return await continuize(data_path)
    
    @mcp.tool()
    async def discretize_tool(data_path: str, n_bins: int = 5):
        logger.info(f"Discretizing data in {data_path} with {n_bins} bins")
        return await discretize(data_path, n_bins)
    
    @mcp.tool()
    async def randomize_tool(data_path: str, rand_type: str = 'all', rand_seed: int = 42):
        logger.info(f"Randomizing data in {data_path}")
        return await randomize(data_path, rand_type, rand_seed)
    
    @mcp.tool()
    async def purge_domain_tool(data_path: str, remove_unused: bool = True, remove_constants: bool = True):
        logger.info(f"Purging domain in {data_path}")
        return await purge_domain(data_path, remove_unused, remove_constants)
    
    @mcp.tool()
    async def melt_tool(data_path: str, id_vars: list, value_vars: list = None):
        logger.info(f"Melting data in {data_path}")
        return await melt(data_path, id_vars, value_vars)
    
    @mcp.tool()
    async def formula_tool(data_path: str, formula_expr: str, new_col_name: str):
        logger.info(f"Applying formula in {data_path}")
        return await formula(data_path, formula_expr, new_col_name)
    
    @mcp.tool()
    async def create_class_tool(data_path: str, class_name: str, values: list):
        logger.info(f"Creating class in {data_path}")
        return await create_class(data_path, class_name, values)
    
    @mcp.tool()
    async def create_instance_tool(data_path: str, values: dict):
        logger.info(f"Creating instance in {data_path}")
        return await create_instance(data_path, values)
    
    @mcp.tool()
    async def python_script_tool(code: str, data_path: str = None):
        logger.info(f"Executing Python script")
        return await python_script(code, data_path)
    
    # Visualize tools
    @mcp.tool()
    async def tree_viewer_tool(model_path: str):
        logger.info(f"Viewing tree from {model_path}")
        return await tree_viewer(model_path)
    
    @mcp.tool()
    async def box_plot_tool(data_path: str, var: str, group_var: str = None):
        logger.info(f"Generating box plot for {data_path}")
        return await box_plot(data_path, var, group_var)
    
    @mcp.tool()
    async def violin_plot_tool(data_path: str, var: str, group_var: str = None):
        logger.info(f"Generating violin plot for {data_path}")
        return await violin_plot(data_path, var, group_var)
    
    @mcp.tool()
    async def distributions_tool(data_path: str, var: str, bins: int = 10):
        logger.info(f"Generating distributions for {data_path}")
        return await distributions(data_path, var, bins)
    
    @mcp.tool()
    async def scatter_plot_tool(data_path: str, x_var: str, y_var: str, color_var: str = None):
        logger.info(f"Generating scatter plot for {data_path}")
        return await scatter_plot(data_path, x_var, y_var, color_var)
    
    @mcp.tool()
    async def line_plot_tool(data_path: str, x_var: str, y_var: str):
        logger.info(f"Generating line plot for {data_path}")
        return await line_plot(data_path, x_var, y_var)
    
    @mcp.tool()
    async def bar_plot_tool(data_path: str, var: str, group_var: str = None):
        logger.info(f"Generating bar plot for {data_path}")
        return await bar_plot(data_path, var, group_var)
    
    @mcp.tool()
    async def sieve_diagram_tool(data_path: str, var1: str, var2: str):
        logger.info(f"Generating sieve diagram for {data_path}")
        return await sieve_diagram(data_path, var1, var2)
    
    @mcp.tool()
    async def mosaic_display_tool(data_path: str, var1: str, var2: str):
        logger.info(f"Generating mosaic display for {data_path}")
        return await mosaic_display(data_path, var1, var2)
    
    @mcp.tool()
    async def freeviz_tool(data_path: str, n_components: int = 2):
        logger.info(f"Generating FreeViz for {data_path}")
        return await freeviz(data_path, n_components)
    
    @mcp.tool()
    async def linear_projection_tool(data_path: str, method: str = 'PCA', n_components: int = 2):
        logger.info(f"Generating linear projection for {data_path}")
        return await linear_projection(data_path, method, n_components)
    
    @mcp.tool()
    async def radviz_tool(data_path: str):
        logger.info(f"Generating Radviz for {data_path}")
        return await radviz(data_path)
    
    @mcp.tool()
    async def heat_map_tool(data_path: str, distance_measure: str = 'euclidean'):
        logger.info(f"Generating heat map for {data_path}")
        return await heat_map(data_path, distance_measure)
    
    @mcp.tool()
    async def silhouette_plot_tool(data_path: str, cluster_labels: list):
        logger.info(f"Generating silhouette plot for {data_path}")
        return await silhouette_plot(data_path, cluster_labels)
    
    @mcp.tool()
    async def cn2_rule_viewer_tool(model_path: str):
        logger.info(f"Viewing CN2 rules from {model_path}")
        return await cn2_rule_viewer(model_path)
    
    # Model tools
    # @mcp.tool()
    # async def train_constant_tool(data_path: str, is_regression: bool = False):
    #     logger.info(f"Training constant model on {data_path}")
    #     return await train_constant(data_path, is_regression=is_regression)
    
    # @mcp.tool()
    # async def train_cn2_tool(data_path: str, gamma: float = 0.7, min_covered: int = 5):
    #     logger.info(f"Training CN2 on {data_path}")
    #     return await train_cn2(data_path, gamma=gamma, min_covered=min_covered)
    
    @mcp.tool()
    async def train_calibrated_tool(data_path: str, base_learner: str = 'svm', method: str = 'sigmoid'):
        logger.info(f"Training calibrated model on {data_path}")
        return await train_calibrated(data_path, base_learner=base_learner, method=method)
    
    @mcp.tool()
    async def train_knn_tool(data_path: str, n_neighbors: int = 5, metric: str = 'euclidean'):
        logger.info(f"Training KNN on {data_path}")
        return await train_knn(data_path, n_neighbors=n_neighbors, metric=metric)
    
    @mcp.tool()
    async def train_tree_tool(data_path: str, target_column: str = 'car', max_depth: int = None, min_samples_split: int = 2):
        logger.info(f"Training tree on {data_path} with target {target_column}")
        return await train_tree(data_path, target_column=target_column, max_depth=max_depth, min_samples_split=min_samples_split)
    
    @mcp.tool()
    async def train_random_forest_tool(data_path: str, n_estimators: int = 100, max_depth: int = None):
        logger.info(f"Training random forest on {data_path}")
        return await train_random_forest(data_path, n_estimators=n_estimators, max_depth=max_depth)
    
    @mcp.tool()
    async def train_gradient_boosting_tool(data_path: str, n_estimators: int = 100, learning_rate: float = 0.1):
        logger.info(f"Training gradient boosting on {data_path}")
        return await train_gradient_boosting(data_path, n_estimators=n_estimators, learning_rate=learning_rate)
    
    @mcp.tool()
    async def train_svm_tool(data_path: str, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale'):
        logger.info(f"Training SVM on {data_path}")
        return await train_svm(data_path, kernel=kernel, C=C, gamma=gamma)
    
    @mcp.tool()
    async def train_linear_regression_tool(data_path: str, alpha: float = 0.0001, fit_intercept: bool = True):
        logger.info(f"Training linear regression on {data_path}")
        return await train_linear_regression(data_path, alpha=alpha, fit_intercept=fit_intercept)
    
    @mcp.tool()
    async def train_logistic_regression_tool(data_path: str, C: float = 1.0, solver: str = 'lbfgs'):
        logger.info(f"Training logistic regression on {data_path}")
        return await train_logistic_regression(data_path, C=C, solver=solver)
    
    @mcp.tool()
    async def train_naive_bayes_tool(data_path: str):
        logger.info(f"Training naive Bayes on {data_path}")
        return await train_naive_bayes(data_path)
    
    @mcp.tool()
    async def train_adaboost_tool(data_path: str, n_estimators: int = 50, learning_rate: float = 1.0):
        logger.info(f"Training AdaBoost on {data_path}")
        return await train_adaboost(data_path, n_estimators=n_estimators, learning_rate=learning_rate)
    
    @mcp.tool()
    async def train_pls_tool(data_path: str, n_components: int = 2):
        logger.info(f"Training PLS on {data_path}")
        return await train_pls(data_path, n_components=n_components)
    
    @mcp.tool()
    async def train_curve_fit_tool(data_path: str, func_name: str = 'linear', p0: list = None):
        logger.info(f"Training curve fit on {data_path}")
        return await train_curve_fit(data_path, func_name=func_name, p0=p0)
    
    @mcp.tool()
    async def train_neural_network_tool(data_path: str, hidden_layer_sizes: tuple = (100,), activation: str = 'relu'):
        logger.info(f"Training neural network on {data_path}")
        return await train_neural_network(data_path, hidden_layer_sizes=hidden_layer_sizes, activation=activation)
    
    @mcp.tool()
    async def train_sgd_tool(data_path: str, loss: str = 'log', penalty: str = 'l2'):
        logger.info(f"Training SGD on {data_path}")
        return await train_sgd(data_path, loss=loss, penalty=penalty)
    
    @mcp.tool()
    async def train_stacking_tool(data_path: str, learners: list[str], meta_learner: str = 'logistic'):
        logger.info(f"Training stacking on {data_path}")
        return await train_stacking(data_path, learners=learners, meta_learner=meta_learner)
    
    @mcp.tool()
    async def save_model_tool(model_path: str, output_path: str):
        logger.info(f"Saving model from {model_path} to {output_path}")
        return await save_model(model_path, output_path)
    
    @mcp.tool()
    async def load_model_tool(model_path: str):
        logger.info(f"Loading model from {model_path}")
        return await load_model(model_path)
    
    # Evaluate tools
    @mcp.tool()
    async def evaluate_cross_validation_tool(data_path: str, learner: str = 'tree', n_folds: int = 5):
        logger.info(f"Evaluating cross validation on {data_path} with {learner}")
        return await evaluate_cross_validation(data_path, learner=learner, n_folds=n_folds)
    
    @mcp.tool()
    async def evaluate_leave_one_out_tool(data_path: str, learner: str = 'tree'):
        logger.info(f"Evaluating leave one out on {data_path} with {learner}")
        return await evaluate_leave_one_out(data_path, learner=learner)
    
    @mcp.tool()
    async def evaluate_test_on_training_tool(data_path: str, learner: str = 'tree'):
        logger.info(f"Evaluating test on training on {data_path} with {learner}")
        return await evaluate_test_on_training(data_path, learner=learner)
    
    @mcp.tool()
    async def evaluate_test_on_test_data_tool(data_path: str, train_path: str, learner: str = 'tree'):
        logger.info(f"Evaluating test on test data from {data_path} and {train_path} with {learner}")
        return await evaluate_test_on_test_data(data_path, train_path, learner=learner)
    
    @mcp.tool()
    async def evaluate_accuracy_tool(data_path: str, model_path: str):
        logger.info(f"Evaluating accuracy on {data_path} with model {model_path}")
        return await evaluate_accuracy(data_path, model_path)
    
    @mcp.tool()
    async def evaluate_rmse_tool(data_path: str, model_path: str):
        logger.info(f"Evaluating RMSE on {data_path} with model {model_path}")
        return await evaluate_rmse(data_path, model_path)
    
    @mcp.tool()
    async def evaluate_mae_tool(data_path: str, model_path: str):
        logger.info(f"Evaluating MAE on {data_path} with model {model_path}")
        return await evaluate_mae(data_path, model_path)
    
    @mcp.tool()
    async def evaluate_r2_tool(data_path: str, model_path: str):
        logger.info(f"Evaluating R2 on {data_path} with model {model_path}")
        return await evaluate_r2(data_path, model_path)
    
    @mcp.tool()
    async def evaluate_f1_tool(data_path: str, model_path: str):
        logger.info(f"Evaluating F1 on {data_path} with model {model_path}")
        return await evaluate_f1(data_path, model_path)

def run_server():
    """啟動 Orange3 MCP Server。"""
    logger.info("正在啟動 Orange3 MCP Server with stdio transport...")
    register_tools()
    try:
        mcp.run(transport='stdio')
    except KeyboardInterrupt:
        logger.info("正在關閉伺服器...")
    except Exception as e:
        logger.error(f"啟動伺服器時出錯: {str(e)}")
        sys.exit(1)
if __name__ == "__main__":
    run_server()