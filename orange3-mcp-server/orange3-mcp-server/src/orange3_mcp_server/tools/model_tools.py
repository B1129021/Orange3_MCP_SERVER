import os
import pickle
import numpy as np
from typing import Optional, List
from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
import scipy.optimize as opt

async def train_calibrated(data_path: str, base_learner: str = 'svm', method: str = 'sigmoid') -> str:
    """Train a calibrated classifier."""
    try:
        data = Table(data_path)
        if base_learner == 'svm':
            base = SVC(probability=True)
        elif base_learner == 'tree':
            base = DecisionTreeClassifier()
        else:
            raise ValueError(f"Unsupported base learner: {base_learner}")
        learner = CalibratedClassifierCV(base_estimator=base, method=method)
        X = data.X
        y = data.Y
        learner.fit(X, y)
        output_path = data_path.rsplit('.', 1)[0] + '_calibrated_model.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(learner, f)
        return output_path
    except Exception as e:
        raise Exception(f"Failed to train calibrated model: {str(e)}")

async def train_knn(data_path: str, n_neighbors: int = 5, metric: str = 'euclidean') -> str:
    """Train a k-nearest neighbors model."""
    try:
        data = Table(data_path)
        model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
        X = data.X
        y = data.Y
        model.fit(X, y)
        output_path = data_path.rsplit('.', 1)[0] + '_knn_model.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        return output_path
    except Exception as e:
        raise Exception(f"Failed to train KNN model: {str(e)}")

async def train_tree(data_path: str, target_column: str = 'car', max_depth: Optional[int] = None, min_samples_split: int = 2) -> str:
    """
    Train a decision tree model on the given data.
    
    Args:
        data_path: Path to the data file (.tab or .csv)
        target_column: Name of the target variable column
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum number of samples required to split an internal node
    
    Returns:
        Path to the saved model file
    """
    try:
        # Load data using Orange Table
        data = Table(data_path)
        
        # Ensure the domain is correctly set
        if target_column not in data.domain:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Create new domain with target as class variable
        attributes = [var for var in data.domain.variables if var.name != target_column]
        class_var = data.domain[target_column]
        if not class_var.is_discrete:
            raise ValueError(f"Target column '{target_column}' must be categorical for DecisionTreeClassifier")
        
        new_domain = Domain(attributes, class_vars=class_var)
        data = data.transform(new_domain)
        
        # Extract features (X) and target (y)
        X = data.X  # Feature matrix
        y = data.Y  # Target vector
        
        if X.shape[1] == 0:
            raise ValueError(f"No features available for training (shape={X.shape})")
        
        # Initialize and train the model
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
        model.fit(X, y)
        
        # Save the model
        output_path = data_path.rsplit('.', 1)[0] + '_tree_model.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        
        return output_path
    
    except Exception as e:
        raise Exception(f"Failed to train tree model: {str(e)}")

async def train_random_forest(data_path: str, n_estimators: int = 100, max_depth: Optional[int] = None) -> str:
    """Train a random forest model."""
    try:
        data = Table(data_path)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        X = data.X
        y = data.Y
        model.fit(X, y)
        output_path = data_path.rsplit('.', 1)[0] + '_rf_model.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        return output_path
    except Exception as e:
        raise Exception(f"Failed to train random forest model: {str(e)}")

async def train_gradient_boosting(data_path: str, n_estimators: int = 100, learning_rate: float = 0.1) -> str:
    """Train a gradient boosting model."""
    try:
        data = Table(data_path)
        model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        X = data.X
        y = data.Y
        model.fit(X, y)
        output_path = data_path.rsplit('.', 1)[0] + '_gb_model.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        return output_path
    except Exception as e:
        raise Exception(f"Failed to train gradient boosting model: {str(e)}")

async def train_svm(data_path: str, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale') -> str:
    """Train an SVM model."""
    try:
        data = Table(data_path)
        model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
        X = data.X
        y = data.Y
        model.fit(X, y)
        output_path = data_path.rsplit('.', 1)[0] + '_svm_model.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        return output_path
    except Exception as e:
        raise Exception(f"Failed to train SVM model: {str(e)}")

async def train_linear_regression(data_path: str, alpha: float = 0.0001, fit_intercept: bool = True) -> str:
    """Train a linear regression model."""
    try:
        data = Table(data_path)
        if not data.domain.class_var.is_continuous:
            raise ValueError("Target variable must be continuous for regression")
        model = LinearRegression(fit_intercept=fit_intercept)
        X = data.X
        y = data.Y
        model.fit(X, y)
        output_path = data_path.rsplit('.', 1)[0] + '_lr_model.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        return output_path
    except Exception as e:
        raise Exception(f"Failed to train linear regression model: {str(e)}")

async def train_logistic_regression(data_path: str, C: float = 1.0, solver: str = 'lbfgs') -> str:
    """Train a logistic regression model."""
    try:
        data = Table(data_path)
        model = LogisticRegression(C=C, solver=solver)
        X = data.X
        y = data.Y
        model.fit(X, y)
        output_path = data_path.rsplit('.', 1)[0] + '_logreg_model.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        return output_path
    except Exception as e:
        raise Exception(f"Failed to train logistic regression model: {str(e)}")

async def train_naive_bayes(data_path: str) -> str:
    """Train a naive Bayes model."""
    try:
        data = Table(data_path)
        model = GaussianNB()
        X = data.X
        y = data.Y
        model.fit(X, y)
        output_path = data_path.rsplit('.', 1)[0] + '_nb_model.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        return output_path
    except Exception as e:
        raise Exception(f"Failed to train naive Bayes model: {str(e)}")

async def train_adaboost(data_path: str, n_estimators: int = 50, learning_rate: float = 1.0) -> str:
    """Train an AdaBoost model."""
    try:
        data = Table(data_path)
        model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        X = data.X
        y = data.Y
        model.fit(X, y)
        output_path = data_path.rsplit('.', 1)[0] + '_adaboost_model.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        return output_path
    except Exception as e:
        raise Exception(f"Failed to train AdaBoost model: {str(e)}")

async def train_pls(data_path: str, n_components: int = 2) -> str:
    """Train a PLS regression model."""
    try:
        data = Table(data_path)
        if not data.domain.class_var.is_continuous:
            raise ValueError("Target variable must be continuous for PLS regression")
        model = PLSRegression(n_components=n_components)
        X = data.X
        y = data.Y
        model.fit(X, y)
        output_path = data_path.rsplit('.', 1)[0] + '_pls_model.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        return output_path
    except Exception as e:
        raise Exception(f"Failed to train PLS model: {str(e)}")

async def train_curve_fit(data_path: str, func_name: str = 'linear', p0: Optional[list] = None) -> str:
    """Train a curve fit model."""
    try:
        data = Table(data_path)
        X = data.X
        y = data.Y
        if X.shape[1] != 1:
            raise ValueError("Curve fitting requires exactly one feature")
        
        def linear(x, a, b):
            return a * x + b
        
        def quadratic(x, a, b, c):
            return a * x**2 + b * x + c
        
        func_map = {'linear': linear, 'quadratic': quadratic}
        if func_name not in func_map:
            raise ValueError(f"Unsupported function: {func_name}")
        
        func = func_map[func_name]
        popt, _ = opt.curve_fit(func, X.ravel(), y, p0=p0)
        
        output_path = data_path.rsplit('.', 1)[0] + '_curvefit_model.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump({'func': func, 'params': popt}, f)
        return output_path
    except Exception as e:
        raise Exception(f"Failed to train curve fit model: {str(e)}")

async def train_neural_network(data_path: str, hidden_layer_sizes: tuple = (100,), activation: str = 'relu') -> str:
    """Train a neural network model."""
    try:
        data = Table(data_path)
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation)
        X = data.X
        y = data.Y
        model.fit(X, y)
        output_path = data_path.rsplit('.', 1)[0] + '_nn_model.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        return output_path
    except Exception as e:
        raise Exception(f"Failed to train neural network model: {str(e)}")

async def train_sgd(data_path: str, loss: str = 'log', penalty: str = 'l2') -> str:
    """Train an SGD classifier."""
    try:
        data = Table(data_path)
        model = SGDClassifier(loss=loss, penalty=penalty)
        X = data.X
        y = data.Y
        model.fit(X, y)
        output_path = data_path.rsplit('.', 1)[0] + '_sgd_model.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        return output_path
    except Exception as e:
        raise Exception(f"Failed to train SGD model: {str(e)}")

async def train_stacking(data_path: str, learners: List[str], meta_learner: str = 'logistic') -> str:
    """Train a stacking classifier."""
    try:
        data = Table(data_path)
        estimators = []
        for learner in learners:
            if learner == 'tree':
                estimators.append(('tree', DecisionTreeClassifier()))
            elif learner == 'svm':
                estimators.append(('svm', SVC(probability=True)))
            elif learner == 'knn':
                estimators.append(('knn', KNeighborsClassifier()))
            else:
                raise ValueError(f"Unsupported learner: {learner}")
        
        if meta_learner == 'logistic':
            final_estimator = LogisticRegression()
        elif meta_learner == 'tree':
            final_estimator = DecisionTreeClassifier()
        else:
            raise ValueError(f"Unsupported meta-learner: {meta_learner}")
        
        model = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
        X = data.X
        y = data.Y
        model.fit(X, y)
        output_path = data_path.rsplit('.', 1)[0] + '_stacking_model.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        return output_path
    except Exception as e:
        raise Exception(f"Failed to train stacking model: {str(e)}")

async def save_model(model_path: str, output_path: str) -> str:
    """Save a model to the specified path."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        return output_path
    except Exception as e:
        raise Exception(f"Failed to save model: {str(e)}")

async def load_model(model_path: str) -> str:
    """Load a model from the specified path."""
    try:
        with open(model_path, 'rb') as f:
            pickle.load(f)
        return model_path
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")