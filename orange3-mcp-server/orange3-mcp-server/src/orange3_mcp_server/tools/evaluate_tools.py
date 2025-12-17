import os
import json
import logging
import tempfile
import pickle
import numpy as np
from Orange.data import Table
from Orange.evaluation import CrossValidation, LeaveOneOut, TestOnTrainingData, TestOnTestData, \
    R2, RMSE, MAE, CA, F1
from Orange.classification import TreeLearner, RandomForestLearner
from Orange.regression import MeanLearner, LinearRegressionLearner

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def evaluate_cross_validation(data_path: str, learner: str = 'tree', n_folds: int = 5, **kwargs) -> str:
    """Evaluate model performance using Cross-Validation.
    
    Args:
        data_path: Input data file path.
        learner: Learner type ('tree', 'random_forest', 'mean', 'linear').
        n_folds: Number of folds for cross-validation.
        **kwargs: Additional params for learner.
    """
    try:
        data = Table(data_path)
        if learner == 'tree':
            model_learner = TreeLearner(**kwargs)
        elif learner == 'random_forest':
            model_learner = RandomForestLearner(**kwargs)
        elif learner == 'mean':
            model_learner = MeanLearner(**kwargs)
        elif learner == 'linear':
            model_learner = LinearRegressionLearner(**kwargs)
        else:
            raise ValueError("Unsupported learner type")
        results = CrossValidation(data, [model_learner], k=n_folds)
        if data.domain.class_var.is_continuous:
            rmse = RMSE(results)[0]
            r2 = R2(results)[0]
            metric = {"RMSE": float(rmse), "R2": float(r2)}
        else:
            accuracy = CA(results)[0]
            f1 = F1(results)[0]
            metric = {"Accuracy": float(accuracy), "F1": float(f1)}
        suggestion = f"If metrics are poor (e.g., RMSE: {rmse:.2f} or Accuracy: {accuracy:.2f}), try increasing n_folds (current: {n_folds}) or tuning learner parameters."
        return json.dumps({
            "metric": metric,
            "tuning_suggestion": suggestion,
            "message": "Cross-validation evaluation completed successfully."
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to evaluate with Cross Validation: {str(e)}")
        return json.dumps({"error": f"Failed to evaluate with Cross Validation: {str(e)}"})

async def evaluate_leave_one_out(data_path: str, learner: str = 'tree', **kwargs) -> str:
    """Evaluate model performance using Leave-One-Out.
    
    Args:
        data_path: Input data file path.
        learner: Learner type ('tree', 'random_forest', 'mean', 'linear').
        **kwargs: Additional params for learner.
    """
    try:
        data = Table(data_path)
        if learner == 'tree':
            model_learner = TreeLearner(**kwargs)
        elif learner == 'random_forest':
            model_learner = RandomForestLearner(**kwargs)
        elif learner == 'mean':
            model_learner = MeanLearner(**kwargs)
        elif learner == 'linear':
            model_learner = LinearRegressionLearner(**kwargs)
        else:
            raise ValueError("Unsupported learner type")
        results = LeaveOneOut(data, [model_learner])
        if data.domain.class_var.is_continuous:
            rmse = RMSE(results)[0]
            r2 = R2(results)[0]
            metric = {"RMSE": float(rmse), "R2": float(r2)}
        else:
            accuracy = CA(results)[0]
            f1 = F1(results)[0]
            metric = {"Accuracy": float(accuracy), "F1": float(f1)}
        suggestion = "If metrics are poor, consider using a different learner or preprocessing data to reduce variance."
        return json.dumps({
            "metric": metric,
            "tuning_suggestion": suggestion,
            "message": "Leave-one-out evaluation completed successfully."
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to evaluate with Leave One Out: {str(e)}")
        return json.dumps({"error": f"Failed to evaluate with Leave One Out: {str(e)}"})

async def evaluate_test_on_training(data_path: str, learner: str = 'tree', **kwargs) -> str:
    """Evaluate model performance on training data.
    
    Args:
        data_path: Input data file path.
        learner: Learner type ('tree', 'random_forest', 'mean', 'linear').
        **kwargs: Additional params for learner.
    """
    try:
        data = Table(data_path)
        if learner == 'tree':
            model_learner = TreeLearner(**kwargs)
        elif learner == 'random_forest':
            model_learner = RandomForestLearner(**kwargs)
        elif learner == 'mean':
            model_learner = MeanLearner(**kwargs)
        elif learner == 'linear':
            model_learner = LinearRegressionLearner(**kwargs)
        else:
            raise ValueError("Unsupported learner type")
        results = TestOnTrainingData(data, [model_learner])
        if data.domain.class_var.is_continuous:
            rmse = RMSE(results)[0]
            r2 = R2(results)[0]
            metric = {"RMSE": float(rmse), "R2": float(r2)}
        else:
            accuracy = CA(results)[0]
            f1 = F1(results)[0]
            metric = {"Accuracy": float(accuracy), "F1": float(f1)}
        suggestion = "If metrics are overly optimistic, this may indicate overfitting; try a more complex learner or regularization."
        return json.dumps({
            "metric": metric,
            "tuning_suggestion": suggestion,
            "message": "Test on training data evaluation completed successfully."
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to evaluate with Test on Training Data: {str(e)}")
        return json.dumps({"error": f"Failed to evaluate with Test on Training Data: {str(e)}"})

async def evaluate_test_on_test_data(data_path: str, train_path: str, learner: str = 'tree', **kwargs) -> str:
    """Evaluate model performance on a separate test set.
    
    Args:
        data_path: Input data file path for training.
        train_path: Input data file path for testing.
        learner: Learner type ('tree', 'random_forest', 'mean', 'linear').
        **kwargs: Additional params for learner.
    """
    try:
        train_data = Table(data_path)
        test_data = Table(train_path)
        if learner == 'tree':
            model_learner = TreeLearner(**kwargs)
        elif learner == 'random_forest':
            model_learner = RandomForestLearner(**kwargs)
        elif learner == 'mean':
            model_learner = MeanLearner(**kwargs)
        elif learner == 'linear':
            model_learner = LinearRegressionLearner(**kwargs)
        else:
            raise ValueError("Unsupported learner type")
        model = model_learner(train_data)
        results = TestOnTestData(train_data, test_data, [model])
        if train_data.domain.class_var.is_continuous:
            rmse = RMSE(results)[0]
            r2 = R2(results)[0]
            metric = {"RMSE": float(rmse), "R2": float(r2)}
        else:
            accuracy = CA(results)[0]
            f1 = F1(results)[0]
            metric = {"Accuracy": float(accuracy), "F1": float(f1)}
        suggestion = "If metrics are poor, ensure test and train data are from the same distribution or adjust learner parameters."
        return json.dumps({
            "metric": metric,
            "tuning_suggestion": suggestion,
            "message": "Test on test data evaluation completed successfully."
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to evaluate with Test on Test Data: {str(e)}")
        return json.dumps({"error": f"Failed to evaluate with Test on Test Data: {str(e)}"})

async def evaluate_accuracy(data_path: str, model_path: str) -> str:
    """Evaluate model accuracy.
    
    Args:
        data_path: Input data file path.
        model_path: Path to saved model.
    """
    try:
        data = Table(data_path)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        results = TestOnTrainingData(data, [model])
        accuracy = CA(results)[0]
        f1 = F1(results)[0]
        suggestion = f"If accuracy is low ({accuracy:.2f}), consider retraining with different parameters or preprocessing data."
        return json.dumps({
            "metric": {"Accuracy": float(accuracy), "F1": float(f1)},
            "tuning_suggestion": suggestion,
            "message": "Accuracy evaluation completed successfully."
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to evaluate accuracy: {str(e)}")
        return json.dumps({"error": f"Failed to evaluate accuracy: {str(e)}"})

async def evaluate_rmse(data_path: str, model_path: str) -> str:
    """Evaluate model RMSE.
    
    Args:
        data_path: Input data file path.
        model_path: Path to saved model.
    """
    try:
        data = Table(data_path)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        results = TestOnTrainingData(data, [model])
        rmse = RMSE(results)[0]
        r2 = R2(results)[0]
        suggestion = f"If RMSE is high ({rmse:.2f}), consider adjusting model parameters or improving data quality."
        return json.dumps({
            "metric": {"RMSE": float(rmse), "R2": float(r2)},
            "tuning_suggestion": suggestion,
            "message": "RMSE evaluation completed successfully."
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to evaluate RMSE: {str(e)}")
        return json.dumps({"error": f"Failed to evaluate RMSE: {str(e)}"})

async def evaluate_mae(data_path: str, model_path: str) -> str:
    """Evaluate model MAE.
    
    Args:
        data_path: Input data file path.
        model_path: Path to saved model.
    """
    try:
        data = Table(data_path)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        results = TestOnTrainingData(data, [model])
        mae = MAE(results)[0]
        r2 = R2(results)[0]
        suggestion = f"If MAE is high ({mae:.2f}), try refining the model with better feature selection or regularization."
        return json.dumps({
            "metric": {"MAE": float(mae), "R2": float(r2)},
            "tuning_suggestion": suggestion,
            "message": "MAE evaluation completed successfully."
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to evaluate MAE: {str(e)}")
        return json.dumps({"error": f"Failed to evaluate MAE: {str(e)}"})

async def evaluate_r2(data_path: str, model_path: str) -> str:
    """Evaluate model R-squared.
    
    Args:
        data_path: Input data file path.
        model_path: Path to saved model.
    """
    try:
        data = Table(data_path)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        results = TestOnTrainingData(data, [model])
        r2 = R2(results)[0]
        rmse = RMSE(results)[0]
        suggestion = f"If R2 is low ({r2:.2f}), consider increasing model complexity or improving data features."
        return json.dumps({
            "metric": {"R2": float(r2), "RMSE": float(rmse)},
            "tuning_suggestion": suggestion,
            "message": "R2 evaluation completed successfully."
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to evaluate R2: {str(e)}")
        return json.dumps({"error": f"Failed to evaluate R2: {str(e)}"})

async def evaluate_f1(data_path: str, model_path: str) -> str:
    """Evaluate model F1 score.
    
    Args:
        data_path: Input data file path.
        model_path: Path to saved model.
    """
    try:
        data = Table(data_path)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        results = TestOnTrainingData(data, [model])
        f1 = F1(results)[0]
        accuracy = CA(results)[0]
        suggestion = f"If F1 is low ({f1:.2f}), adjust model to balance precision and recall, or check class imbalance."
        return json.dumps({
            "metric": {"F1": float(f1), "Accuracy": float(accuracy)},
            "tuning_suggestion": suggestion,
            "message": "F1 evaluation completed successfully."
        }, indent=2)
    except Exception as e:
        logger.error(f"Failed to evaluate F1: {str(e)}")
        return json.dumps({"error": f"Failed to evaluate F1: {str(e)}"})