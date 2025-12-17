import os
import json
import logging
import pandas as pd
from Orange.data import Table, Domain, DiscreteVariable
from Orange.data.io import TabReader
from Orange.data.util import get_unique_names
import numpy as np

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def load_file(file_path: str, file_type: str = 'auto') -> str:
    """Load a file into an Orange Table."""
    try:
        if file_type == 'auto':
            file_type = os.path.splitext(file_path)[1].lower().lstrip('.')
        if file_type in ['tab', 'tsv']:
            data = Table(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        logger.debug(f"Loaded file from {file_path}")
        return json.dumps({"data_path": file_path, "message": "File loaded successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to load file: {str(e)}")
        return json.dumps({"error": f"Failed to load file: {str(e)}"})

async def load_csv_file(file_path: str, delimiter: str = ',', encoding: str = 'utf-8') -> str:
    """Load a CSV file into an Orange Table."""
    try:
        df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
        output_path = f"{os.path.splitext(file_path)[0]}_converted.tab"
        Table.from_numpy(None, df.values).save(output_path)
        logger.debug(f"Loaded CSV from {file_path} and saved as {output_path}")
        return json.dumps({"data_path": output_path, "message": "CSV file loaded successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to load CSV file: {str(e)}")
        return json.dumps({"error": f"Failed to load CSV file: {str(e)}"})

async def load_datasets(dataset_name: str = 'iris') -> str:
    """Load a built-in Orange dataset."""
    try:
        data = Table(dataset_name)
        output_path = f"{dataset_name}.tab"
        data.save(output_path)
        logger.debug(f"Loaded dataset {dataset_name} and saved to {output_path}")
        return json.dumps({"data_path": output_path, "message": f"Dataset {dataset_name} loaded successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        return json.dumps({"error": f"Failed to load dataset: {str(e)}"})

async def load_data_table(data_path: str) -> str:
    """Load an Orange data table."""
    try:
        data = Table(data_path)
        logger.debug(f"Loaded data table from {data_path}")
        return json.dumps({"data_path": data_path, "message": "Data table loaded successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to load data table: {str(e)}")
        return json.dumps({"error": f"Failed to load data table: {str(e)}"})

async def get_data_info(data_path: str) -> str:
    """Get information about an Orange data table."""
    try:
        data = Table(data_path)
        info = {
            "n_rows": len(data),
            "n_columns": len(data.domain.attributes) + len(data.domain.metas) + (1 if data.domain.class_var else 0),
            "attributes": [var.name for var in data.domain.attributes],
            "class_var": data.domain.class_var.name if data.domain.class_var else None,
            "metas": [var.name for var in data.domain.metas]
        }
        logger.debug(f"Got data info for {data_path}: {info}")
        return json.dumps({"info": info, "message": "Data info retrieved successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to get data info: {str(e)}")
        return json.dumps({"error": f"Failed to get data info: {str(e)}"})

async def rank_features(data_path: str, method: str = 'gain_ratio', k: int = 5) -> str:
    """Rank features using a specified method."""
    try:
        from Orange.preprocess.score import GainRatio
        data = Table(data_path)
        if method == 'gain_ratio':
            scorer = GainRatio()
        else:
            raise ValueError(f"Unsupported ranking method: {method}")
        scores = [scorer(data, attr) for attr in data.domain.attributes]
        ranked = sorted(zip(data.domain.attributes, scores), key=lambda x: x[1], reverse=True)[:k]
        result = [{"feature": attr.name, "score": float(score)} for attr, score in ranked]
        logger.debug(f"Ranked features for {data_path}: {result}")
        return json.dumps({"ranked_features": result, "message": "Features ranked successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to rank features: {str(e)}")
        return json.dumps({"error": f"Failed to rank features: {str(e)}"})

async def edit_domain(data_path: str, new_domain: dict, output_path: str = None) -> str:
    """Edit the domain of an Orange data table and save the transformed data."""
    try:
        data = Table(data_path)
        attributes = []
        class_var = None
        for var in data.domain.variables + data.domain.metas:
            if var.name in new_domain.get('features', []):
                attributes.append(DiscreteVariable(var.name, values=var.values))
            elif var.name == new_domain.get('target'):
                class_var = DiscreteVariable(var.name, values=var.values)
        if class_var is None:
            raise ValueError(f"Class variable '{new_domain.get('target')}' not found.")
        new_domain = Domain(attributes, class_var)
        new_data = data.transform(new_domain)
        if output_path is None:
            output_path = f"{os.path.splitext(data_path)[0]}_processed.tab"
        new_data.save(output_path)
        logger.debug(f"Domain edited and data saved to {output_path}")
        return json.dumps({"output_path": output_path, "message": "Domain edited and data saved successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to edit domain: {str(e)}")
        return json.dumps({"error": f"Failed to edit domain: {str(e)}"})

async def get_feature_statistics(data_path: str, features: list = None) -> str:
    """Get statistics for specified features."""
    try:
        data = Table(data_path)
        if features is None:
            features = [var.name for var in data.domain.attributes]
        stats = {}
        for feature in features:
            col = data.get_column_view(feature)[0]
            stats[feature] = {
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "min": float(np.min(col)),
                "max": float(np.max(col))
            }
        logger.debug(f"Got feature statistics for {data_path}: {stats}")
        return json.dumps({"statistics": stats, "message": "Feature statistics retrieved successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to get feature statistics: {str(e)}")
        return json.dumps({"error": f"Failed to get feature statistics: {str(e)}"})

async def save_data(data_path: str, output_path: str, format: str = 'tab') -> str:
    """Save an Orange data table to a file."""
    try:
        data = Table(data_path)
        if format == 'tab':
            data.save(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        logger.debug(f"Saved data from {data_path} to {output_path}")
        return json.dumps({"output_path": output_path, "message": "Data saved successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to save data: {str(e)}")
        return json.dumps({"error": f"Failed to save data: {str(e)}"})
