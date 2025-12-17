import json
import logging
import numpy as np
import pandas as pd
from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
from Orange.preprocess import Impute, Continuize, Discretize, Randomize
from Orange.data.util import get_unique_names
import os

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def data_sampler(data_path: str, proportion: float = 0.1, stratified: bool = True, replace: bool = False) -> str:
    """Sample a subset of the data."""
    try:
        data = Table(data_path)
        if stratified and data.domain.class_var:
            from Orange.preprocess import Randomize
            randomizer = Randomize(random_state=42)
            indices = []
            for cls in data.domain.class_var.values:
                cls_data = data[data.Y == data.domain.class_var.to_val(cls)]
                cls_indices = np.random.choice(len(cls_data), int(len(cls_data) * proportion), replace=replace)
                indices.extend(cls_indices)
        else:
            indices = np.random.choice(len(data), int(len(data) * proportion), replace=replace)
        sampled_data = data[indices]
        output_path = f"{os.path.splitext(data_path)[0]}_sampled.tab"
        sampled_data.save(output_path)
        logger.debug(f"Sampled data saved to {output_path}")
        return json.dumps({"output_path": output_path, "message": "Data sampled successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to sample data: {str(e)}")
        return json.dumps({"error": f"Failed to sample data: {str(e)}"})

async def select_columns(data_path: str, columns: list) -> str:
    """Select specified columns from the data."""
    try:
        data = Table(data_path)
        selected_vars = [var for var in data.domain.attributes + data.domain.metas if var.name in columns]
        class_var = data.domain.class_var if data.domain.class_var and data.domain.class_var.name in columns else None
        new_domain = Domain(selected_vars, class_var)
        new_data = data.transform(new_domain)
        output_path = f"{os.path.splitext(data_path)[0]}_selected_columns.tab"
        new_data.save(output_path)
        logger.debug(f"Selected columns {columns} saved to {output_path}")
        return json.dumps({"output_path": output_path, "message": "Columns selected successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to select columns: {str(e)}")
        return json.dumps({"error": f"Failed to select columns: {str(e)}"})

async def select_rows(data_path: str, row_indices: list) -> str:
    """Select specified rows from the data."""
    try:
        data = Table(data_path)
        new_data = data[row_indices]
        output_path = f"{os.path.splitext(data_path)[0]}_selected_rows.tab"
        new_data.save(output_path)
        logger.debug(f"Selected rows {row_indices} saved to {output_path}")
        return json.dumps({"output_path": output_path, "message": "Rows selected successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to select rows: {str(e)}")
        return json.dumps({"error": f"Failed to select rows: {str(e)}"})

async def transpose(data_path: str) -> str:
    """Transpose the data table."""
    try:
        data = Table(data_path)
        new_data = Table.from_numpy(None, data.X.T)
        output_path = f"{os.path.splitext(data_path)[0]}_transposed.tab"
        new_data.save(output_path)
        logger.debug(f"Transposed data saved to {output_path}")
        return json.dumps({"output_path": output_path, "message": "Data transposed successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to transpose data: {str(e)}")
        return json.dumps({"error": f"Failed to transpose data: {str(e)}"})

async def merge_data(data_path1: str, data_path2: str, on: list = None, how: str = 'inner') -> str:
    """Merge two data tables."""
    try:
        data1 = Table(data_path1)
        data2 = Table(data_path2)
        df1 = data1.to_pandas_dfs()
        df2 = data2.to_pandas_dfs()
        merged_df = pd.merge(df1, df2, on=on, how=how)
        new_data = Table.from_pandas(merged_df)
        output_path = f"{os.path.splitext(data_path1)[0]}_merged.tab"
        new_data.save(output_path)
        logger.debug(f"Merged data saved to {output_path}")
        return json.dumps({"output_path": output_path, "message": "Data merged successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to merge data: {str(e)}")
        return json.dumps({"error": f"Failed to merge data: {str(e)}"})

async def concatenate(data_path1: str, data_path2: str, axis: int = 0) -> str:
    """Concatenate two data tables."""
    try:
        data1 = Table(data_path1)
        data2 = Table(data_path2)
        new_data = Table.concatenate([data1, data2], axis=axis)
        output_path = f"{os.path.splitext(data_path1)[0]}_concatenated.tab"
        new_data.save(output_path)
        logger.debug(f"Concatenated data saved to {output_path}")
        return json.dumps({"output_path": output_path, "message": "Data concatenated successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to concatenate data: {str(e)}")
        return json.dumps({"error": f"Failed to concatenate data: {str(e)}"})

async def select_by_data_index(data_path: str, index: int) -> str:
    """Select a single row by index."""
    try:
        data = Table(data_path)
        new_data = data[index:index+1]
        output_path = f"{os.path.splitext(data_path)[0]}_index_{index}.tab"
        new_data.save(output_path)
        logger.debug(f"Selected index {index} saved to {output_path}")
        return json.dumps({"output_path": output_path, "message": "Index selected successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to select by index: {str(e)}")
        return json.dumps({"error": f"Failed to select by index: {str(e)}"})

async def unique(data_path: str, columns: list) -> str:
    """Get unique values for specified columns."""
    try:
        data = Table(data_path)
        unique_vals = {col: list(set(data.get_column_view(col)[0].astype(str))) for col in columns}
        logger.debug(f"Unique values for {columns}: {unique_vals}")
        return json.dumps({"unique_values": unique_vals, "message": "Unique values retrieved successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to get unique values: {str(e)}")
        return json.dumps({"error": f"Failed to get unique values: {str(e)}"})

async def aggregate_columns(data_path: str, aggregation: str = 'mean') -> str:
    """Aggregate columns using a specified method."""
    try:
        data = Table(data_path)
        if aggregation == 'mean':
            agg_data = np.mean(data.X, axis=0)
        elif aggregation == 'sum':
            agg_data = np.sum(data.X, axis=0)
        elif aggregation == 'min':
            agg_data = np.min(data.X, axis=0)
        elif aggregation == 'max':
            agg_data = np.max(data.X, axis=0)
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        new_data = Table.from_numpy(None, agg_data.reshape(1, -1), domain=Domain(data.domain.attributes))
        output_path = f"{os.path.splitext(data_path)[0]}_aggregated.tab"
        new_data.save(output_path)
        logger.debug(f"Aggregated data saved to {output_path}")
        return json.dumps({"output_path": output_path, "message": "Columns aggregated successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to aggregate columns: {str(e)}")
        return json.dumps({"error": f"Failed to aggregate columns: {str(e)}"})

async def group_by_tool(data_path: str, group_by: list) -> str:
    """Group data by specified columns."""
    try:
        data = Table(data_path)
        df = data.to_pandas_dfs()
        grouped = df.groupby(group_by).size().reset_index(name='count')
        new_data = Table.from_pandas(grouped)
        output_path = f"{os.path.splitext(data_path)[0]}_grouped.tab"
        new_data.save(output_path)
        logger.debug(f"Grouped data saved to {output_path}")
        return json.dumps({"output_path": output_path, "message": "Data grouped successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to group data: {str(e)}")
        return json.dumps({"error": f"Failed to group data: {str(e)}"})

async def pivot_table(data_path: str, values: str, index: str, columns: str) -> str:
    """Create a pivot table."""
    try:
        data = Table(data_path)
        df = data.to_pandas_dfs()
        pivot = pd.pivot_table(df, values=values, index=index, columns=columns, aggfunc='mean')
        new_data = Table.from_pandas(pivot.reset_index())
        output_path = f"{os.path.splitext(data_path)[0]}_pivot.tab"
        new_data.save(output_path)
        logger.debug(f"Pivot table saved to {output_path}")
        return json.dumps({"output_path": output_path, "message": "Pivot table created successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to create pivot table: {str(e)}")
        return json.dumps({"error": f"Failed to create pivot table: {str(e)}"})

async def apply_domain(data_path: str, new_domain: dict) -> str:
    """Apply a new domain to the data."""
    try:
        data = Table(data_path)
        attributes = [DiscreteVariable(name, values=vals) if vals else ContinuousVariable(name) 
                      for name, vals in new_domain.get('attributes', {}).items()]
        class_var = DiscreteVariable(new_domain['class_var']['name'], values=new_domain['class_var']['values']) if 'class_var' in new_domain else None
        new_domain = Domain(attributes, class_var)
        new_data = data.transform(new_domain)
        output_path = f"{os.path.splitext(data_path)[0]}_applied_domain.tab"
        new_data.save(output_path)
        logger.debug(f"Applied domain to {data_path}, saved to {output_path}")
        return json.dumps({"output_path": output_path, "message": "Domain applied successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to apply domain: {str(e)}")
        return json.dumps({"error": f"Failed to apply domain: {str(e)}"})

async def preprocess(data_path: str, method: str = 'normalize') -> str:
    """Preprocess the data using a specified method."""
    try:
        data = Table(data_path)
        if method == 'normalize':
            from Orange.preprocess import Normalize
            preprocessor = Normalize()
            new_data = preprocessor(data)
        elif method == 'standardize':
            from Orange.preprocess import Scale
            preprocessor = Scale()
            new_data = preprocessor(data)
        else:
            raise ValueError(f"Unsupported preprocess method: {method}")
        output_path = f"{os.path.splitext(data_path)[0]}_preprocessed.tab"
        new_data.save(output_path)
        logger.debug(f"Preprocessed data saved to {output_path}")
        return json.dumps({"output_path": output_path, "message": "Data preprocessed successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to preprocess data: {str(e)}")
        return json.dumps({"error": f"Failed to preprocess data: {str(e)}"})

async def impute(data_path: str, method: str = 'mean') -> str:
    """Impute missing values in the data."""
    try:
        data = Table(data_path)
        imputer = Impute(method=method)
        new_data = imputer(data)
        output_path = f"{os.path.splitext(data_path)[0]}_imputed.tab"
        new_data.save(output_path)
        logger.debug(f"Imputed data saved to {output_path}")
        return json.dumps({"output_path": output_path, "message": "Data imputed successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to impute data: {str(e)}")
        return json.dumps({"error": f"Failed to impute data: {str(e)}"})

async def continuize(data_path: str) -> str:
    """Continuize discrete variables in the data."""
    try:
        data = Table(data_path)
        continuizer = Continuize()
        new_data = continuizer(data)
        output_path = f"{os.path.splitext(data_path)[0]}_continuized.tab"
        new_data.save(output_path)
        logger.debug(f"Continuized data saved to {output_path}")
        return json.dumps({"output_path": output_path, "message": "Data continuized successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to continuize data: {str(e)}")
        return json.dumps({"error": f"Failed to continuize data: {str(e)}"})

async def discretize(data_path: str, n_bins: int = 5) -> str:
    """Discretize continuous variables in the data."""
    try:
        data = Table(data_path)
        discretizer = Discretize()
        new_data = discretizer(data)
        output_path = f"{os.path.splitext(data_path)[0]}_discretized.tab"
        new_data.save(output_path)
        logger.debug(f"Discretized data saved to {output_path}")
        return json.dumps({"output_path": output_path, "message": "Data discretized successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to discretize data: {str(e)}")
        return json.dumps({"error": f"Failed to discretize data: {str(e)}"})

async def randomize(data_path: str, rand_type: str = 'all', rand_seed: int = 42) -> str:
    """Randomize the data order."""
    try:
        data = Table(data_path)
        randomizer = Randomize(rand_seed=rand_seed, randomize_class=(rand_type in ['all', 'class']))
        new_data = randomizer(data)
        output_path = f"{os.path.splitext(data_path)[0]}_randomized.tab"
        new_data.save(output_path)
        logger.debug(f"Randomized data saved to {output_path}")
        return json.dumps({"output_path": output_path, "message": "Data randomized successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to randomize data: {str(e)}")
        return json.dumps({"error": f"Failed to randomize data: {str(e)}"})

async def purge_domain(data_path: str, remove_unused: bool = True, remove_constants: bool = True) -> str:
    """Purge unused or constant variables from the domain."""
    try:
        data = Table(data_path)
        attributes = [var for var in data.domain.attributes if not (remove_constants and len(set(data.get_column_view(var.name)[0])) == 1)]
        if remove_unused:
            attributes = [var for var in attributes if len(set(data.get_column_view(var.name)[0])) > 1]
        new_domain = Domain(attributes, data.domain.class_var)
        new_data = data.transform(new_domain)
        output_path = f"{os.path.splitext(data_path)[0]}_purged.tab"
        new_data.save(output_path)
        logger.debug(f"Purged domain saved to {output_path}")
        return json.dumps({"output_path": output_path, "message": "Domain purged successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to purge domain: {str(e)}")
        return json.dumps({"error": f"Failed to purge domain: {str(e)}"})

async def melt(data_path: str, id_vars: list, value_vars: list = None) -> str:
    """Melt the data table."""
    try:
        data = Table(data_path)
        df = data.to_pandas_dfs()
        value_vars = value_vars or [var.name for var in data.domain.attributes if var.name not in id_vars]
        melted_df = pd.melt(df, id_vars=id_vars, value_vars=value_vars)
        new_data = Table.from_pandas(melted_df)
        output_path = f"{os.path.splitext(data_path)[0]}_melted.tab"
        new_data.save(output_path)
        logger.debug(f"Melted data saved to {output_path}")
        return json.dumps({"output_path": output_path, "message": "Data melted successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to melt data: {str(e)}")
        return json.dumps({"error": f"Failed to melt data: {str(e)}"})

async def formula(data_path: str, formula_expr: str, new_col_name: str) -> str:
    """Apply a formula to create a new column."""
    try:
        data = Table(data_path)
        df = data.to_pandas_dfs()
        df[new_col_name] = df.eval(formula_expr)
        new_data = Table.from_pandas(df)
        output_path = f"{os.path.splitext(data_path)[0]}_formula.tab"
        new_data.save(output_path)
        logger.debug(f"Formula applied, new column {new_col_name} saved to {output_path}")
        return json.dumps({"output_path": output_path, "message": "Formula applied successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to apply formula: {str(e)}")
        return json.dumps({"error": f"Failed to apply formula: {str(e)}"})

async def create_class(data_path: str, class_name: str, values: list) -> str:
    """Create a new class variable."""
    try:
        data = Table(data_path)
        new_class_var = DiscreteVariable(class_name, values=values)
        new_domain = Domain(data.domain.attributes, new_class_var)
        new_data = data.transform(new_domain)
        output_path = f"{os.path.splitext(data_path)[0]}_new_class.tab"
        new_data.save(output_path)
        logger.debug(f"New class {class_name} created, saved to {output_path}")
        return json.dumps({"output_path": output_path, "message": "Class created successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to create class: {str(e)}")
        return json.dumps({"error": f"Failed to create class: {str(e)}"})

async def create_instance(data_path: str, values: dict) -> str:
    """Create a new instance and append it to the data."""
    try:
        data = Table(data_path)
        new_instance = np.array([[values.get(var.name, np.nan) for var in data.domain.attributes]])
        new_data = Table.from_numpy(data.domain, np.vstack([data.X, new_instance]), data.Y)
        output_path = f"{os.path.splitext(data_path)[0]}_new_instance.tab"
        new_data.save(output_path)
        logger.debug(f"New instance added, saved to {output_path}")
        return json.dumps({"output_path": output_path, "message": "Instance created successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to create instance: {str(e)}")
        return json.dumps({"error": f"Failed to create instance: {str(e)}"})

async def python_script(code: str, data_path: str = None) -> str:
    """Execute a user-provided Python script."""
    try:
        restricted_globals = {'Table': Table, 'np': np, 'pd': pd}
        locals_dict = {}
        if data_path:
            data = Table(data_path)
            restricted_globals['data'] = data
        exec(code, restricted_globals, locals_dict)
        result = locals_dict.get('result', None)
        if isinstance(result, Table):
            output_path = f"{os.path.splitext(data_path or 'script')[0]}_script_output.tab"
            result.save(output_path)
            logger.debug(f"Python script executed, output saved to {output_path}")
            return json.dumps({"output_path": output_path, "message": "Python script executed successfully"}, indent=2)
        else:
            logger.debug("Python script executed, no Table output")
            return json.dumps({"result": str(result), "message": "Python script executed successfully"}, indent=2)
    except Exception as e:
        logger.error(f"Failed to execute Python script: {str(e)}")
        return json.dumps({"error": f"Failed to execute Python script: {str(e)}"})