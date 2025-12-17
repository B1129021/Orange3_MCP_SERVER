# Iris Dataset MCP Demo

This is a minimal end-to-end workflow demo using the built-in **Iris** dataset.

## Suggested flow

1. Load dataset: `load_datasets_tool("iris")` (or your tool name exposed in client)
2. Inspect: `get_data_info_tool(...)`
3. Preprocess: `preprocess_tool(...)` / `impute_tool(...)`
4. Train: `train_tree_tool(...)` / `train_svm_tool(...)` / etc.
5. Evaluate: `cross_validation_tool(...)`

> Exact tool names are the ones registered in `src/orange3_mcp_server/main.py`.
> In Claude Desktop you can ask: “List the tools of orange3 server”.
