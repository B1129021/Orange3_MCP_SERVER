# 🍊 Orange3 MCP Server

An **Orange3-based MCP (Model Context Protocol) Server** that exposes classic data mining and ML workflows as MCP tools.
Use it from MCP clients (e.g., **Gemini cli**) via **stdio transport**.

---

## ✨ Features

- 📂 **Data Loading & Inspection**
  - Load CSV / Tab files
  - Load built-in Orange datasets (e.g., Iris)
  - Dataset summary, statistics, feature ranking, save/export

- 🔄 **Data Transformation**
  - Column/row selection, sampling, merge/concat, pivot/melt
  - Preprocess, impute, discretize, continuize, randomize, etc.
  - `python_script` (restricted exec) for advanced transforms

- 🤖 **Modeling**
  - Train models (Tree / kNN / SVM / Regression …)
  - Save/load models

- 📊 **Evaluation**
  - Cross-validation, leave-one-out, test evaluation
  - Common metrics (accuracy / RMSE / MAE / F1 …)

- 🧠 **Unsupervised Learning**
  - PCA, t-SNE
  - KMeans, DBSCAN, SOM
  - Distance matrix

- 📈 **Visualization (descriptive outputs)**
  - Scatter/heatmap/silhouette style outputs (returns data/summary for plotting)

---

## 🔌 Supported MCP Clients

- ✅ Claude Desktop
- ✅ Gemini cli
- ✅ Any MCP client that supports **stdio** servers

---

## 🚀 Quick Start (Conda, recommended)

### 1). Get the project

```bash
git clone https://github.com/B1129021/Orange3_MCP_SERVER.git
cd .../Orange3_MCP_SERVER/orange3-mcp-server/orange3-mcp-server
```

### 2) Create the environment

```bash
conda env create -f environment.yml
conda activate orange3_env
```

### 3) Install this server (editable mode)

```bash
pip install -e .
```

### 4) Run the MCP server

```bash
orange3-mcp-server
```

or

```bash
python -m orange3_mcp_server
```

---

## 🧩 Claude Desktop Configuration

Open **Claude Desktop → Settings → Developer → Edit Config** and add:

```json
{
  "mcpServers": {
    "orange3": {
      "command": "orange3-mcp-server"
    }
  }
}
```

> If you prefer to run from source without installing the console script, use:
>
> ```json
> {
>   "mcpServers": {
>     "orange3-dev": {
>       "command": "python",
>       "args": ["-m", "orange3_mcp_server"]
>     }
>   }
> }
> ```

Restart Claude Desktop after saving.

---

## 📚 Tool Categories (overview)

This server registers tools in these modules:

- `data_tools.py`
- `transform_tools.py`
- `model_tools.py`
- `evaluate_tools.py`
- `unsupervised_tools.py`
- `visualize_tools.py`

> Tip: In Claude, ask “List available tools” or “What tools does orange3 provide?” to explore.

---

## 📖 Example Workflow

See: `examples/iris_demo.md`

---

## ⚠️ Security Notice

This server provides a `python_script` tool that executes Python code in a restricted environment.
Use it **only** in trusted/local scenarios. Do not run untrusted code.

---

## 🧪 Dev Tips

- Verify Orange3 imports:

```bash
python -c "import Orange; print(Orange.__version__)"
```

- Run server directly:

```bash
python -m orange3_mcp_server
```

---

## 📄 License

MIT License
