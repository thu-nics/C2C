# Workflow Scripts

Scripts for running multi-agent research workflows with local search.

## Prerequisites

- Activate the conda environment:
  ```bash
  conda activate c2c
  ```

- Build the search database first (one-time):
  ```bash
  python script/workflow/build_search_database.py
  ```

## Server Setup

### 1. Launch Embedding Server (for local search)

```bash
CUDA_VISIBLE_DEVICES=2 python -m sglang.launch_server \
    --model-path Qwen/Qwen3-Embedding-0.6B \
    --host 0.0.0.0 \
    --port 30001 \
    --is-embedding
```

alternatively:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
    --model-path Qwen/Qwen3-Embedding-0.6B \
    --host 0.0.0.0 \
    --port 30001 \
    --is-embedding \
    --tp-size 2 \
    --mem-fraction-static 0.2
```

### 2. Launch LLM Server (for agents)

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
    --model-path Qwen/Qwen3-32B \
    --host 0.0.0.0 \
    --tp-size 2 \
    --tool-call-parser qwen \
    --port 30000
```

## Running Examples

### Test Local Search Tool

```bash
python script/workflow/examples/example_local_search_tool.py
```

This tests the `search_engine` function as a CAMEL `FunctionTool` with a `ChatAgent`.

### Run Subagent Research

```bash
python script/workflow/subagent_research.py
```

This runs the multi-agent research workflow that:
1. Decomposes a question into search subtasks
2. Uses search agents to find information
3. Synthesizes a final answer

## Configuration

| Service | Port | Model |
|---------|------|-------|
| Embedding | 30001 | `Qwen/Qwen3-Embedding-0.6B` |
| LLM | 30000 | `Qwen/Qwen3-32B` |
