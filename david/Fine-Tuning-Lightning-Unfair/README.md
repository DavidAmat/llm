#  Build Your Custom AI/LLM With PyTorch Lightning

Sources: 
- [Medium post](https://medium.com/@jz77/build-your-custom-ai-llm-with-pytorch-lightning-4eeb943dd88a)
- [GitHub repo](https://github.com/zjohn77/lightning-mlflow-hf/tree/main)
- repo: `lightning-mlflow-hf`

## Creating environment

```bash
conda create --name r_unfair python=3.10.14
conda activate r_unfair


```
To install pytorch:
- Go to [torch](https://download.pytorch.org/whl/torch/):
```bash
# Select
# torch-2.5.1+cu121-cp310-cp310-linux_x86_64.whl
# https://download.pytorch.org/whl/cu121_full/torch-2.5.1%2Bcu121-cp310-cp310-linux_x86_64.whl#sha256=2cb5923cc771377d3d7591eeaf9e98c901145542564ccd0f24114cbdcb9aed59
pip install https://download.pytorch.org/whl/cu121_full/torch-2.5.1%2Bcu121-cp310-cp310-linux_x86_64.whl#sha256=2cb5923cc771377d3d7591eeaf9e98c901145542564ccd0f24114cbdcb9aed59
```

Install the rest:
```bash
pip install lightning torchmetrics transformers peft 
pip install datasets mlflow onnx polars sentencepiece

# Check any broken conflict
pip check
```

Install kernel

```bash
pip install jupyterlab
pip install ipykernel 
python -m ipykernel install --user --name "kr_unfair"
```

```bash

```

```bash

```

```bash

```

```bash

```
