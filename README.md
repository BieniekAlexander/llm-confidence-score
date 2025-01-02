# llm-confidence-score
A confidence score for LLM answers. Forked from [mbaak/llm-confidence-score/tree/e8f733d](https://github.com/mbaak/llm-confidence-score/tree/e8f733d1d5498ab5140f21ae3e75cbc71f5d9cf6), provided as a resource for [this article](https://medium.com/wbaa/a-confidence-score-for-llm-answers-c668844d52c8).

## Prerequisites
- Python 3.10.15
- Resource Requirements
    - Using [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) -> [a2-highgpu-1g](https://cloud.google.com/compute/docs/gpus)
    - Using [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) -> untested, a2-highgpu-1g doesn't have enough memory
    pyarrow
- [Hugging Face](https://huggingface.co/):
    - account with access token ([note](https://github.com/huggingface/diffusers/issues/6223#issuecomment-2141411382))
    - access to [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
    - access to [deepset/roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2)

## Setup
```bash
# create virtual environment and install requirements
python -m venv venv
source venv/bin/activate
pip uninstall -y dataproc_jupyter_plugin # this package caused issues on VAI workbenches, though I don't recall what exactly
pip install -r requirements.txt

# setup Hugging Face access
git config --global credential.helper store
huggingface-cli login
```

## Run experiment
```bash
# note: work provided in a python script, due to import errors with `pyarrow`
python script.py
```

> Note: I couldn't get the notebook to run in an ipykernel in a Vertex AI Workbench, so I reproduced the notebook as a script