#!/bin/bash
pip uninstall -y dataproc_jupyter_plugin
pip install -r requirements.txt
git config --global credential.helper store
huggingface-cli login