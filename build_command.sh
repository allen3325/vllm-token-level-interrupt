# https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source
# Setup python environment
uv venv --python 3.12 --seed
source .venv/bin/activate

# Set up using Python-only build (without compilation)
# If you only need to change Python code, you can build and install vLLM without compilation. Using uv pip's --editable flag, changes you make to the code will be reflected when you run vLLM:

git clone https://github.com/vllm-project/vllm.git
cd vllm
# SETUPTOOLS_SCM_PRETEND_VERSION is you need to change
SETUPTOOLS_SCM_PRETEND_VERSION=0.12.0+tokenlevel VLLM_USE_PRECOMPILED=1 uv pip install --editable .