#!/usr/bin/env bash
set -euo pipefail

ROOT="${A800_ROOT:-/vePFS-0x0d/nzh}"
CONDA_PKGS_DIRS="${ROOT}/conda_pkgs"
LCM_ENV="${ROOT}/envs/lcm"
BSEM_ENV="${ROOT}/envs/bsem"
MINICONDA_DIR="${ROOT}/miniconda3"

mkdir -p "${ROOT}/envs" "${CONDA_PKGS_DIRS}"
export CONDA_PKGS_DIRS

if [[ -x /root/miniconda3/bin/conda ]]; then
  CONDA_BIN=/root/miniconda3/bin/conda
elif [[ -x "${MINICONDA_DIR}/bin/conda" ]]; then
  CONDA_BIN="${MINICONDA_DIR}/bin/conda"
else
  mkdir -p "${MINICONDA_DIR}"
  INSTALLER="${ROOT}/Miniconda3-latest-Linux-x86_64.sh"
  if command -v curl >/dev/null 2>&1; then
    curl -L -o "${INSTALLER}" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  else
    wget -O "${INSTALLER}" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  fi
  bash "${INSTALLER}" -b -p "${MINICONDA_DIR}"
  CONDA_BIN="${MINICONDA_DIR}/bin/conda"
fi

eval "$("${CONDA_BIN}" shell.bash hook)"

if [[ ! -d "${LCM_ENV}" ]]; then
  conda create -y -p "${LCM_ENV}" python=3.10 pip
fi
conda activate "${LCM_ENV}"
python -m pip install --upgrade pip

# Use official pip CUDA wheels for torch to avoid libtorch/oneAPI symbol issues
# that showed up with the conda install path on HuoShan A800.
python -m pip uninstall -y torch torchvision torchaudio || true
conda remove -y pytorch torchvision torchaudio pytorch-cuda || true

python -m pip install --no-cache-dir \
  torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install \
  pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
python -m pip install \
  torch-geometric==2.5.3 \
  numpy pandas scipy scikit-learn pyyaml openpyxl tqdm \
  einops timm omegaconf hydra-core networkx matplotlib seaborn \
  requests idna
conda deactivate

if [[ ! -d "${BSEM_ENV}" ]]; then
  conda create -y -p "${BSEM_ENV}" python=3.10 pip
fi
conda activate "${BSEM_ENV}"
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn pyyaml tqdm openpyxl
conda deactivate

echo "LCM env: ${LCM_ENV}"
echo "BSEM env: ${BSEM_ENV}"
echo "Conda package cache: ${CONDA_PKGS_DIRS}"
