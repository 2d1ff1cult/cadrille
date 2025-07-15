@echo off
:: TODO: SET LONG PATH USING REGISTRY EDITOR
:: TODO: git config --system core.longpaths true
:: MAKE SURE TO DOWNLOAD PYTHON 3.10 AND INSTALL AS FOLLOWS:
:: do NOT install for all users
:: add to path
:: done and restart all terminals

dir
pause
:: useful when training
SET PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

:: ------------------------- INITIAL SETUP -------------------------
:: for later use; packing the environment for cross-user repeatability
py -3.10 -m pip install venv-pack wheel
git config --system core.longpaths true
git lfs install

:: define a new virtual environment on py3.10
py -3.10 -m venv 2d3dgen
:: activate virtual environment
call .\2d3dgen\Scripts\activate

:: upgrade pip
pip install --upgrade pip

:: uninstall and reinstall wheel and setuptools
py -3.10 -m pip uninstall wheel setuptools -y
py -3.10 -m pip install --upgrade wheel setuptools

py -3.10 -m pip install ninja
py -3.10 -m pip install trimesh

:: HuggingFace lib, line below will be used to fetch training data, if required
py -3.10 -m pip install datasets

:: ------------------------- BUILDING TORCH -------------------------
py -3.10 -m pip uninstall torch torchvision torchaudio -y
:: # py -3.10 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
:: # py -3.10 -m pip install torch==2.5.1+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
:: # py -3.10 -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio --index-url https://download.pytorch.org/whl/cu121
py -3.10 -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129

py -3.10 -m pip install ninja pybind11>=2.12 cmake
pause

git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
py -3.10 setup.py install
pause
cd ../
dir
pause

py -3.10 -m pip install qwen-vl-utils==0.0.10

:: ------------------------- SETTING UP PYTORCH3D -------------------------
:: py -3.10 -m pip install git+https://github.com/facebookresearch/pytorch3d.git@stable
:: MAYBE REMOVE EITHER ABOVE OR BELOW LINE
git clone --recursive https://github.com/facebookresearch/pytorch3d.git
pause
cd pytorch3d
dir
pause

python setup.py install
:: cd ../
dir
pause

:: for debugging; comment later
@echo on
call
@echo off
:: echo Pausing to validate env start; should see (2d3dgen) prefixed to shell

:: ------------------------- SETTING UP CADQUERY -------------------------
:: setting up cadquery
py -3.10 -m pip install cadquery==2.5.2
py -3.10 -m pip install CQ-editor

:: replicate the pip installs from the Dockerfile from cadrille github:
py -3.10 -m pip install accelerate==0.34.2 cadquery-ocp==7.7.2 casadi==3.6.7 einops==0.8.0 transformers==4.50.3 manifold3d==3.0.0 trimesh==4.5.3 contourpy==1.3.1 scipy==1.14.1 imageio==2.36.1 scikit-image==0.25.0 ipykernel==6.29.5 ipywidgets==8.1.5 cadquery==2.5.2
py -3.10 -m pip install open3d==0.19.0 

:: clone repos and install reqs
git clone https://github.com/NVlabs/PartPacker.git
cd PartPacker
py -3.10 -m pip install -r requirements.txt
py -3.10 -m pip install -r requirements.lock.txt
py -3.10 -m pip install meshiki

py -3.10 -m pip install -r requirements.txt
py -3.10 -m pip install -r requirements.lock.txt

:: ------------------------- DOWNLOADING PRETRAINED MODELS FOR PARTPACKER -------------------------
:: windows CMD
mkdir pretrained
cd pretrained
curl -L "https://huggingface.co/nvidia/PartPacker/resolve/main/vae.pt" --output "vae.pt"
curl -L "https://huggingface.co/nvidia/PartPacker/resolve/main/flow.pt" --output "flow.pt"
:: linux
:: wget https://huggingface.co/nvidia/PartPacker/resolve/main/vae.pt
:: wget https://huggingface.co/nvidia/PartPacker/resolve/main/flow.pt


echo Changing back to directory of this script
cd ../../
dir
pause

:: ------------------------- SETTING UP CADRILLE FROM SOURCE -------------------------
git clone https://github.com/2d1ff1cult/cadrille.git

cd cadrille

py -3.10 imports.py

cd data
:: git clone https://huggingface.co/datasets/filapro/cad-recode-v1.5
:: git clone https://huggingface.co/datasets/maksimko123/text2cad
:: git clone https://huggingface.co/datasets/maksimko123/fusion360_test_mesh
:: git clone https://huggingface.co/datasets/maksimko123/deepcad_test_mesh

cd ../
:: create train/val pickles
py -3.10 -u prepare_data.py

cd ../
dir
pause

:: ------------------------- COPYING TRAINING DATASET ------------------------- 
copy cadrille\data\text2cad\text2cad.zip .
tar -xf text2cad.zip -C cadrille\data\text2cad
pause

:: ------------------------- CLEAN UP -------------------------
:: clear install cache and remove all wheels
pip cache purge

:: provide a pause so user can read outputs
pause