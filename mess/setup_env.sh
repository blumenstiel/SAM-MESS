# run script with
# bash mess/setup_env.sh

# Create new environment "sam"
conda create --name sam -y python=3.8
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sam

# Install SAM requirements
conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
pip install transformers[pytorch]
pip install opencv-python
pip install scipy

# Install packages for dataset preparation
pip install gdown
pip install kaggle
pip install rasterio
pip install pandas