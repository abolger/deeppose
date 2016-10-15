# DeepPose

NOTE: This is not official implementation. Original paper is [DeepPose: Human Pose Estimation via Deep Neural Networks](http://arxiv.org/abs/1312.4659).

# Requirements
- [CUDA](http://www.nvidia.com/object/cuda_home_new.html)
- Python 3.5.1+ or Python 2.7.+
  - [Chainer 1.13.0+](https://github.com/pfnet/chainer)
  - numpy 1.9+
  - scikit-image 0.11.3+
  - OpenCV 3.1.0+


I strongly recommend to use Anaconda environment. 

## Installation of dependencies for Linux: 
```
pip install chainer
pip install numpy
pip install scikit-image
# for python3
conda install -c https://conda.binstar.org/menpo opencv3
# for python2 (gets you opencv 3.1 requirement)
conda install -c menpo opencv3
```

## Installation of dependencies for Max OS X (El Capitan) and Python 2.7: 
For MAC: use [homebrew](http://brew.sh) to install CUDA, as well as working python libraries that have CuPy set up for MACs:   
```
brew install python
brew install Caskroom/cask/cuda
```

Add the following to ~/.bash_profile to link your CUDA library.:
```
PATH="/Developer/NVIDIA/CUDA-8.0/bin/:$PATH"
export LD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-8.0/lib/
export CUDA_ROOT=/Developer/NVIDIA/CUDA-8.0/
export LDFLAGS="-F/Library/Frameworks/"
```
Close and re-open Terminal for .bash_profile changes to take effect.
Next, use your homebrew-ed python to get the right dependency versions:  
```

pip install chainer
pip install scikit-image
pip install scipy
pip install matplotlib

# for python3
brew install opencv3 --with-python3

# for python2 (gets you opencv 3.1 requirement)
brew install homebrew/science/opencv3 --with-contrib --with-cuda
echo /usr/local/opt/opencv3/lib/python2.7/site-packages >> /usr/local/lib/python2.7/site-packages/opencv3.pth
mkdir -p /Users/abolger/Library/Python/2.7/lib/python/site-packages
echo 'import site; site.addsitedir("/usr/local/lib/python2.7/site-packages")' >> /Users/abolger/Library/Python/2.7/lib/python/site-packages/homebrew.pth
```


# Dataset preparation
MAC only: use [homebrew](http://brew.sh) to install wget to run scripts.  
```
brew install wget
```

```
bash datasets/download.sh
python datasets/flic_dataset.py
python datasets/lsp_dataset.py
python datasets/mpii_dataset.py
```

- [FLIC-full dataset](http://vision.grasp.upenn.edu/cgi-bin/index.php?n=VideoLearning.FLIC)
- [LSP Extended dataset](http://www.comp.leeds.ac.uk/mat4saj/lspet_dataset.zip)
- **MPII dataset**
    - [Annotation](http://datasets.d2.mpi-inf.mpg.de/leonid14cvpr/mpii_human_pose_v1_u12_1.tar.gz)
    - [Images](http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz)

## MPII Dataset

- [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/#download)
- training images: 18079, test images: 6908
  - test images don't have any annotations
  - so we split trining imges into training/test joint set
  - each joint set has
- training joint set: 17928, test joint set: 1991

# Start training

Starting with the prepared shells is the easiest way. If you want to run `train.py` with your own settings, please check the options first by `python scripts/train.py --help` and modify one of the following shells to customize training settings.

## For FLIC Dataset

```
bash shells/train_flic.sh
```

## For LSP Dataset

```
bash shells/train_lsp.sh
```

## For MPII Dataset

```
bash shells/train_mpii.sh
```

### GPU memory requirement

- AlexNet
  - batchsize: 128 -> about 2870 MiB
  - batchsize: 64 -> about 1890 MiB
  - batchsize: 32 (default) -> 1374 MiB
- ResNet50
  - batchsize: 32 -> 6877 MiB

# Prediction

Will add some tools soon
