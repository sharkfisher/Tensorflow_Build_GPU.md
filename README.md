# Tensorflow

### Motivation
I have CUDA 9.1 installed for macOS High Sierra (10.13.1), and installed tensorflow-gpu through pip. However, when I tried to run the object detection sample `python object_detection/builders/model_builder_test.py`, I got an ImportError
```
Library not loaded: @rpath/libcublas.8.0.dylib
```
It was looking for CUDA 8.0 while I have CUDA 9.1. Instead of installing CUDA 8.0, I decided to try compiling tensorflow from scratch with GPU support for CUDA 9.1.

### System information
* OS - High Sierra 10.13.1
* Tensorflow - 1.6.0rc1 (git master as of 2018/02/20)
* Xcode command line tools - 8.3.3 (9.0 does not work)
* Cmake - 3.10.2 (Homebrew)
* Bazel - 0.10.1 (Homebrew)
* CUDA - 9.1 (NVIDIA)
* cuDNN - 7.0 (NVIDIA)

### Requirements
* I use python3 in virtualenv, which is activated before the following `pip install`
* `pip install six numpy wheel`
* `brew install coreutils gcc`

## Step-by-step guide
 ### Switch to Xcode 8.3.3
 * `cd ~/Downloads`
 * `unzip Xcode8.3.3.xip`
 * `sudo mv Xcode.app /Applications/Xcode-8.3.3.app`
 * `sudo xcode-select -s /Applications/Xcode-8.3.3.app`
 * `/usr/bin/gcc --version`
 ```
Configured with: --prefix=/Applications/Xcode-8.3.3.app/Contents/Developer/usr --with-gxx-include-dir=/usr/include/c++/4.2.1
Apple LLVM version 8.1.0 (clang-802.0.42)
Target: x86_64-apple-darwin17.2.0
Thread model: posix
InstalledDir: /Applications/Xcode-8.3.3.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin
```
 ### Link gomp library of gcc by Homebrew
  * `sudo ln -s /usr/local/Cellar/gcc/7.3.0/lib/gcc/7/libgomp.dylib /usr/local/lib/libgomp.dylib`
  
 ### Remove ALL occurrences __align__(sizeof(T)) from following files:
 * tensorflow/core/kernels/depthwise_conv_op_gpu.cu.cc (4 places)
 * tensorflow/core/kernels/split_lib_gpu.cu.cc
 * tensorflow/core/kernels/concat_lib_gpu.impl.cu.cc
 
     For example, `extern __shared__ __align__(sizeof(T)) unsigned char smem[];` => `extern __shared__ /*__align__(sizeof(T))*/ unsigned char smem[];`
     
 ### Steps:
 * No need to disable SIP
 * `./configure`   (Find CUDA compute value from https://developer.nvidia.com/cuda-gpus)
 ```
 You have bazel 0.10.1-homebrew installed.
 Please specify the location of python. [Default is ...]: /usr/local/Cellar/python3/3.6.4_2/bin/python3


 Found possible Python library paths:
   /usr/local/Cellar/python3/3.6.4_2/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages
 Please input the desired Python library path to use.  Default is [...] /usr/local/Cellar/python3/3.6.4_2/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages
 Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: 
 Google Cloud Platform support will be enabled for TensorFlow.

 Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: 
 Hadoop File System support will be enabled for TensorFlow.

 Do you wish to build TensorFlow with Amazon S3 File System support? [Y/n]: 
 Amazon S3 File System support will be enabled for TensorFlow.

 Do you wish to build TensorFlow with Apache Kafka Platform support? [y/N]: y
 Apache Kafka Platform support will be enabled for TensorFlow.

 Do you wish to build TensorFlow with XLA JIT support? [y/N]: 
 No XLA JIT support will be enabled for TensorFlow.

 Do you wish to build TensorFlow with GDR support? [y/N]: 
 No GDR support will be enabled for TensorFlow.

 Do you wish to build TensorFlow with VERBS support? [y/N]: 
 No VERBS support will be enabled for TensorFlow.

 Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: 
 No OpenCL SYCL support will be enabled for TensorFlow.

 Do you wish to build TensorFlow with CUDA support? [y/N]: y
 CUDA support will be enabled for TensorFlow.

 Please specify the CUDA SDK version you want to use, e.g. 7.0. [Leave empty to default to CUDA 9.0]: 9.1


 Please specify the location where CUDA 9.1 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: 


 Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7.0]: 


 Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:


 Please specify a list of comma-separated Cuda compute capabilities you want to build with.
 You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
 Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 3.5,5.2]


 Do you want to use clang as CUDA compiler? [y/N]: 
 nvcc will be used as CUDA compiler.

 Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: 


 Do you wish to build TensorFlow with MPI support? [y/N]: 
 No MPI support will be enabled for TensorFlow.

 Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: 


 Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: 
 Not configuring the WORKSPACE for Android builds.

 Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See tools/bazel.rc for more details.
	 --config=mkl         	# Build with MKL support.
	 --config=monolithic  	# Config for mostly static monolithic build.
 Configuration finished
 ```
 * Add following paths:
   ```
   export CUDA_HOME=/usr/local/cuda
   export DYLD_LIBRARY_PATH=/Users/__USERNAME__/lib:/usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib (Replace USERNAME with your machine username)
   export LD_LIBRARY_PATH=$DYLD_LIBRARY_PATH
   export PATH=$DYLD_LIBRARY_PATH:$PATH
   ```
 * Start build
     ```
     bazel build --config=cuda --config=opt --action_env PATH --action_env LD_LIBRARY_PATH --action_env DYLD_LIBRARY_PATH //tensorflow/tools/pip_package:build_pip_package
     ```
 * Generate a wheel for installation 
     ```
     bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
     ```
 * Install tensorflow wheel 
     ```
     sudo pip install /tmp/tensorflow_pkg/tensorflow-1.6.0rc1-cp36-cp36m-macosx_10_12_x86_64.whl (File name depends on tensorflow version and python version)
     ```
