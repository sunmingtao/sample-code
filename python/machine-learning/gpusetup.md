ssh instance

### Install Cuda tool

    curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb

    # update to cuda 9.0
    sudo apt-get update
    sudo apt-get install cuda-9-0
    sudo nvidia-smi -pm 1

    # add to environment variable
    echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
    echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64' >> ~/.bashrc
    source ~/.bashrc
    nvidia-smi

### Install cuDNN

Download cudnn-9.0-linux-x64-v7.2.1.38.tgz from https://developer.nvidia.com/rdp/cudnn-download

    gcloud compute scp /Users/msun/Downloads/cudnn-9.0-linux-x64-v7.2.1.38.tgz instance-3:/tmp

    cd /tmp
    tar xzvf cudnn-9.0-linux-x64-v7.2.1.38.tgz
    sudo cp cuda/lib64/* /usr/local/cuda/lib64/
    sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
    rm -rf /tmp/cuda
    rm cudnn-9.0-linux-x64-v7.2.1.38.tgz

Assuming Anaconda has been installed

### Install Tensorflow GPU

    pip install --upgrade tensorflow-gpu
   
Check if install is successful

    import tensorflow as tf
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

### Install other python libraries

    pip install keras
    pip install gym

Reference:

https://medium.com/@kstseng/%E5%9C%A8-google-cloud-platform-%E4%B8%8A%E4%BD%BF%E7%94%A8-gpu-%E5%92%8C%E5%AE%89%E8%A3%9D%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E7%9B%B8%E9%97%9C%E5%A5%97%E4%BB%B6-1b118e291015
https://medium.com/@jayden.chua/quick-install-cuda-on-google-cloud-compute-6c85447f86a1
