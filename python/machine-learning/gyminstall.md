
    sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb ffmpeg xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
    
    pip install gym
    pip install gym[all]
    
## Troubleshooting
    
#### /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.20' not found (required by layout)

    conda install libgcc

https://askubuntu.com/questions/575505/glibcxx-3-4-20-not-found-how-to-fix-this-error
