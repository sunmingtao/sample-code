
    sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb ffmpeg xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
    
    pip install gym
    pip install gym[all]
    
## Troubleshooting
    
#### /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.20' not found (required by layout)

    conda install libgcc

Find where libstdc++ is

    /sbin/ldconfig -p | grep stdc++
    
Find versions of GLIBCXX

    strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX


https://askubuntu.com/questions/575505/glibcxx-3-4-20-not-found-how-to-fix-this-error

https://zcwlwen.online/2017/02/05/%E8%A7%A3%E5%86%B3%E5%9C%A8Ubuntu%E4%B8%8A%E5%87%BA%E7%8E%B0GLIBCXX-3-4-not-found%E9%97%AE%E9%A2%98/
