# How to use Jupyter notebook on Google cloud

Computer Engine -> Create VM instance

us-east1 - us-east1-b

4 CPU

Click customize

GPU 1

![alt text](https://github.com/sunmingtao/sample-code/blob/master/python/machine-learning/images/vm-instance.png)

Change hard disk -> Ubuntu 16.04 LTS, size 20 GB

![alt text](https://github.com/sunmingtao/sample-code/blob/master/python/machine-learning/images/operating-system.png)

Allow HTTP\
Allow HTTPS

VPC Network -> External IP address, change type to 'static'

![alt text](https://github.com/sunmingtao/sample-code/blob/master/python/machine-learning/images/external-ip.png)

VPC Network -> firewall rule -> create new rule

Source IP: 0.0.0.0/0
Protocal and port: tcp:5000

![alt text](https://github.com/sunmingtao/sample-code/blob/master/python/machine-learning/images/firewall.png)

Click SSH link, open SSH console

    wget https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh
    bash Anaconda3-4.2.0-Linux-x86_64.sh
    export PATH=/home/<user_directory>/anaconda3/bin:$PATH
    jupyter notebook --generate-config
    vi .jupyter/jupyter_notebook_config.py
    
Contents of jupyter_notebook_config.py

    c = get_config()
    # Kernel config
    c.IPKernelApp.pylab = 'inline'  # if you want plotting support always in your notebook
    c.NotebookApp.ip = '*'
    c.NotebookApp.open_browser = False  #so that the ipython notebook does not opens up a browser by default
    c.NotebookApp.port = 5000
    
Back to SSH console

    jupyter notebook
    
In your browser 

    http://<external-ip-address>:5000
