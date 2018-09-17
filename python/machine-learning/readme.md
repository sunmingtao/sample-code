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


