# How to use Jupyter notebook on Google cloud

Computer Engine -> Create VM instance

us-east1 - us-east1-b

4 CPU

Click customize

GPU 1

Change hard disk -> Ubuntu 16.04 LTS, size 20 GB

Allow HTTP\
Allow HTTPS

VPC Network -> External IP address, change type to 'static'

VPC Network -> firewall rule -> create new rule

Source IP: 0.0.0.0/0
Protocal and port: tcp:5000

Click SSH link, open SSH console


