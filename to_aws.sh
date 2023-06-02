#!/bin/bash

echo "Starting ray up..."
ray up aws-ray-cluster.yml --yes

echo "Starting ray rsync-up..."
ray rsync-up aws-ray-cluster.yml data/tirosinemia.csv tirosinemia.csv 
ray rsync-up aws-ray-cluster.yml data/tirosinemia_italia.csv tirosinemia_italia.csv


ray rsync-up aws-ray-cluster.yml requirements.txt requirements.txt
ray rsync-up aws-ray-cluster.yml update_base_environment.sh update_base_environment.sh


echo "Starting ray submit..."
ray submit aws-ray-cluster.yml to_ray_cluster.py

echo "All AWS tasks finished."

echo "downloading AWS results and logs"
ray rsync-down aws-ray-cluster.yml /home/ubuntu/1a_hello.pickle aws-downloads/1a_hello_aws.pickle
