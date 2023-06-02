#!/bin/bash

echo "Starting ray up..."
ray up aws-ray-cluster.yml --yes

echo "Starting ray rsync-up..."
ray rsync-up aws-ray-cluster.yml graph.graphml graph.graphml

echo "Starting ray submit..."
ray submit aws-ray-cluster.yml centralities.py

echo "All AWS tasks finished."

echo "downloading AWS results and logs"
ray rsync-down aws-ray-cluster.yml /home/ubuntu/centralities.log aws-downloads/centralities.log
ray rsync-down aws-ray-cluster.yml /home/ubuntu/centralities.pickle aws-downloads/centralities.pickle