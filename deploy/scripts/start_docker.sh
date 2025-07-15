#!/bin/bash
# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to ECR..."
aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin 908052928183.dkr.ecr.eu-north-1.amazonaws.com

echo "Pulling Docker image..."
docker pull 908052928183.dkr.ecr.eu-north-1.amazonaws.com/jatinguptauber:latest

echo "Checking for existing container..."
if [ "$(docker ps -q -f name=demand-prediction)" ]; then
    echo "Stopping existing container..."
    docker stop demand-prediction
fi

if [ "$(docker ps -aq -f name=demand-prediction)" ]; then
    echo "Removing existing container..."
    docker rm demand-prediction
fi

echo "Starting new container..."
docker run --name demand-prediction -d -p 80:8000 -e DAGSHUB_USER_TOKEN=${DAGSHUB_USER_TOKEN} 908052928183.dkr.ecr.eu-north-1.amazonaws.com/jatinguptauber:latest 

echo "Container started successfully."