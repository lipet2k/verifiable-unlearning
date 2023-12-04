#!/bin/bash

run()
{
    docker run -p 5000:5000 verifiable-unlearning
}

build()
{
    docker build -t verifiable-unlearning .
}

shell() 
{
    docker run -it verifiable-unlearning
}

evaluation() 
{
    docker run -it verifiable-unlearning bash /root/verifiable-unlearning/evaluation/trials.sh
}

print_usage() 
{
    echo "Choose: docker.sh {build|shell|eval|run}"
    echo "    build - Build the container"
    echo "    shell - Spawn a shell inside the container"
    echo "    eval  - Run evaluation in the container"
    echo "    run   - Run the server"
}

if [[ $1 == "" ]]; then
    echo "No argument provided"
    print_usage
elif [[ $1 == "build" ]]; then
    build
elif [[ $1 == "shell" ]]; then
    shell 
elif [[ $1 == "eval" ]]; then
    evaluation
elif [[ $1 == "run" ]]; then
    run
else 
    echo "Argument not recognized!"
    print_usage
fi