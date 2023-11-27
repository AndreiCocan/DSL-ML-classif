#! /bin/bash

nb_programs=$1
compile=$2
clean=$3

pathFuzzer=$(pwd)
mkdir -p compiled_programs

if [ $clean = "clean" -o $clean = "c" ]
then
    rm ./generated_programs/*.neoml
    rm ./compiled_programs/*.py
    rm ./compiled_programs/*r
    wait
fi

for i in $(seq 1 $nb_programs); do
    python3 fuzzer.py
done

if [ $compile = "compile" -o $clean = "c" ]
then
    cd ../NeoML/
    npm run build
    for file in $pathFuzzer/generated_programs/*.neoml
    do
        ./bin/cli.js generate -d $pathFuzzer/compiled_programs -l Python $file
        ./bin/cli.js generate -d $pathFuzzer/compiled_programs -l R $file
    done
fi