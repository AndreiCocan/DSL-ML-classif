#! /bin/bash

PURPLE='\033[0;35m'
NC='\033[0m' # No color

nb_programs=$1
clean=$2
compile=$3
run=$4

pathFuzzer=$(pwd)
mkdir -p generated_programs
mkdir -p compiled_programs
mkdir -p run_results

if [ $clean = "clean" ]
then
    rm -f ./generated_programs/*.neoml
    rm -f ./compiled_programs/*.py
    rm -f ./compiled_programs/*.r
    rm -f ./run_results/*.txt
    wait
fi

for i in $(seq 1 $nb_programs); do
    python3 fuzzer.py
done

if [ $compile = "compile" ]
then
    cd ../NeoML/
    for file in $pathFuzzer/generated_programs/*.neoml
    do
        ./bin/cli.js generate -d $pathFuzzer/compiled_programs -l Python $file
        ./bin/cli.js generate -d $pathFuzzer/compiled_programs -l R $file
    done
fi

cd $pathFuzzer

if [ $run = "run" ]
then
    for file in $pathFuzzer/compiled_programs/*.py
    do
        python3 $file #&> ./run_results/$(basename "$file").txt
        echo -e "${PURPLE}Runned $file${NC}"
    done
    for file in $pathFuzzer/compiled_programs/*.r
    do
        Rscript $file #&> ./run_results/$(basename "$file").txt
        echo -e "${PURPLE}Runned $file${NC}"
    done
fi