#! /bin/bash

npm run build
./bin/cli.js generate -d tests/res/ -l Python tests/test0.neoml
./bin/cli.js generate -d tests/res/ -l Python tests/test1.neoml