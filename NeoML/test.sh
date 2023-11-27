#! /bin/bash

#npm run build
#./bin/cli.js generate -d tests/res/ -l Python tests/test0.neoml
#./bin/cli.js generate -d tests/res/ -l Python tests/test1.neoml


npm run build

for f in tests/*.neoml
do
    ./bin/cli.js generate -d tests/res-python/ -l Python "$f"
    ./bin/cli.js generate -d tests/res-r/ -l R "$f"

done
