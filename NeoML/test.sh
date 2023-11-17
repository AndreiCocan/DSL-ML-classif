#! /bin/bash

npm run build
./bin/cli.js generate -d tests/res/ tests/test0.neoml
./bin/cli.js generate -d tests/res/ tests/test1.neoml