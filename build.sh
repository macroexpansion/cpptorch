#!/bin/sh

if [ -d "build" ] 
then
    cd build
else
    mkdir build && cd build
fi

cmake -DCMAKE_PREFIX_PATH="$PWD/../libtorch;$PWD/../opencv/build" \
      -DPYTHON_EXECUTABLE=$(which python) \
      ..
cmake --build . --config Release
