# conda activate extraction_venv
# mkdir -p build/eigen
# wget https://github.com/clab/dynet/releases/download/2.1/eigen-b2e267dc99d4.zip
# unzip eigen-b2e267dc99d4.zip -d build/eigen

export EIGEN3_INCLUDE_DIR="$PWD/build/eigen"
export CPLUS_INCLUDE_PATH="$EIGEN3_INCLUDE_DIR:$CPLUS_INCLUDE_PATH"

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"

export CFLAGS="-std=c++14"
export CXXFLAGS="-std=c++14"

python setup.py develop