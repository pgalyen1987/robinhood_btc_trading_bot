#!/bin/bash
set -e

# Remove any existing installations
sudo rm -rf /usr/local/lib/libta_lib*
sudo rm -rf /usr/local/include/ta-lib
sudo pip3 uninstall -y TA-Lib numpy

# Install build dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    wget \
    automake \
    libtool

# Check Python and pip versions
python3 --version
pip3 --version

# Install numpy and check its version
pip3 install numpy
NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__.split('.')[0])")

# Set TA-Lib version based on numpy version
if [ "$NUMPY_VERSION" -ge "2" ]; then
    TALIB_VERSION="0.5.2"  # Latest version for numpy>=2
    TALIB_C_VERSION="0.4.28"  # Latest C library version
else
    TALIB_VERSION="0.4.34"  # Latest version for numpy<2
    TALIB_C_VERSION="0.4.28"  # Latest C library version
fi

echo "Installing TA-Lib C library version ${TALIB_C_VERSION}"
echo "Will install Python TA-Lib version ${TALIB_VERSION}"

# Download and install TA-Lib C library
wget https://sourceforge.net/projects/ta-lib/files/ta-lib/${TALIB_C_VERSION}/ta-lib-${TALIB_C_VERSION}.tar.gz
tar -xzf ta-lib-${TALIB_C_VERSION}.tar.gz
cd ta-lib/

# Copy newer config.guess for better platform support
cp /usr/share/automake-1.16/config.guess config.guess

# Configure and build TA-Lib
./configure --prefix=/usr
make
sudo make install

# Update library cache
sudo ldconfig

# Clean up
cd ..
rm -rf ta-lib-${TALIB_C_VERSION}.tar.gz ta-lib/

# Set environment variables
export TA_INCLUDE_PATH="/usr/include"
export TA_LIBRARY_PATH="/usr/lib"

# Install specific TA-Lib version
pip3 install TA-Lib==${TALIB_VERSION}

# Verify installation
python3 -c "import talib; print('TA-Lib version:', talib.__version__)"