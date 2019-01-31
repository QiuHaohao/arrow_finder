#Update and upgrade any existing packages:
sudo apt-get update -y && sudo apt-get upgrade -y

# Install developer tools (Installed earlier)
sudo apt-get install -y build-essential pkg-config

# Install CMake ( for installing OpenCV source later)
sudo apt-get install -y cmake

# Image I/O packages:
sudo apt-get install -y libgtk2.0-dev libgtk-3-dev
# Video I/O packages:
sudo apt-get install -y libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
# Display I/O packages: for highgui module in OpenCV
sudo apt-get install -y libxvidcore-dev libx264-dev
# Linear algebra library and fortran compiler
sudo apt-get install -y libatlas-base-dev gfortran

# Here we download OpenCV 3.3.0
pushd ~
wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.3.0.zip
unzip opencv.zip
wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.3.0.zip
unzip opencv_contrib.zip

# Before compile the OpenCV source codes, you need to install pip and numpy first
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
sudo python3 get-pip.py #for python 3.x user
pip install numpy

# Compile and Install OpenCV
cd ~/opencv-3.3.0/
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
		-D CMAKE_INSTALL_PREFIX=/usr/local \
			-D INSTALL_PYTHON_EXAMPLES=ON \
				-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.3.0/modules \
					-D BUILD_EXAMPLES=ON ..

# Configure your swap space size before compiling
sudo nano /etc/dphys-swapfile

# Edit CONF_SWAPSIZE=1024
sudo /etc/init.d/dphys-swapfile stop
# CTRL+x to save and exit
sudo /etc/init.d/dphys-swapfile start
# Activate the change and compile OpenCV by 4 core (it will take nearly an hour)
make -j4

# Install OpenCV
sudo make install
sudo ldconfig
# Reconfigure your swap space size back to 100
sudo nano /etc/dphys-swapfile
# Edit CONF_SWAPSIZE=1024
sudo /etc/init.d/dphys-swapfile stop
sudo /etc/init.d/dphys-swapfile start

popd
