The new dot marker sensor uses USB camera and there is no need of ROS and Raspberry Pi.

T0 set up the enviroment

Create a python 3.8 enviroment(suggest using anaconda):
```sh
conda create -n <name> python=3.8
```
Upgrad pip:
```sh
pip intall -U pip
```
Install packages:
```sh
pip install numpy scipy
pip install -U scikit-learn scikit-image
pip install opencv-python opencv-contrib-python
pip install matplotlib
```
run code to do nHHD:
```sh
python dot_nHHD.py
