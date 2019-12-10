# BirdNET-Demo
Source code for BMBF InnoTruck demo of BirdNET.

## This repo is currently under development.

## Setup Raspberry Pi

Clone the repository with and change the directory:

´´´
git clone https://github.com/kahst/BirdNET-Demo.git
cd BirdNET-Demo
´´´

Install dependencies (you'll need to install Python3 and pip3 if not already provided with the OS image):

´´´
sudo pip3 install -r requirements.txt
´´´

Start playback script after startup by adding this line to /etc/rc.local:

´´´
sudo nano /etc/rc.local
/usr/bin/python3 /home/pi/BirdNET-Demo/pi/playback.py &
´´´