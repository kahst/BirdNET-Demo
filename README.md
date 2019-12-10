# BirdNET-Demo
Source code for BMBF InnoTruck demo of BirdNET.

<b>This repo is currently under development.</b>

## Setup Raspberry Pi

Clone the repository:

```
git clone https://github.com/kahst/BirdNET-Demo.git
```

Install dependencies (you'll need to install Python3 and pip3 if not already provided with the OS image):

```
sudo pip3 install -r requirements.txt
```

Start playback script after startup by adding this line to <i>/etc/rc.local</i> (before exit 0):

```
cd /home/pi/BirdNET-Demo && python3 pi/playback.py &
```

Change the path to <i>BirdNET-Demo</i> accordingly if you used a different location.