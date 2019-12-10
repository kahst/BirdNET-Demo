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
sudo pip3 install RPi.GPIO
```

Start playback script after startup by adding this line to <i>/etc/rc.local</i> (before exit 0):

```
cd /home/pi/BirdNET-Demo && python3 pi/playback.py &
```

Change the path to <i>BirdNET-Demo</i> accordingly if you used a different location. After reboot, the script will start automatically and play sounds when a button is pressed.

The script uses the following GPIO pins to play sounds:

```

Pin 11 GPIO.17: Common Starling (Sturnus vulgaris)
Pin 12 GPIO.18: Common House Martin (Delichon urbicum)
Pin 13 GPIO.27: Common Linnet (Linaria cannabina)
Pin 15 GPIO.22: European Pied Flycatcher (Ficedula hypoleuca)

Pin 18 GPIO.23: Random species (Goldcrest, Yellowhammer, Eurasian Blue Tit, Common Chiffchaff, European Goldfinch)

```