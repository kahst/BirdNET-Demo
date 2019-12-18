#!/bin/bash
cd /home/birdnet/BirdNET-Demo/
echo 'STARTING BirdNET ANALYZER...'
python3 analyzer.py &> /tmp/birdnet_analyzer.log &
echo 'STARTING BirdNET SERVER...'
python server.py &> /tmp/birdnet_server.log &
echo 'WAITING FOR 5 SECONDS...'
sleep 5
echo 'STARTING CHROMIUM BROWSER...'
chromium-browser --kiosk --autoplay-policy=no-user-gesture-required --incognito --password-store=basic --app=http://localhost:8080
echo 'DONE! STARTUP COMPLETE!'