#!/bin/bash
/usr/bin/v4l2-ctl -d /dev/video0  --all | grep "Pixel Format" | awk '{print $4}' > /srv/www/board
/usr/bin/v4l2-ctl -d /dev/video0 --set-ctrl=brightness=137 --set-ctrl=contrast=154 --set-ctrl=gain_red=122 --set-ctrl=gain_green=102 --set-ctrl=gain_blue=138
gpioset gpiochip0 18=1

gpioset gpiochip0 6=1
gpioset gpiochip0 7=1
gpioset gpiochip0 5=1
gpioset gpiochip0 5=0
sleep 1
gpioset gpiochip0 5=1

/opt/microchip/cam_imx334_init
sleep 1

if [ -f ~/VectorBlox-SDK-release-v2.0/example/soc-video-c/Makefile ]; then
	(cd ~/VectorBlox-SDK-release-v2.0/example/soc-video-c && make overlay)
	sleep 1
	
	/opt/microchip/auto_gain
	sleep 1
fi
