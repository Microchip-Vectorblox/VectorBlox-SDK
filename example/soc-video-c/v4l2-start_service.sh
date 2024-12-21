#!/bin/bash
v4l2-ctl --device /dev/video0 --set-fmt-video=width=1920,height=1072,pixelformat=H264
/usr/bin/v4l2-ctl -d /dev/video0  --all | grep "Pixel Format" | awk '{print $4}' > /srv/www/board
/usr/bin/v4l2-ctl -d /dev/video0 --set-ctrl=brightness=137 --set-ctrl=contrast=154 --set-ctrl=gain_red=122 --set-ctrl=gain_green=102 --set-ctrl=gain_blue=138
gpioset gpiochip0 18=1

KER=`uname -r`
PATH_FILE=$(echo "/lib/modules/$KER/kernel/drivers/media/i2c/imx334.ko")
sudo mv $PATH_FILE ~/imx334.ko 

cd cam_init_src && make -B 
sudo mv cam_imx334_init /opt/microchip/

/opt/microchip/camera_init & 

sleep 1
