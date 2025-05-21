if [ ! -x cam_imx334_init ] && [ -f cam_imx334_init ]; then
	echo "Setting permission for Camera Init"
	chmod +x cam_imx334_init
	./cam_imx334_init
	sudo cp cam_imx334_init /opt/microchip/
fi
	
if [ ! -x auto_gain ] && [ -f auto_gain ]; then	
	echo "Setting Auto Gain feature for Camera"
	chmod +x auto_gain
	
fi


if [ -f v4l2-start_service.sh ]; then
    sed -i "s|~/VectorBlox-SDK-release-v2.0/example/soc-video-c|$(pwd)|" v4l2-start_service.sh
	echo "Setting up camera boot on startup"
	sudo cp v4l2-start_service.sh /opt/microchip/multimedia/v4l2/
	sudo cp auto_gain /opt/microchip/
fi