CC ?= gcc
CXX ?= g++

all:run-video-model

C_SRCS=libfixmath/fix16.c libfixmath/fix16_exp.c libfixmath/fix16_sqrt.c libfixmath/fix16_str.c
C_SRCS+=libfixmath/fix16_trig.c libfixmath/fract32.c libfixmath/uint32.c
C_SRCS+=libfixmatrix/fixarray.c libfixmatrix/fixmatrix.c
C_SRCS+=postprocess.c postprocess_scrfd.c postprocess_ssd.c postprocess_retinaface.c postprocess_license_plate.c
C_SRCS+=frameDrawing/ascii_characters.c frameDrawing/draw_assist.c frameDrawing/draw.c
C_SRCS+=imageScaler/scaler.c
C_SRCS+=warpAffine/warp.c
C_SRCS+=tracking.c detectionDemo.c recognitionDemo.c
C_SRCS+=../../drivers/vectorblox/vbx_cnn_api.c ../../drivers/vectorblox/vbx_cnn_model.c
CXX_SRCS=run-video-model.cpp
C_OBJS=$(addsuffix .o,$(addprefix obj/,$(abspath $(C_SRCS))))
CXX_OBJS=$(addsuffix .o,$(addprefix obj/,$(abspath $(CXX_SRCS))))
C_FLAGS=-Wall -I./ -I../../drivers/vectorblox/ -IframeDrawing/ -IimageScaler/ -IwarpAffine/ -Ilibfixmath/ -Ilibfixmatrix/ -MD -DVBX_SOC_DRIVER

$(CXX_OBJS) $(C_OBJS):
$(C_OBJS) $(CXX_OBJS):obj/%.o:%
	mkdir -p $(dir $@)
	$(CC) $(C_FLAGS) -c  $< -o $@

run-video-model: $(CXX_OBJS) $(C_OBJS)
	$(CXX) -o $@ $^ -ljpeg


.PHONY: overlay
overlay:
	cd dts && dtc -@ -O dtb -o mpfs_vbx.dtbo mpfs_vbx.dtso -Wno-reg_format -Wno-avoid_default_addr_size && \
		bash overlay.sh mpfs_vbx.dtbo

.PHONY: hdmi
hdmi:
	i=0; while [ "$$i" -le 60 ]; do \
		devmem2 0x40001110 w 1; \
		sleep 0.0167; \
		i=$$((i + 1)); \
	done

.PHONY: clean
clean:
	rm -rf run-video-model */*.dtbo *.o */*.o *.a obj
