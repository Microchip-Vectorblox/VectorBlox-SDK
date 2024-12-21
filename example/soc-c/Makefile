CC ?= gcc
CXX ?= g++

kit = videokit
ifeq ($(kit),icicle)
	C_FLAGS+= -DUSE_INTERRUPTS=0 -DMSS_DDR
else
	C_FLAGS+= -DUSE_INTERRUPTS=1
endif


all:run-model

C_SRCS=pdma/pdma_helpers.c
C_SRCS+=image.c
C_SRCS+=libfixmath/fix16.c libfixmath/fix16_exp.c libfixmath/fix16_sqrt.c libfixmath/fix16_str.c
C_SRCS+=libfixmath/fix16_trig.c libfixmath/fract32.c libfixmath/uint32.c
C_SRCS+=postprocess.c postprocess_scrfd.c postprocess_ssd.c postprocess_retinaface.c postprocess_license_plate.c postprocess_pose.c
C_SRCS+=../../drivers/vectorblox/vbx_cnn_api.c ../../drivers/vectorblox/vbx_cnn_model.c
CXX_SRCS=run-model.cpp
C_OBJS=$(addsuffix .o,$(addprefix obj/,$(abspath $(C_SRCS))))
CXX_OBJS=$(addsuffix .o,$(addprefix obj/,$(abspath $(CXX_SRCS))))
C_FLAGS+=-Wall -O3 -I../../drivers/vectorblox/ -Ilibfixmath/ -Ipdma/ -MD -DVBX_SOC_DRIVER

$(CXX_OBJS) $(C_OBJS):
$(C_OBJS) $(CXX_OBJS):obj/%.o:%
	mkdir -p $(dir $@)
	$(CC) $(C_FLAGS) -c  $< -o $@

run-model: $(CXX_OBJS) $(C_OBJS)
	$(CXX) -o $@ $^ -ljpeg

.PHONY: overlay
overlay:
	cd dts && dtc -@ -O dtb -o mpfs_vbx.dtbo mpfs_vbx.dtso -Wno-reg_format -Wno-avoid_default_addr_size && \
	bash overlay.sh mpfs_vbx.dtbo

.PHONY: clean
clean:
	rm -rf run-video-model */*.dtbo *.o */*.o *.a obj
