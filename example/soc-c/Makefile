CC ?= gcc
CXX ?= g++

all:run-model

libfixmath.a:$(wildcard libfixmath-master/libfixmath/*.c)
	wget --no-check-certificate https://codeload.github.com/PetteriAimonen/libfixmath/zip/master -O libfixmath-master.zip
	rm -rf libfixmath-master
	unzip libfixmath-master.zip
	cd libfixmath-master/libfixmath/;$(CC) -c *.c
	ar qc $@ libfixmath-master/libfixmath/*.o

C_SRCS=image.c
C_SRCS+=postprocess.c
C_SRCS+=postprocess_scrfd.c
C_SRCS+=postprocess_ssd.c
C_SRCS+=postprocess_retinaface.c
C_SRCS+=../../drivers/vectorblox/vbx_cnn_api.c
C_SRCS+=../../drivers/vectorblox/vbx_cnn_model.c
CXX_SRCS=run-model.cpp
C_OBJS=$(addsuffix .o,$(addprefix obj/,$(abspath $(C_SRCS))))
CXX_OBJS=$(addsuffix .o,$(addprefix obj/,$(abspath $(CXX_SRCS))))
C_FLAGS=-Wall -I../../drivers/vectorblox -Ilibfixmath-master/ -MD -DVBX_SOC_DRIVER

$(CXX_OBJS) $(C_OBJS):libfixmath.a
$(C_OBJS) $(CXX_OBJS):obj/%.o:%
	mkdir -p $(dir $@)
	$(CC) $(C_FLAGS) -c  $< -o $@

run-model: $(CXX_OBJS) $(C_OBJS) libfixmath.a
	$(CXX) -o $@ $^ -ljpeg

.PHONY: overlay
overlay:
	cd dts && dtc -@ -O dtb -o mpfs_icicle_vbx.dtbo mpfs_icicle_vbx.dtso -Wno-reg_format -Wno-avoid_default_addr_size && \
	/opt/microchip/dt-overlays/overlay.sh mpfs_icicle_vbx.dtbo

.PHONY: clean
clean:
	rm -rf run-model *.o *.a obj libfixmath*
