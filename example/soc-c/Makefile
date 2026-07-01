# =========================
# On-target Makefile
# =========================

# Compilers (override in env if needed)
CC ?= gcc
CXX ?= g++

# Select target hardware
kit = videokit
ifeq ($(kit),discovery)
	C_FLAGS+= -DUSE_INTERRUPTS=0 -DMSS_DDR
else ifeq ($(kit),icicle)
	C_FLAGS+= -DUSE_INTERRUPTS=0 -DMSS_DDR -DPDMA
else
	C_FLAGS+= -DUSE_INTERRUPTS=1 -DPDMA
endif

all:run-model

# -----------------
# Source file lists
# -----------------
# pdma_helpers only needed when PDMA is defined (non-discovery kits)
ifneq ($(kit),discovery)
C_SRCS=pdma/pdma_helpers.c
else
C_SRCS=
endif
C_SRCS+=../postprocess/image.c
C_SRCS+=../postprocess/libfixmath/fix16.c ../postprocess/libfixmath/fix16_exp.c ../postprocess/libfixmath/fix16_sqrt.c ../postprocess/libfixmath/fix16_str.c
C_SRCS+=../postprocess/libfixmath/fix16_trig.c ../postprocess/libfixmath/fract32.c ../postprocess/libfixmath/uint32.c
C_SRCS+=../postprocess/postprocess.c ../postprocess/postprocess_scrfd.c ../postprocess/postprocess_ssd.c ../postprocess/postprocess_retinaface.c ../postprocess/postprocess_license_plate.c ../postprocess/postprocess_pose.c ../postprocess/postprocess_space.c
C_SRCS+=../../drivers/vectorblox/vbx_cnn_api.c ../../drivers/vectorblox/vbx_cnn_model.c
DEBUG_SRC = ../../drivers/vectorblox/vbx_cnn_api_debug.c
ifneq ($(wildcard $(DEBUG_SRC)),)
  C_SRCS += $(DEBUG_SRC)
  C_FLAGS += -DDEBUG_CYCLE
endif

CXX_SRCS=run-model.cpp
C_OBJS=$(addsuffix .o,$(addprefix obj/,$(abspath $(C_SRCS))))
CXX_OBJS=$(addsuffix .o,$(addprefix obj/,$(abspath $(CXX_SRCS))))
C_FLAGS+=-Wall -O3 -I../../drivers/vectorblox/ -I../postprocess/libfixmath/ -Ipdma/ -I../postprocess/ -MD -DVBX_SOC_DRIVER



# Static pattern rules (explicitly bind lists to recipes)
$(CXX_OBJS) $(C_OBJS):
$(C_OBJS) $(CXX_OBJS):obj/%.o:%
	mkdir -p $(dir $@)
	$(CC) $(C_FLAGS) -c  $< -o $@

# Final link
run-model: $(CXX_OBJS) $(C_OBJS)
	$(CXX) -o $@ $^ -ljpeg

# Optional overlay rule
.PHONY: overlay
overlay:
	cd dts && dtc -@ -O dtb -o mpfs_vbx.dtbo mpfs_vbx.dtso -Wno-reg_format -Wno-avoid_default_addr_size && \
	bash overlay.sh mpfs_vbx.dtbo

.PHONY: clean
clean:
	rm -rf run-model */*.dtbo *.o */*.o *.a obj
