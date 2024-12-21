CC ?= gcc
CXX ?= g++

all:sim-run-model


C_SRCS=image.c
C_SRCS+=libfixmath/fix16.c libfixmath/fix16_exp.c libfixmath/fix16_sqrt.c libfixmath/fix16_str.c
C_SRCS+=libfixmath/fix16_trig.c libfixmath/fract32.c libfixmath/uint32.c
C_SRCS+=postprocess.c postprocess_scrfd.c postprocess_ssd.c postprocess_retinaface.c postprocess_license_plate.c postprocess_pose.c
CXX_SRCS=sim-run-model.cpp
C_OBJS=$(addsuffix .o,$(addprefix obj/,$(abspath $(C_SRCS))))
CXX_OBJS=$(addsuffix .o,$(addprefix obj/,$(abspath $(CXX_SRCS))))
C_FLAGS=-Wall -I../../drivers/vectorblox/ -Ilibfixmath/

$(CXX_OBJS) $(C_OBJS):
$(C_OBJS) $(CXX_OBJS):obj/%.o:%
	mkdir -p $(dir $@)
	$(CC) $(C_FLAGS) -c  $< -o $@

sim-run-model: $(CXX_OBJS) $(C_OBJS)
	$(CXX) -o $@ $^ -ljpeg -lm -lvbx_cnn_sim -L../../lib -Wl,-rpath='$$ORIGIN/../../lib'


.PHONY: clean
clean:
	rm -rf sim-run-model *.o *.a obj
