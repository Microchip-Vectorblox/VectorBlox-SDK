CC ?= gcc
CXX ?= g++

all:sim-run-model

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
CXX_SRCS=sim-run-model.cpp
C_OBJS=$(addsuffix .o,$(addprefix obj/,$(abspath $(C_SRCS))))
CXX_OBJS=$(addsuffix .o,$(addprefix obj/,$(abspath $(CXX_SRCS))))
C_FLAGS=-Wall -I../../drivers/vectorblox -Ilibfixmath-master/
$(CXX_OBJS) $(C_OBJS):libfixmath.a
$(C_OBJS) $(CXX_OBJS):obj/%.o:%
	mkdir -p $(dir $@)
	$(CC) $(C_FLAGS) -c  $< -o $@

sim-run-model: $(CXX_OBJS) $(C_OBJS) libfixmath.a
	$(CXX) -o $@ $^ -ljpeg -lm -lvbx_cnn_sim -L../../lib -Wl,-rpath='$$ORIGIN/../../lib'


.PHONY: clean
clean:
	rm -rf sim-run-model *.o *.a obj libfixmath*
