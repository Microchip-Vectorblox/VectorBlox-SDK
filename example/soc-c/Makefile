all:run-model

libfixmath.a:$(wildcard libfixmath-master/libfixmath/*.c)
	curl -s https://codeload.github.com/PetteriAimonen/libfixmath/zip/master > libfixmath-master.zip
	rm -rf libfixmath-master
	unzip libfixmath-master.zip
	cd libfixmath-master/libfixmath/;gcc -c *.c
	ar qc $@ libfixmath-master/libfixmath/*.o

C_SRCS=image.c
C_SRCS+=postprocess.c
C_SRCS+=../../drivers/vectorblox/vbx_cnn_api.c
C_SRCS+=../../drivers/vectorblox/vbx_cnn_model.c
CXX_SRCS=run-model.cpp
C_OBJS=$(addsuffix .o,$(addprefix obj/,$(abspath $(C_SRCS))))
CXX_OBJS=$(addsuffix .o,$(addprefix obj/,$(abspath $(CXX_SRCS))))
DEPS=$(addsuffix .d,$(addprefix obj/,$(abspath $(C_SRCS) $(CXX_SRCS))))
C_FLAGS=-Wall -I../../drivers/vectorblox -Ilibfixmath-master/ -MD -DVBX_SOC_DRIVER
-include $(DEPS)
$(CXX_OBJS) $(C_OBJS):libfixmath.a
$(C_OBJS) $(CXX_OBJS):obj/%.o:%
	mkdir -p $(dir $@)
	gcc $(C_FLAGS) -c  $< -o $@

run-model: $(CXX_OBJS) $(C_OBJS) libfixmath.a
	g++ -o $@ $^ -ljpeg
