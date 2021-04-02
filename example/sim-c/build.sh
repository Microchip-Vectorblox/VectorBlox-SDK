
curl -s https://codeload.github.com/PetteriAimonen/libfixmath/zip/master > libfixmath-master.zip
rm -rf libfixmath-master
unzip libfixmath-master.zip
(
	cd libfixmath-master/libfixmath/
	gcc -c *.c
)
ar qc libfixmath.a libfixmath-master/libfixmath/*.o
gcc sim-run-model.cpp image.c postprocess.c libfixmath.a  \
    -lstdc++ \
	-I../../drivers/vectorblox -Ilibfixmath-master/ \
	-ljpeg -lm -lvbx_cnn_sim -L../../lib -Wl,-rpath='$ORIGIN/../../lib' -o sim-run
