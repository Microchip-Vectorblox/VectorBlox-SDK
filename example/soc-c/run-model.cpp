#include "postprocess.h"
#include "vbx_cnn_api.h"
#include <stdio.h>
#include <string.h>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <fcntl.h>

extern "C" int read_JPEG_file (const char * filename, int* width, int* height,
		unsigned char **image, const int grayscale);
extern "C" int resize_image(uint8_t *image_in, int in_w, int in_h,
		uint8_t *image_out, int out_w, int out_h);



#ifndef USE_INTERRUPTS
#define USE_INTERRUPTS 1
#endif


#if USE_INTERRUPTS
void enable_interrupt(vbx_cnn_t *vbx_cnn){
	uint32_t reenable = 1;
	ssize_t writeSize = write(vbx_cnn->fd, &reenable, sizeof(uint32_t));
	if(writeSize < 0) {
			close(vbx_cnn->fd);
		}
};
#endif


void* read_image(const char* filename, const int channels, const int height, const int width, int data_type){
	unsigned char* image;
	int h,w;
	read_JPEG_file (filename,&w,&h,&image, channels == 1);
	unsigned char* planer_img = (unsigned char*)malloc(w*h*channels);
	for(int r=0;r<h;r++){
		for(int c=0;c<w;c++){
			for(int ch=0;ch<channels;ch++){ // read as BGR
				planer_img[ch*w*h+r*w+c] = image[(r*w+c)*channels+((channels-1)-ch)];
			}
		}
	}
	free(image);

	unsigned char* resized_planar_img = (unsigned char*)malloc(width*height*channels);
	for(int ch=0;ch<channels;ch++){
		resize_image((uint8_t*)planer_img  +ch*w*h,w,h,
				(uint8_t*)resized_planar_img+ch*width*height,width,height);
	}
	free(planer_img);
	return resized_planar_img;
}


static uint32_t fletcher32(const uint16_t *data, size_t len)
{
	uint32_t c0, c1;
	unsigned int i;

	for (c0 = c1 = 0; len >= 360; len -= 360) {
		for (i = 0; i < 360; ++i) {
			c0 = c0 + *data++;
			c1 = c1 + c0;
		}
		c0 = c0 % 65535;
		c1 = c1 % 65535;
	}
	for (i = 0; i < len; ++i) {
		c0 = c0 + *data++;
		c1 = c1 + c0;
	}
	c0 = c0 % 65535;
	c1 = c1 % 65535;
	return (c1 << 16 | c0);
}

void *read_firmware_file(const char *filename) {
	void *firmware_instr = malloc(VBX_INSTRUCTION_SIZE);
	FILE *fd = fopen(filename, "r");
	int nread;
	if (!fd) {
		goto err;
	}
	nread = fread(firmware_instr, 1, VBX_INSTRUCTION_SIZE, fd);
	if (nread != VBX_INSTRUCTION_SIZE) {
		fprintf(stderr, "FILE %s is incorrect size. expected %d got %d\n", filename,
				VBX_INSTRUCTION_SIZE, nread);
		goto err;
	}
	return firmware_instr;

err:
	if (fd) {
		fclose(fd);
	}
	free(firmware_instr);
	return NULL;
}


model_t *read_model_file(vbx_cnn_t *vbx_cnn, const char *filename) {
	FILE *model_file = fopen(filename, "r");
	if (model_file == NULL) {
		return NULL;
	}
	fseek(model_file, 0, SEEK_END);
	int file_size = ftell(model_file);
	fseek(model_file, 0, SEEK_SET);
	model_t *model = (model_t *)malloc(file_size);
	printf("Reading model\n");
	int size_read = fread(model, 1, file_size, model_file);
	printf("Done\n");
	if (size_read != file_size) {
		fprintf(stderr, "Error reading full model file %s\n", filename);
		return NULL;
	}
	int model_data_size = model_get_data_bytes(model);
	if (model_data_size != file_size) {
		fprintf(stderr, "Error model file is not correct size%s\n", filename);
		return NULL;
	}
	int model_allocate_size = model_get_allocate_bytes(model);
	model = (model_t *)realloc(model, model_allocate_size);
	model_t *dma_model =
		(model_t *)vbx_allocate_dma_buffer(vbx_cnn, model_allocate_size, 0);
	if (dma_model) {
		memcpy(dma_model, model, model_data_size);
	}
	free(model);
	return dma_model;
}


int gettimediff_us(struct timeval start, struct timeval end) {

	int sec = end.tv_sec - start.tv_sec;
	int usec = end.tv_usec - start.tv_usec;
	return sec * 1000000 + usec;
}


int main(int argc, char **argv) {

	if(argc < 4){
		fprintf(stderr,
		"Usage: %s FIRMWARE_FILE MODEL_FILE IMAGE.jpg [POST_PROCESS]\n"
		"   if using POST_PROCESS to select post-processing, must be one of:\n"
		"   CLASSIFY, YOLOV2, YOLOV3, YOLOV4, YOLOV5,\n"
		"   BLAZEFACE, SCRFD, RETINAFACE, SSDV2, PLATE, LPD, LPR\n",
				argv[0]);
		return 1;
	}

	void *firmware_instr = read_firmware_file(argv[1]);
	if (!firmware_instr) {
		fprintf(stderr, "Unable to correctly read %s. Exiting\n", argv[1]);
		exit(1);
	}
	vbx_cnn_t *vbx_cnn = vbx_cnn_init(NULL, firmware_instr);
	if (!vbx_cnn) {
		fprintf(stderr, "Unable to initialize vbx_cnn. Exiting\n");
		exit(1);
	}
	model_t *model = read_model_file(vbx_cnn, argv[2]);
	if (!model) {
		fprintf(stderr, "Unable to correctly read %s. Exiting\n", argv[2]);
		exit(1);
	}
	if (model_check_sanity(model) != 0) {
		printf("Model %s is not sane\n", argv[2]);
	};

	int input_length = model_get_input_length(model, 0);
	uint8_t *input_buffer = (uint8_t *)vbx_allocate_dma_buffer(
			vbx_cnn, input_length * sizeof(uint8_t), 1);
	if(!input_buffer){
		fprintf(stderr, "Input_buffer requested exceeds buffer length.\n");
		exit(1);
	}
	void *read_buffer = NULL;
	if(std::string(argv[3]) != "TEST_DATA"){
		printf("Reading %s\n", argv[3]);
		int input_datatype = model_get_input_datatype(model,0);
		int* input_dims = model_get_input_dims(model,0);
		read_buffer = read_image(argv[3], input_dims[0], input_dims[1], input_dims[2], input_datatype);
		memcpy(input_buffer, read_buffer, input_length);
	} else {
		memcpy(input_buffer, (uint8_t *)model_get_test_input(model, 0),
				input_length);
	}

	vbx_cnn_io_ptr_t io_buffers[MAX_IO_BUFFERS];
	io_buffers[0] = (vbx_cnn_io_ptr_t)input_buffer;
	for (unsigned o = 0; o < model_get_num_outputs(model); ++o) {
		io_buffers[1 + o] = (vbx_cnn_io_ptr_t)vbx_allocate_dma_buffer(
				vbx_cnn, model_get_output_length(model, o) * sizeof(uint32_t), 0);
		if(!io_buffers[1 + o]){
			fprintf(stderr,"Model io_buffer requested exceeds buffer length.\n");
			exit(1);
		}
	}
#if USE_INTERRUPTS
	enable_interrupt(vbx_cnn);
#endif
	printf("Starting inference runs\n");
	for(int run=0; run < 4; ++run){
		struct timeval tv1, tv2;
		gettimeofday(&tv1, NULL);
		int status = vbx_cnn_model_start(vbx_cnn, model, io_buffers);
#if USE_INTERRUPTS
		status = vbx_cnn_model_wfi(vbx_cnn);
#else
		while(vbx_cnn_model_poll(vbx_cnn) > 0);
#endif
		gettimeofday(&tv2, NULL);
		if (status < 0) {
			printf("Model failed with error %d\n", vbx_cnn_get_error_val(vbx_cnn));
		}	
		printf("network took %d ms\n", gettimediff_us(tv1, tv2) / 1000);

	}

	// users can modify this post-processing function in post_process.c
	if (argc > 4) pprint_post_process(argv[2], argv[4], model, io_buffers);

	unsigned checksum = fletcher32((uint16_t*)(io_buffers[1]),model_get_output_length(model, 0)*sizeof(fix16_t)/sizeof(uint16_t));
	for(unsigned o =1;o<model_get_num_outputs(model);++o){
		checksum ^= fletcher32((uint16_t*)io_buffers[1+o], model_get_output_length(model, o)*sizeof(fix16_t)/sizeof(uint16_t));
	}
	printf("CHECKSUM = 0x%08x\n",checksum);
	if (read_buffer) free(read_buffer);

	return 0;
}
