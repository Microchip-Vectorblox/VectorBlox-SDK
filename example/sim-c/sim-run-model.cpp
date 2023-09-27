#include <stdio.h>
#include <string>
#include "vbx_cnn_api.h"
#include "postprocess.h"


extern "C" int read_JPEG_file (const char * filename, int* width, int* height,
		unsigned char **image, const int grayscale);
extern "C" int resize_image(uint8_t* image_in,int in_w,int in_h,
		uint8_t* image_out,int out_w,int out_h);


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

int main(int argc, char** argv){

	//On hardware these two variables would be set with real values
	//because this is for the simulator, we use NULL
	void* ctrl_reg_addr = NULL;
	void* firmware_blob = NULL;
	vbx_cnn_t* vbx_cnn = vbx_cnn_init(ctrl_reg_addr,firmware_blob);

	if(argc < 3){
		fprintf(stderr,
		"Usage: %s  MODEL_FILE IMAGE.jpg [POST_PROCESS]\n"
		"   if using POST_PROCESS to select post-processing, must be one of:\n"
		"   CLASSIFY, YOLOV2, YOLOV3, YOLOV4, YOLOV5,\n"
		"   BLAZEFACE, SCRFD, RETINAFACE, SSDV2, PLATE, LPD, LPR\n",
				argv[0]);
		return 1;
	}

	FILE* model_file = fopen(argv[1],"r");
	if(model_file == NULL){
		printf("Unable to open file %s\n", argv[1]);
		return 1;
	}
	fseek(model_file,0,SEEK_END);
	int file_size = ftell(model_file);
	fseek(model_file,0,SEEK_SET);
	model_t* model = (model_t*)malloc(file_size);
	int size_read = fread(model,1,file_size,model_file);
	if(size_read != file_size){
		fprintf(stderr,"Error reading full model file %s\n",argv[1]);
	}
	int model_data_size = model_get_data_bytes(model);
	if(model_data_size != file_size){
		fprintf(stderr,"Error model file is not correct size%s\n",argv[1]);
	}
	int model_allocate_size = model_get_allocate_bytes(model);
	model = (model_t*)realloc(model, model_allocate_size);
	uint8_t* input_buffer=NULL;
	void* read_buffer=NULL;
	if(std::string(argv[2]) != "TEST_DATA"){
		int input_datatype = model_get_input_datatype(model,0);
		int* input_dims = model_get_input_dims(model, 0);
		read_buffer = read_image(argv[2], input_dims[0], input_dims[1], input_dims[2], input_datatype);
		input_buffer = (uint8_t*)read_buffer;
	} else {
		input_buffer = (uint8_t*)model_get_test_input(model,0);
	}
	vbx_cnn_io_ptr_t* io_buffers= (vbx_cnn_io_ptr_t*)malloc(sizeof(vbx_cnn_io_ptr_t)*(1+model_get_num_outputs(model)));
	io_buffers[0] = (uintptr_t)input_buffer;
	for (unsigned o = 0; o < model_get_num_outputs(model); ++o) {
		int output_length = model_get_output_length(model, o);
		io_buffers[o+1] = (uintptr_t)malloc(output_length*sizeof(fix16_t));
	}

	//buffers are now setup,
	//we can run the model.

	vbx_cnn_model_start(vbx_cnn, model, io_buffers);
	int err=1;
	while (err>0) {
		err = vbx_cnn_model_poll(vbx_cnn);
	}
	if (err<0) {
		printf("Model Run failed with error code: %d\n",err);
	}
	// data should be available int the output buffers now.
	
	// users can modify this post-processing function in post_process.c
	if (argc > 3) pprint_post_process(argv[1], argv[3], model, io_buffers);


	unsigned checksum = fletcher32((uint16_t*)(io_buffers[1]),model_get_output_length(model, 0)*sizeof(fix16_t)/sizeof(uint16_t));
	for(unsigned o =1;o<model_get_num_outputs(model);++o){
		checksum ^= fletcher32((uint16_t*)io_buffers[1+o], model_get_output_length(model, o)*sizeof(fix16_t)/sizeof(uint16_t));
	}
	printf("CHECKSUM = 0x%08x\n",checksum);
	if (read_buffer) {
		free(read_buffer);
	}
	free(model);

	return 0;
}
