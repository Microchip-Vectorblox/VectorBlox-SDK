#include <stdio.h>
#include <string>
#include "vbx_cnn_api.h"
#include "postprocess.h"

#define TEST_OUT 0
#define INT8FLAG 1
#define WRITE_OUT 0
extern "C" int read_JPEG_file (const char * filename, int* width, int* height,
		unsigned char **image, const int grayscale);
extern "C" int resize_image(uint8_t* image_in,int in_w,int in_h,
		uint8_t* image_out,int out_w,int out_h);

void* read_image(const char* filename, const int channels, const int height, const int width, int data_type, int use_bgr){
	unsigned char* image;
	int h,w;
	read_JPEG_file (filename,&w,&h,&image, channels == 1);
	unsigned char* planer_img = (unsigned char*)malloc(w*h*channels);
	for(int r=0;r<h;r++){
		for(int c=0;c<w;c++){
			for(int ch=0;ch<channels;ch++){
				if (use_bgr) {
					planer_img[ch*w*h+r*w+c] = image[(r*w+c)*channels+((channels-1)-ch)]; // read as BGR
				} else {
					planer_img[ch*w*h+r*w+c] = image[(r*w+c)*channels+ch]; // read as RGB
				}
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


int main(int argc, char** argv){

	//On hardware would be set with real values
	//because this is for the simulator, we use NULL
	void* ctrl_reg_addr = NULL;
	vbx_cnn_t* vbx_cnn = vbx_cnn_init(ctrl_reg_addr);

	if(argc < 2){
		fprintf(stderr,
		"Usage: %s  MODEL_FILE IMAGE.jpg [POST_PROCESS]\n"
		"   if using POST_PROCESS to select post-processing, must be one of:\n"
		"   CLASSIFY, YOLOV2, YOLOV3, YOLOV4, YOLOV5\n"
		"   ULTRALYTICS, ULTRALYTICS_FULL, ULTRALYTICS_OBB, ULTRALYTICS_POSE\n"
		"   BLAZEFACE, SCRFD, RETINAFACE, SSDV2, PLATE, LPD, LPR\n",
				argv[0]);
		return 1;
	}
	
	FILE* model_file = fopen(argv[1],"r");
	if(model_file == NULL){
		fprintf(stderr,"Unable to open file %s\n", argv[1]);
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
		fprintf(stderr,"Error model file is not correct size %s\n",argv[1]);
	}
	int model_allocate_size = model_get_allocate_bytes(model);
	if (model_check_sanity(model) != 0) {
		printf("Model %s is not sane\n", argv[1]);
		exit(1);
	};
	model = (model_t*)realloc(model, model_allocate_size);
	vbx_cnn_io_ptr_t* io_buffers= (vbx_cnn_io_ptr_t*)malloc(sizeof(vbx_cnn_io_ptr_t)*(model_get_num_inputs(model)+model_get_num_outputs(model)));	
	vbx_cnn_io_ptr_t* output_buffers = (vbx_cnn_io_ptr_t*)malloc(sizeof(vbx_cnn_io_ptr_t)*(model_get_num_outputs(model)));
	//Initialize individual buffers
	for (unsigned o = 0; o < model_get_num_outputs(model); ++o) {
		int output_length = model_get_output_length(model, o);
		io_buffers[model_get_num_inputs(model) + o] = (uintptr_t)malloc(output_length*sizeof(fix16_t));
		output_buffers[o] = (uintptr_t)malloc(output_length*sizeof(uint32_t));
		io_buffers[model_get_num_inputs(model) + o] = (uintptr_t)output_buffers[o];
	}

	if (argc>2){
		if(strcmp(argv[2],"TEST_DATA")!=0){
			for (unsigned i = 0; i < model_get_num_inputs(model); ++i) {
				int input_datatype = model_get_input_datatype(model,i);
				int* input_shape = model_get_input_shape(model,i);
				int dims = model_get_input_dims(model,i);
				io_buffers[i] = (uintptr_t)read_image(argv[2], input_shape[dims-3], input_shape[dims-2], input_shape[dims-1], input_datatype, 0); // don't use_bgr
			}
		} else {
			for (unsigned i = 0; i < model_get_num_inputs(model); ++i) {
				io_buffers[i] = (uintptr_t)model_get_test_input(model,i);
			}
		}
	} else {
		for (unsigned i = 0; i < model_get_num_inputs(model); ++i) {
			io_buffers[i] = (uintptr_t)model_get_test_input(model,i);
		}
	}

#if TEST_OUT
	for(unsigned o =0; o<model_get_num_outputs(model); ++o){
		output_buffers[o] = (uintptr_t)model_get_test_output(model,o);
	}
	vbx_cnn_get_state(vbx_cnn);
	//buffers are now setup,
	//we can run the model.
#else
	vbx_cnn_model_start(vbx_cnn, model, io_buffers);
	int err=1;
	while (err>0) {
		err = vbx_cnn_model_poll(vbx_cnn);
	}
	if (err<0) {
		printf("Model Run failed with error code: %d\n",err);
	}
	// data should be available int the output buffers now.
#endif	
	

	fix16_t* fix16_output_buffers[model_get_num_outputs(model)];
	for (int o = 0; o < (int)model_get_num_outputs(model); ++o){
		int size=model_get_output_length(model, o);
		fix16_t scale = (fix16_t)model_get_output_scale_fix16_value(model,o); // get output scale
		int32_t zero_point = model_get_output_zeropoint(model,o); // get output zero
		fix16_output_buffers[o] = (fix16_t*)malloc(size*sizeof(fix16_t));
		int8_to_fix16(fix16_output_buffers[o], (int8_t*)io_buffers[model_get_num_inputs(model)+o], size, scale, zero_point);
	}	
	// users can modify this post-processing function in post_process.c
	
#if INT8FLAG
	if (argc > 3) pprint_post_process(argv[1], argv[3], model, (fix16_t**)(uintptr_t)output_buffers,1,0);
#else
	if (argc > 3) pprint_post_process(argv[1], argv[3], model, fix16_output_buffers,0,0);
#endif

	int output_bytes = model_get_output_datatype(model,0) == VBX_CNN_CALC_TYPE_INT16 ? 2 : 1;
	if (model_get_output_datatype(model,0) == VBX_CNN_CALC_TYPE_INT32) output_bytes = 4;
	unsigned checksum = fletcher32((uint16_t*)(io_buffers[model_get_num_inputs(model)]),model_get_output_length(model, 0)*output_bytes/sizeof(uint16_t));
	for(unsigned o =1;o<model_get_num_outputs(model);++o){
		int output_bytes = model_get_output_datatype(model,0) == VBX_CNN_CALC_TYPE_INT16 ? 2 : 1;
		if (model_get_output_datatype(model,0) == VBX_CNN_CALC_TYPE_INT32) output_bytes = 4;
		checksum ^= fletcher32((uint16_t*)io_buffers[model_get_num_inputs(model)+o], model_get_output_length(model, o)*output_bytes/sizeof(uint16_t));
	}
	printf("CHECKSUM = 0x%08x\n",checksum);
	if(WRITE_OUT){
		print_json(model, io_buffers, INT8FLAG);
	}
	if (argc>2){
		if(std::string(argv[2]) != "TEST_DATA"){
			for (unsigned i = 0; i < model_get_num_inputs(model); ++i) {
				if ((void*)io_buffers[i] != NULL) free((void*)io_buffers[i]);
				if ((void*)output_buffers[i] != NULL) free((void*)output_buffers[i]);
			}
		}
	}

	for (int o = 0; o < (int)model_get_num_outputs(model); ++o){
		if(fix16_output_buffers[o]){
			free((void*)fix16_output_buffers[o]);
		}
	}
	free(io_buffers);
	free(output_buffers);
	free(model);

	return 0;
}
