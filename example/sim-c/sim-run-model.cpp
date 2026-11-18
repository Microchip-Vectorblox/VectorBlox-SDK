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


static int get_output_element_bytes(model_t *model, unsigned index) {
	int dt = model_get_output_datatype(model, index);
	if (dt == VBX_CNN_CALC_TYPE_INT32) return 4;
	if (dt == VBX_CNN_CALC_TYPE_INT16) return 2;
	return 1;
}

void* read_image(const char* filename, const int channels, const int height, const int width, int use_bgr){
	unsigned char* image;
	int h,w;
	read_JPEG_file (filename, &w, &h, &image, channels == 1);
	unsigned char* planar_img = (unsigned char*)malloc(w*h*channels);

	for (int r = 0; r < h; r++) {
		for (int c = 0; c < w; c++) {
			for (int ch = 0; ch < channels; ch++) {
				if (use_bgr) {
					planar_img[ch*w*h+r*w+c] = image[(r*w+c)*channels+((channels-1)-ch)]; // read as BGR
				} else {
					planar_img[ch*w*h+r*w+c] = image[(r*w+c)*channels+ch]; // read as RGB
				}
			}
		}
	}

	free(image);

	unsigned char* resized_planar_img = (unsigned char*)malloc(width * height * channels);
	for (int ch = 0; ch < channels; ch++) {
		resize_image((uint8_t*)planar_img + ch * w * h, w, h,
		(uint8_t*)resized_planar_img + ch * width * height, width, height);
	}

	free(planar_img);

	return resized_planar_img;
}
// Reads the model file and stores the model in memory 
model_t *read_model_file(const char *filename) {
	FILE* model_file = fopen(filename,"r");

	if(model_file == NULL){
		fprintf(stderr,"Unable to open file %s\n", filename);
		return NULL;
	}

	fseek(model_file, 0, SEEK_END);
	int file_size = ftell(model_file);
	fseek(model_file, 0, SEEK_SET);
	model_t* model = (model_t*)malloc(file_size);

	int size_read = fread(model, 1, file_size, model_file);
	fclose(model_file);

	const char *error = NULL;
	if (size_read != file_size){
		error = "Error reading the full model file";
	} else if (model_check_sanity(model) != 0) {
		error = "Model is not sane";
	} else if ((int)model_get_data_bytes(model) != file_size){
		error = "Model file is not correct size";
	} 

	if(error != NULL){
		fprintf(stderr, "Error with %s: %s\n", filename, error);
		free(model);
		return NULL;
	} 

	int model_allocate_size = model_get_allocate_bytes(model);
	model = (model_t*)realloc(model, model_allocate_size);

	return model;
}

// Initialize individual buffers and load input
void setup_io_buffers(model_t *model, vbx_cnn_io_ptr_t* io_buffers,vbx_cnn_io_ptr_t* output_buffers){
	for (int o = 0; o < (int)model_get_num_outputs(model); o++) {
		int output_length = model_get_output_length(model, o);
		output_buffers[o] = (uintptr_t)malloc(output_length*sizeof(uint32_t));
		
		io_buffers[(int)model_get_num_inputs(model) + o] = (uintptr_t)output_buffers[o];
		memset((void *)(io_buffers[(int)model_get_num_inputs(model) + o]), 0, (size_t)(output_length*sizeof(uint32_t)));
	}
}

// Load either test data or input image into input portion of the io_buffers
void load_input_to_buffer(model_t *model, vbx_cnn_io_ptr_t* io_buffers, const char * input_string) {
	for (int i = 0; i < (int)model_get_num_inputs(model); i++) {
		if (strcmp(input_string,"TEST_DATA")!=0) {										// Use input image path						
			int* input_shape = model_get_input_shape(model,i);
			int dims = model_get_input_dims(model,i);
			
			int use_bgr = 0; 															// Don't use_bgr
			io_buffers[i] = (uintptr_t)read_image(input_string, input_shape[dims-3],  
				input_shape[dims-2], input_shape[dims-1], use_bgr);  			
		} else {
			io_buffers[i] = (uintptr_t)model_get_test_input(model,i); 					// Use testdata path
		}
	}
}

// Performs checksum operation on output buffer for corresponding Model
unsigned obtain_buffer_checksum(model_t *model, vbx_cnn_io_ptr_t *output_buffer) {
	unsigned checksum = 0, c = 0;
	for (int o = 0; o < (int)model_get_num_outputs(model); o++) {
		int elem_bytes = get_output_element_bytes(model, o);
        int size_bytes = model_get_output_length(model, o) * elem_bytes;
		if(size_bytes % sizeof(uint16_t) == 0){
			c = fletcher32((uint16_t*)output_buffer[o], size_bytes / sizeof(uint16_t));
		}
		else{
			/* Need to pad with a trailing zero byte, like we do in model.py Fletcher32 */
			int padded_bytes = size_bytes + (size_bytes % sizeof(uint16_t));
			uint16_t *padded_out = (uint16_t *)malloc(padded_bytes * sizeof(uint8_t));
			memcpy(padded_out, (void *)output_buffer[o], size_bytes);
			((uint8_t *)padded_out)[size_bytes] = 0x00; /* pad byte */
			c = fletcher32(padded_out, padded_bytes / sizeof(uint16_t));
			free(padded_out);
		}
		
		checksum = (o == 0) ? c : checksum ^ c;
	}	
	return checksum;
}



int main(int argc, char** argv){
	// On hardware,ctrl_reg_addr, would be set with real values
	// because this is for the simulator, we use NULL
	void* ctrl_reg_addr = NULL; 		
	vbx_cnn_t* vbx_cnn = vbx_cnn_init(ctrl_reg_addr);

	if(argc < 2){
		fprintf(stderr,
		"Usage: %s  MODEL_FILE IMAGE.jpg [POST_PROCESS]\n"
		"   if using POST_PROCESS to select post-processing, must be one of:\n"
		"   CLASSIFY, YOLOV2, YOLOV3, YOLOV4, YOLOV5\n"
		"   OBJECT_DETECT, OBJECT_DETECT_FULL, OBB_DETECT, POSE_DETECT\n"
		"   BLAZEFACE, SCRFD, RETINAFACE, YUNET, SSDV2, PLATE, LPD, LPR, SPACE, SPACE_T\n",
				argv[0]);
		return 1;
	}
	
	// Read model file to model_t* model
	model_t* model = read_model_file(argv[1]);
	if (!model) return 1; 	// Error in read_model_file

	// Setup and load input output buffers
	vbx_cnn_io_ptr_t* io_buffers= (vbx_cnn_io_ptr_t*)malloc(sizeof(vbx_cnn_io_ptr_t)*(model_get_num_inputs(model)+model_get_num_outputs(model)));	
	vbx_cnn_io_ptr_t* output_buffers = (vbx_cnn_io_ptr_t*)malloc(sizeof(vbx_cnn_io_ptr_t)*(model_get_num_outputs(model)));
	setup_io_buffers(model, io_buffers, output_buffers); 
	const char * input_str = "TEST_DATA"; 
	if ((argc > 2) && (strcmp(argv[2],"TEST_DATA")!=0)) {	// if TEST_DATA is not used, load the image file name into input_str and use to load input into buffer
		input_str = argv[2];
	} 
	load_input_to_buffer(model, io_buffers, input_str); 

	// Simulate Model
#if TEST_OUT
	for (int o = 0; o < (int)model_get_num_outputs(model); o++) {
		free((void*)output_buffers[o]);										// this is needed to free the original output_buffers assigned in setup_io_buffers
		output_buffers[o] = (uintptr_t)model_get_test_output(model, o);		// reassign output_buffers to point to where test output is stored with the models memory
	}
	vbx_cnn_get_state(vbx_cnn); 											
#else
	vbx_cnn_model_start(vbx_cnn, model, io_buffers);
	int err = 1;
	while (err > 0) {
		err = vbx_cnn_model_poll(vbx_cnn);
	}
	if (err < 0) {
		printf("Model Run failed with error code: %d\n",err);
	}
#endif																									
	
	// Postprocessing outputs
	int is_int8 = 1;
	int display_fps = 0;
	if (argc > 3) pprint_post_process(argv[1], argv[3], model, (fix16_t**)(uintptr_t)output_buffers, is_int8, display_fps);

	// Compare checksums
	unsigned checksum = obtain_buffer_checksum(model, output_buffers);
	printf("CHECKSUM = 0x%08x\n", checksum);

	if(WRITE_OUT){
		print_json(model, io_buffers, INT8FLAG);
	}

	// Free memory
	if ((argc > 2) &&  (strcmp(argv[2], "TEST_DATA") != 0)) { // Clear input image from memory if input image is used
		for (int i = 0; i < (int)model_get_num_inputs(model); i++) {
			if ((void*)io_buffers[i] != NULL) free((void*)io_buffers[i]); 
		}
	}
#if !TEST_OUT // this prevents a double free since if TEST_OUT is used, output_buffers, points to output stored in model and original malloc has been freed in simulate section
	for (int o = 0; o < (int)model_get_num_outputs(model); o++) {
		if ((void*)output_buffers[o] != NULL) free((void*)output_buffers[o]);
	}
#endif
	free(io_buffers);
	free(output_buffers);
	free(model);

	return 0;
}