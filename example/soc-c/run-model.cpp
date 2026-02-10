#include "postprocess.h"
#include "vbx_cnn_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include "pdma/pdma_helpers.h"
#include <cassert>

extern "C" int read_JPEG_file (const char * filename, int* width, int* height,
		unsigned char **image, const int grayscale);
extern "C" int resize_image(uint8_t *image_in, int in_w, int in_h,
		uint8_t *image_out, int out_w, int out_h);


#define TFLITE 1
#ifndef USE_INTERRUPTS
#define USE_INTERRUPTS 1
#endif
#define TEST_OUT 0
#define INT8FLAG 1

#ifndef NUM_LOOPS
#define NUM_LOOPS 1
#endif

static inline void* virt_to_phys(vbx_cnn_t* vbx_cnn,void* virt){
	return (char*)(virt) + vbx_cnn->dma_phys_trans_offset;
}
#if USE_INTERRUPTS
void enable_interrupt(vbx_cnn_t *vbx_cnn){
	uint32_t reenable = 1;
	ssize_t writeSize = write(vbx_cnn->fd, &reenable, sizeof(uint32_t));
	if(writeSize < 0) {
			close(vbx_cnn->fd);
		}
};
#endif

int8_t* pdma_mmap_t;
uint64_t pdma_mmap(int total_size/*total num of all outputs*/){
	
	char cdev[256] = "/dev/udmabuf-ddr-c0";
	uint64_t ddrc_phyadr = get_phy_addr(cdev);  
	int32_t fdc = open(cdev, O_RDWR); 
	off_t oft = 0;
	pdma_mmap_t = (int8_t *)mmap(NULL, total_size*sizeof(int8_t), PROT_READ | PROT_WRITE,  MAP_SHARED, fdc, oft);
	assert(pdma_mmap_t != NULL);
	uint64_t output_data_phys = ddrc_phyadr + oft;

	return output_data_phys;

}


int32_t pdma_ch_transfer(uint64_t output_data_phys, void* source_buffer,int offset,int size,vbx_cnn_t *vbx_cnn,int32_t channel){
	uint64_t srcbuf=0x3000000000 + (uint64_t)(uintptr_t)virt_to_phys(vbx_cnn, source_buffer);
	return pdma_ch_cpy(output_data_phys + offset, srcbuf, size, channel);

}


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

ucomp_model_t *read_ucompmodel_file(vbx_cnn_t *vbx_cnn, const char *filename) {
	ucomp_model_t *model_info = (ucomp_model_t*)malloc(sizeof(ucomp_model_t));
	FILE *model_file = fopen(filename, "r");
	if (model_file == NULL) {
		return NULL;
	}
	fseek(model_file, 0, SEEK_END);
	int file_size = ftell(model_file);
	fseek(model_file, 0, SEEK_SET);
	model_t *model = (model_t *)malloc(file_size);
	int size_read = fread(model, 1, file_size, model_file);
	if (size_read != file_size) {
		fprintf(stderr, "Error reading full model file %s\n", filename);
		return NULL;
	}
	fclose(model_file);
	
	// Read binary header information
	uint32_t header_size = *((uint32_t*)model);
	uint32_t mxp_model_size  = *((uint32_t*)model + 1);
	uint32_t tsnp_model_size = *((uint32_t*)model + 2);
	
	// Copy header information
	model_info->header_info = (uint8_t*)vbx_allocate_dma_buffer(vbx_cnn, header_size, 0);
	if (model_info->header_info) {
		memcpy(model_info->header_info, model, header_size);
	} else {
		return 0;
	}
	
	// Copy mxp version of the model
	uint32_t model_allocate_size = model_get_allocate_bytes((model_t*)((char*)model + header_size));
	model_info->mxp_model = (model_t*)vbx_allocate_dma_buffer(vbx_cnn, model_allocate_size, 0);
	if (model_info->mxp_model) {
		memcpy(model_info->mxp_model, (model_t*)((char*)model + header_size), mxp_model_size);
	} else {
		return NULL;
	}
	
	// Copy tsnp version of the model with 4096 byte alignment
	model_info->tsnp_model = (model_t*)vbx_allocate_dma_buffer(vbx_cnn, tsnp_model_size, 12);
	if (model_info->tsnp_model) {
		memcpy(model_info->tsnp_model, (model_t*)((char*)model + header_size + mxp_model_size), tsnp_model_size);
	} else {
		return NULL;
	}	
	
	free(model);
	return model_info;
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

	if(argc < 2){
		fprintf(stderr,
		"Usage: %s MODEL_FILE IMAGE.jpg [POST_PROCESS]\n"
		"   if using POST_PROCESS to select post-processing, must be one of:\n"
		"   CLASSIFY, YOLOV2, YOLOV3, YOLOV4, YOLOV5, ULTRALYTICS, ULTRALYTICS_FULL\n"
		"   BLAZEFACE, SCRFD, RETINAFACE, SSDV2, PLATE, LPD, LPR\n",
				argv[0]);
		return 1;
	}

	vbx_cnn_t *vbx_cnn = vbx_cnn_init(NULL);
	if (!vbx_cnn) {
		fprintf(stderr, "Unable to initialize vbx_cnn. Exiting\n");
		exit(1);
	}
	
	model_t *model;
	model_t *tsnp_model = NULL;
	uint8_t *model_header = NULL;
	uint32_t input_offset = 0;
	if (vbx_cnn->comp_config == 2) {
		ucomp_model_t *ucomp_model = read_ucompmodel_file(vbx_cnn, argv[1]);
		model = ucomp_model->mxp_model;
		tsnp_model = ucomp_model->tsnp_model;
		model_header = ucomp_model->header_info;
		input_offset = *((uint32_t*)model_header + 6);
	} else {
		model = read_model_file(vbx_cnn, argv[1]);
	}
	if (!model) {
		fprintf(stderr, "Unable to correctly read %s. Exiting\n", argv[1]);
		exit(1);
	}
	int verify_model = model_check_configuration(model, vbx_cnn);
	if (verify_model == -1) {
		printf("Model %s version mismatch. Please generate the model with appropriate version of VBX_SDK \n", argv[1]);
		exit(1);
	} else if (verify_model == -2) {
		printf("Model %s and VBX_CORE compression configuration mismatch. Make sure the compression configuration is set properly when the model is generated. \n", argv[1]);
		exit(1);
	} else if (verify_model == -3) {
		printf("Model %s and VBX_CORE size configuration mismatch. Make sure the size configuration is set properly when the model is generated. \n", argv[1]);
		exit(1);
	}
	int total_size = 32*1024*1024; //#TODO Check limit size in comparison
	
	uint64_t pdma_out = pdma_mmap(total_size);
	int32_t pdma_channel = pdma_ch_open();
	void *read_buffer = NULL;
	
	vbx_cnn_io_ptr_t io_buffers[MAX_IO_BUFFERS];
	for(unsigned i =0;i<model_get_num_inputs(model);++i){
		io_buffers[i] = (vbx_cnn_io_ptr_t)vbx_allocate_dma_buffer(vbx_cnn, model_get_input_length(model,i)*sizeof(uint8_t),1);
		if(!io_buffers[i]){
			fprintf(stderr,"Model io_buffer requested exceeds buffer length.\n");
			exit(1);
		}
	}
	
	unsigned expected_checksum = 0;
	if(argc > 2){
		if(std::string(argv[2]) != "TEST_DATA"){
			printf("Reading %s\n", argv[2]);
			for (unsigned i = 0; i < model_get_num_inputs(model); ++i){
				int input_datatype = model_get_input_datatype(model,i);
				int* input_shape = model_get_input_shape(model,i);
				int input_length = model_get_input_length(model, i);
				int dims = model_get_input_dims(model,i);
				uint8_t *input_buffer = (uint8_t *)vbx_allocate_dma_buffer(vbx_cnn, input_length * sizeof(uint8_t), 0);
				if(!input_buffer){
					fprintf(stderr, "Input_buffer requested exceeds buffer length.\n");
					exit(1);
				}
				int use_bgr=0; //read as RGB
				read_buffer = read_image(argv[2], input_shape[dims-3], input_shape[dims-2], input_shape[dims-1], input_datatype,use_bgr);
				memcpy(input_buffer, read_buffer, input_length);
				io_buffers[i] = (vbx_cnn_io_ptr_t)input_buffer;
			}
		}
		else {
			for (unsigned i = 0; i < model_get_num_inputs(model); ++i) {
				io_buffers[i] = (vbx_cnn_io_ptr_t)(uint8_t*)model_get_test_input(model,i);
			}
			int output_bytes = model_get_output_datatype(model,0) == VBX_CNN_CALC_TYPE_INT16 ? 2 : 1;
			if (model_get_output_datatype(model,0) == VBX_CNN_CALC_TYPE_INT32) output_bytes = 4;
			expected_checksum = fletcher32((uint16_t*)model_get_test_output(model, 0), model_get_output_length(model, 0)*output_bytes/sizeof(uint16_t));
			for(unsigned o =1;o<model_get_num_outputs(model);++o){
				output_bytes = model_get_output_datatype(model,o) == VBX_CNN_CALC_TYPE_INT16 ? 2 : 1;
				if (model_get_output_datatype(model,o) == VBX_CNN_CALC_TYPE_INT32) output_bytes = 4;
				expected_checksum ^= fletcher32((uint16_t*)model_get_test_output(model, o), model_get_output_length(model, o)*output_bytes/sizeof(uint16_t));
			}
		}
	} else {
		for (unsigned i = 0; i < model_get_num_inputs(model); ++i) {
			io_buffers[i] = (vbx_cnn_io_ptr_t)(uint8_t*)model_get_test_input(model,i);
		}
		int output_bytes = model_get_output_datatype(model,0) == VBX_CNN_CALC_TYPE_INT16 ? 2 : 1;
		if (model_get_output_datatype(model,0) == VBX_CNN_CALC_TYPE_INT32) output_bytes = 4;
		expected_checksum = fletcher32((uint16_t*)model_get_test_output(model, 0), model_get_output_length(model, 0)*output_bytes/sizeof(uint16_t));
		for(unsigned o =1;o<model_get_num_outputs(model);++o){
			output_bytes = model_get_output_datatype(model,o) == VBX_CNN_CALC_TYPE_INT16 ? 2 : 1;
			if (model_get_output_datatype(model,o) == VBX_CNN_CALC_TYPE_INT32) output_bytes = 4;
			expected_checksum ^= fletcher32((uint16_t*)model_get_test_output(model, o), model_get_output_length(model, o)*output_bytes/sizeof(uint16_t));
		}
	}
	unsigned num_outputs = model_get_num_outputs(model);
	for (unsigned o = 0; o < num_outputs; ++o) {
		if (vbx_cnn->comp_config == 2) {
			uint32_t output_length = model_get_output_length(model, o);
			unsigned j;
			uint32_t output_offset = 0;
			for(j = 0; j < num_outputs; j++) {
				uint32_t output_size = *((uint32_t*)model_header + 7 + (2 * j));
				output_offset = *((uint32_t*)model_header + 8 + (2 * j));
				if (output_length == output_size) {
					break;
				}
			}
			io_buffers[model_get_num_inputs(model) + o] = (vbx_cnn_io_ptr_t)((uint32_t)(uintptr_t)model + output_offset);
		} else {
			io_buffers[model_get_num_inputs(model) + o] = (vbx_cnn_io_ptr_t)vbx_allocate_dma_buffer(
					vbx_cnn, model_get_output_length(model, o) * sizeof(uint32_t), 0);
			if(!io_buffers[model_get_num_inputs(model) + o]){
				fprintf(stderr,"Model io_buffer requested exceeds buffer length.\n");
				exit(1);
			}
			memset((void *)(io_buffers[model_get_num_inputs(model) + o]), 0, (size_t)(model_get_output_length(model, o) * sizeof(uint32_t)));
		}
	}
		
#if USE_INTERRUPTS
	enable_interrupt(vbx_cnn);
#endif
#if TEST_OUT
	for(unsigned o =0; o<model_get_num_outputs(model); ++o){
		io_buffers[model_get_num_inputs(model) + o] = (uintptr_t)model_get_test_output(model,o);
	}
	vbx_cnn_get_state(vbx_cnn);
	//buffers are now setup,
	//we can run the model.
#else
	printf("Starting inference runs\n");
	struct timeval tv1, tv2;
	gettimeofday(&tv1, NULL);
	for(int run = 0; run < NUM_LOOPS; ++run){
		int status;
		if (vbx_cnn->comp_config == 2) {
			status = vbx_tsnp_model_start(vbx_cnn, model, tsnp_model, input_offset, io_buffers);
		} else {
			status = vbx_cnn_model_start(vbx_cnn, model, io_buffers);
		}
#if USE_INTERRUPTS
		status = vbx_cnn_model_wfi(vbx_cnn);
#else
		while(vbx_cnn_model_poll(vbx_cnn) > 0);
#endif
		if (status < 0) {
			printf("Model failed with error %d\n", vbx_cnn_get_error_val(vbx_cnn));
		}	
	}
	gettimeofday(&tv2, NULL);
	printf("network took %3.4f ms (%3.2f FPS) over %d cycles\n", gettimediff_us(tv1, tv2) * 1.0 / 1000 / NUM_LOOPS, 1000./(gettimediff_us(tv1, tv2) * 1.0 / 1000 / NUM_LOOPS), NUM_LOOPS);
#endif
#if INT8FLAG
	// users can modify this post-processing function in post_process.c
	vbx_cnn_io_ptr_t pdma_buffer[model_get_num_outputs(model)];
	int output_offset=0;
	for(int o =0; o<(int)model_get_num_outputs(model);o++){
#if !TEST_OUT
		int output_length = model_get_output_length(model, o);
		pdma_ch_transfer(pdma_out, (void*)io_buffers[model_get_num_inputs(model)+o], output_offset, output_length, vbx_cnn, pdma_channel);
		pdma_buffer[o] = (vbx_cnn_io_ptr_t)(pdma_mmap_t + output_offset);
		output_offset+= output_length;
#else
		pdma_buffer[o] = (vbx_cnn_io_ptr_t)model_get_test_output(model, o);
#endif
	}
	if (argc > 3) pprint_post_process(argv[1], argv[3], model, (fix16_t**)(uintptr_t)pdma_buffer,1,0);
#else	
	fix16_t* fix16_output_buffers[model_get_num_outputs(model)];
	for (int o = 0; o < (int)model_get_num_outputs(model); ++o){
		int size=model_get_output_length(model, o);
		fix16_t scale = (fix16_t)model_get_output_scale_fix16_value(model,o); // get output scale
		int32_t zero_point = model_get_output_zeropoint(model,o); // get output zero
		fix16_output_buffers[o] = (fix16_t*)malloc(size*sizeof(fix16_t));
		int8_to_fix16(fix16_output_buffers[o], (int8_t*)io_buffers[model_get_num_inputs(model)+o], size, scale, zero_point);
	}
	if (argc > 3) pprint_post_process(argv[1], argv[3], model, fix16_output_buffers,0,0);
#endif
	
	int output_bytes = model_get_output_datatype(model,0) == VBX_CNN_CALC_TYPE_INT16 ? 2 : 1;
	if (model_get_output_datatype(model,0) == VBX_CNN_CALC_TYPE_INT32) output_bytes = 4;
	unsigned checksum = 0;
	int size_of_output_in_bytes = model_get_output_length(model, 0)*output_bytes;
	size_of_output_in_bytes += (size_of_output_in_bytes % sizeof(uint16_t)); // output buffers are init to uint32_t, OK to increase byte if odd (can only happen if int8)
	if (vbx_cnn->comp_config == 2) {
		checksum = fletcher32((uint16_t*)pdma_buffer[0], size_of_output_in_bytes/sizeof(uint16_t));
	}
	else {
		checksum = fletcher32((uint16_t*)(io_buffers[model_get_num_inputs(model)]),size_of_output_in_bytes/sizeof(uint16_t));
	}
	for(unsigned o =1;o<model_get_num_outputs(model);++o){
		int output_bytes = model_get_output_datatype(model,o) == VBX_CNN_CALC_TYPE_INT16 ? 2 : 1;
		if (model_get_output_datatype(model,o) == VBX_CNN_CALC_TYPE_INT32) output_bytes = 4;
		int size_of_output_in_bytes = model_get_output_length(model, o)*output_bytes;
		size_of_output_in_bytes += (size_of_output_in_bytes % sizeof(uint16_t));
		if (vbx_cnn->comp_config == 2) {
			checksum ^= fletcher32((uint16_t*)pdma_buffer[o], size_of_output_in_bytes/sizeof(uint16_t));
		} else {
			checksum ^= fletcher32((uint16_t*)io_buffers[model_get_num_inputs(model)+o], size_of_output_in_bytes/sizeof(uint16_t));
		}
	}
	printf("CHECKSUM = %08x \n", checksum);
	if(argc>=3){
		if ((expected_checksum != checksum) && (std::string(argv[2]) == "TEST_DATA")) {
			printf("Checksum mismatch for the model %s: expected = %08x, actual = %08x \n", argv[1], expected_checksum, checksum);
		}
	}
	if(getenv("WRITE_OUT") != NULL || (argc<=3 && !strcmp(argv[1],"test.vnnx"))){
		print_json(model,io_buffers,INT8FLAG);
	}
	if (read_buffer) free(read_buffer);
	return 0;
}
