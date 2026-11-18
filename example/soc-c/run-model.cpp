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
#include <cassert>
#ifdef PDMA
#include "pdma/pdma_helpers.h"
#endif

extern "C" int read_JPEG_file (const char * filename, int* width, int* height,
		unsigned char **image, const int grayscale);
extern "C" int resize_image(uint8_t *image_in, int in_w, int in_h,
		uint8_t *image_out, int out_w, int out_h);

#ifndef USE_INTERRUPTS
#define USE_INTERRUPTS 1
#endif
#define INT8FLAG 1

#ifndef NUM_LOOPS
#define NUM_LOOPS 1
#endif

#ifndef DEBUG_CYCLE
#define DEBUG_CYCLE 0
#endif

#define TEST_OUT 0

// Utility functions
static inline void* virt_to_phys(vbx_cnn_t* vbx_cnn,void* virt) {
	return (char*)(virt) + vbx_cnn->dma_phys_trans_offset;
}

int gettimediff_us(struct timeval start, struct timeval end) {

	int sec = end.tv_sec - start.tv_sec;
	int usec = end.tv_usec - start.tv_usec;
	return sec * 1000000 + usec;
}

static int get_output_element_bytes(model_t *model, unsigned index) {
    int dt = model_get_output_datatype(model, index);
    if (dt == VBX_CNN_CALC_TYPE_INT32) return 4;
    if (dt == VBX_CNN_CALC_TYPE_INT16) return 2;
    return 1;
}

void enable_interrupt(vbx_cnn_t *vbx_cnn) {
	uint32_t reenable = 1;
	ssize_t writeSize = write(vbx_cnn->fd, &reenable, sizeof(uint32_t));
	if(writeSize < 0) {
			close(vbx_cnn->fd);
		}
};

// Use PDMA functions
#ifdef PDMA
int8_t* g_pdma_mmap_t; // Global var
uint64_t pdma_mmap(int total_size/*total num of all outputs*/) {
	char cdev[256] = "/dev/udmabuf-ddr-c0";
	uint64_t ddrc_phyadr = get_phy_addr(cdev);  
	int32_t fdc = open(cdev, O_RDWR); 
	off_t oft = 0;
	g_pdma_mmap_t = (int8_t *)mmap(NULL, total_size*sizeof(int8_t), PROT_READ | PROT_WRITE,  MAP_SHARED, fdc, oft);
	assert(g_pdma_mmap_t != NULL);
	uint64_t output_data_phys = ddrc_phyadr + oft;
	return output_data_phys;
}

int32_t pdma_ch_transfer(uint64_t output_data_phys, void* source_buffer,int offset,int size,vbx_cnn_t *vbx_cnn,int32_t channel) {
	uint64_t srcbuf=0x3000000000 + (uint64_t)(uintptr_t)virt_to_phys(vbx_cnn, source_buffer);
	return pdma_ch_cpy(output_data_phys + offset, srcbuf, size, channel);
}
#endif

typedef struct {
	uint64_t pdma_out = 0;
	int32_t pdma_channel = 0;
} buffer_info_t;

buffer_info_t init_pdma(){
	int total_size = 32*1024*1024; 	// TODO: Check limit size in comparison
	buffer_info_t buffer_info;
	#ifdef PDMA				
	buffer_info.pdma_out = pdma_mmap(total_size);
	buffer_info.pdma_channel = pdma_ch_open();
	#endif
	return buffer_info;
}

void write_output_buffer(model_t *model, vbx_cnn_io_ptr_t *io_buffers, vbx_cnn_io_ptr_t *output_buffer, vbx_cnn_t *vbx_cnn, uint64_t pdma_out, int32_t pdma_channel){	

	int output_offset = 0;
	for (unsigned o = 0; o <  model_get_num_outputs(model); o++) {
		#ifdef PDMA				
			int output_length = model_get_output_length(model, o);
			pdma_ch_transfer(pdma_out, (void*)io_buffers[model_get_num_inputs(model) + o], output_offset, output_length, vbx_cnn, pdma_channel);
			output_buffer[o] = (vbx_cnn_io_ptr_t)(g_pdma_mmap_t + output_offset);
			output_offset+= output_length;
		#else
			output_buffer[o] = io_buffers[model_get_num_inputs(model) + o];	
		#endif
	}
}	

void* read_image(const char* filename, const int channels, const int height, const int width, int use_bgr) {
	unsigned char* image;
	int h,w;
	read_JPEG_file (filename,&w,&h,&image, channels == 1);
	unsigned char* planer_img = (unsigned char*)malloc(w*h*channels);
	
	for (int r = 0; r < h; r++) {
		for (int c = 0; c < w; c++) {
			for (int ch = 0; ch < channels; ch++) {
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
	for (int ch = 0; ch < channels; ch++) {
		resize_image((uint8_t*)planer_img  +ch*w*h,w,h,
				(uint8_t*)resized_planar_img+ch*width*height,width,height);
	}
	free(planer_img);

	return resized_planar_img;
}

// Read input file for .ucomp of Unstructred Compression (UCOMP)
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

// Read input file for .vnnx of No Compression (NCOMP) or Compression (COMP)
model_t *read_model_file(vbx_cnn_t *vbx_cnn, const char *filename) {
	FILE *model_file = fopen(filename, "r");
	if(model_file == NULL){
		fprintf(stderr,"Unable to open file %s\n", filename);
		return NULL;
	}
	fseek(model_file, 0, SEEK_END);
	int file_size = ftell(model_file);
	fseek(model_file, 0, SEEK_SET);
	model_t *model = (model_t *)malloc(file_size);
	printf("Reading model\n");
	int size_read = fread(model, 1, file_size, model_file);
	printf("Done\n");
	fclose(model_file);

	if (size_read != file_size) {
		fprintf(stderr, "Error reading full model file %s\n", filename);
		free(model);
		return NULL;
	}
	int model_data_size = model_get_data_bytes(model);
	if (model_data_size != file_size) {
		fprintf(stderr, "Error model file is not correct size%s\n", filename);
		free(model);
		return NULL;
	}

	int model_allocate_size = model_get_allocate_bytes(model);
	model = (model_t *)realloc(model, model_allocate_size);
	model_t *dma_model = (model_t *)vbx_allocate_dma_buffer(vbx_cnn, model_allocate_size, 0);  // third arg passed is phys_alignment_bit 
	if (dma_model) {
		memcpy(dma_model, model, model_data_size);
	}

	free(model);
	return dma_model;
}

bool verify_model(model_t *model, vbx_cnn_t *vbx_cnn, const char *filename) {
	if (!model) {
		fprintf(stderr, "Unable to correctly read %s. Exiting\n", filename);
		return false;
	}
	const char *error_msg = NULL;
	switch (model_check_configuration(model, vbx_cnn)) {
	case -1:
		error_msg = "version mismatch. Please generate the model with appropriate version of VBX_SDK,";
		break;
	case -2:
		error_msg = "and VBX_CORE compression configuration mismatch. Make sure the compression configuration is set properly when the model is generated.";
		break;
	case -3:
		error_msg = "and VBX_CORE size configuration mismatch. Make sure the size configuration is set properly when the model is generated.";
		break;
	}

	if (error_msg) {
		fprintf(stderr, "Model %s %s\n", filename, error_msg); 
		return false;
	}
	return true;
}

// Setting values to input buffer(s)
void load_input_to_buffer(model_t *model, vbx_cnn_io_ptr_t *io_buffers, const char *input_name, bool use_test_data) {
	int use_bgr = 0;	// read input img as RGB if use_test_data == false
	void *read_buffer = NULL;
	if (!use_test_data) printf("Reading %s\n", input_name);
	for (int i = 0; i < (int)model_get_num_inputs(model); i++) {
		if(use_test_data){
			io_buffers[i] = (vbx_cnn_io_ptr_t)(uint8_t*)model_get_test_input(model,i);
		} else {
			int* input_shape = model_get_input_shape(model,i);
			int input_length = model_get_input_length(model, i);
			int dims = model_get_input_dims(model,i);
			if(!io_buffers[i]) {
				fprintf(stderr, "io_buffers not allocated.\n");
				exit(1);
			}
			read_buffer = read_image(input_name, input_shape[dims-3], input_shape[dims-2], input_shape[dims-1], use_bgr);
			memcpy((void*)io_buffers[i], read_buffer, (size_t)input_length);
			free(read_buffer);
			read_buffer = NULL;

		}
	}
}

// Setup and initialize buffers:
// Note: io_buffers = the io_buffers to assign
// Note: model_header is req'ed for unstructured compression models
void setup_io_buffers(model_t *model, vbx_cnn_t *vbx_cnn, vbx_cnn_io_ptr_t *io_buffers, uint8_t *model_header, bool is_unstructured_compression) {
	unsigned num_inputs = model_get_num_inputs(model);
	unsigned num_outputs =  model_get_num_outputs(model);

	for (unsigned i = 0; i < num_inputs; i++) {		// Allocate input buffers
		io_buffers[i] = (vbx_cnn_io_ptr_t)vbx_allocate_dma_buffer(vbx_cnn, model_get_input_length(model,i)*sizeof(uint8_t),1);
		if(!io_buffers[i]) {
			fprintf(stderr,"Model io_buffer requested exceeds buffer length.\n");
			exit(1);
		}
	}
	
	for (unsigned o = 0; o < num_outputs; o++) {	// Allocate output buffers
		if (is_unstructured_compression) {			// For unstructured compression (UCOMP) only 
			uint32_t output_length = model_get_output_length(model, o);
			uint32_t output_offset = 0;
			for (unsigned j = 0; j < num_outputs; j++) {
				uint32_t output_size = *((uint32_t*)model_header + 7 + (2 * j));
				output_offset = *((uint32_t*)model_header + 8 + (2 * j));
				if (output_length == output_size) {
					break;
				}
			}
			io_buffers[num_inputs + o] = (vbx_cnn_io_ptr_t)((uintptr_t)model + (uintptr_t)output_offset);
		} else {
			io_buffers[num_inputs + o] = (vbx_cnn_io_ptr_t)vbx_allocate_dma_buffer(vbx_cnn, model_get_output_length(model, o) * sizeof(uint32_t), 0);
			if(!io_buffers[num_inputs + o]) {
				fprintf(stderr,"Model io_buffer requested exceeds buffer length.\n");
				exit(1);
			}
			memset((void *)(io_buffers[num_inputs + o]), 0, (size_t)(model_get_output_length(model, o) * sizeof(uint32_t)));
		}
	}
}

// Run inference on a model and print runtime per cycle
void model_inference(model_t *model, model_t *tsnp_model, vbx_cnn_t *vbx_cnn, uint32_t input_offset, vbx_cnn_io_ptr_t *io_buffers, bool is_unstructured_compression) {
	printf("Starting inference runs\n");
	struct timeval tv1, tv2;
	gettimeofday(&tv1, NULL);

	for (int run = 0; run < NUM_LOOPS; run++) {
		int status;
		if (is_unstructured_compression) {
			status = vbx_tsnp_model_start(vbx_cnn, model, tsnp_model, input_offset, io_buffers);	// Run unstructured compression model
		} else {
			status = vbx_cnn_model_start(vbx_cnn, model, io_buffers); 								// Run no compression or compression model
		}
		// Check status of model
#if USE_INTERRUPTS
		enable_interrupt(vbx_cnn);
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
#if DEBUG_CYCLE
	vbx_cnn_get_and_print_cycle_counters(vbx_cnn);
#endif
}

void inference_or_load_test_outputs(model_t *model, model_t *tsnp_model, vbx_cnn_t *vbx_cnn, uint32_t input_offset, vbx_cnn_io_ptr_t *io_buffers, bool is_unstructured_compression){
#if TEST_OUT
	// Note: TEST_OUT loads test outputs that are embedded within model
	for(unsigned o = 0; o < model_get_num_outputs(model); ++o){								
		io_buffers[model_get_num_inputs(model) + o] = (uintptr_t)model_get_test_output(model,o);
	}
	vbx_cnn_get_state(vbx_cnn);
#else
	// Run inference
	model_inference(model, tsnp_model, vbx_cnn, input_offset, io_buffers, is_unstructured_compression);	
#endif
}

// Performs checksum operation on output buffer for corresponding Model
unsigned obtain_buffer_checksum(model_t *model, vbx_cnn_io_ptr_t *output_buffer) {
	unsigned checksum = 0, c = 0;
	for (int o = 0; o < (int)model_get_num_outputs(model); o++) {
		int elem_bytes = get_output_element_bytes(model, o);
        int size_bytes = model_get_output_length(model, o) * elem_bytes;
        size_bytes += (size_bytes % sizeof(uint16_t));
		unsigned c = fletcher32((uint16_t*)(output_buffer[o]),size_bytes/sizeof(uint16_t));
		checksum = (o == 0) ? c : checksum ^ c;
	}	
	return checksum;
}


unsigned find_expected_checksum(model_t *model, bool  is_test_data_arg, bool is_unstructured_compression){
	vbx_cnn_io_ptr_t test_output_buffer[MAX_IO_BUFFERS];	
	if ((is_test_data_arg) && (is_unstructured_compression)) {	
		for (unsigned o = 0; o < model_get_num_outputs(model); o++) {
			test_output_buffer[o] = (vbx_cnn_io_ptr_t)model_get_test_output(model, o); 
		}
		return obtain_buffer_checksum(model, test_output_buffer);
	}
	return 0;
}



int main(int argc, char **argv) {

	if(argc < 2) {
		fprintf(stderr,
		"Usage: %s MODEL_FILE IMAGE.jpg [POST_PROCESS]\n"
		"   if using POST_PROCESS to select post-processing, must be one of:\n"
		"   CLASSIFY, YOLOV2, YOLOV3, YOLOV4, YOLOV5, OBJECT_DETECT, OBJECT_DETECT_FULL\n"
		"   BLAZEFACE, SCRFD, RETINAFACE, YUNET, SSDV2, PLATE, LPD, LPR, SPACE, SPACE_T\n",
				argv[0]);
		return 1;
	}
	
	vbx_cnn_t *vbx_cnn = vbx_cnn_init(NULL);
	if (!vbx_cnn) {
		fprintf(stderr, "Unable to initialize vbx_cnn. Exiting\n");
		exit(1);
	}

	model_t *model = NULL;
	model_t *tsnp_model = NULL;
	uint8_t *model_header = NULL;
	uint32_t input_offset = 0;
	bool is_unstructured_compression = (vbx_cnn->comp_config == 2);

	// Read Model
	if (is_unstructured_compression) { 							
		ucomp_model_t *ucomp_model = read_ucompmodel_file(vbx_cnn, argv[1]);
		model = ucomp_model->mxp_model;
		tsnp_model = ucomp_model->tsnp_model;
		model_header = ucomp_model->header_info;
		input_offset = *((uint32_t*)model_header + 6);
	} else {
		model = read_model_file(vbx_cnn, argv[1]);
	}

	// Check Model settings
	if(!(verify_model(model, vbx_cnn, argv[1]))) exit(1);

	// Configure PDMA, if PDMA is not used, pdma_out & pdma_channel = 0
	buffer_info_t buffer_info = init_pdma();
	uint64_t pdma_out = buffer_info.pdma_out;
	int32_t pdma_channel = buffer_info.pdma_channel;

	bool is_test_data_arg = ((argc > 2) && (std::string(argv[2]) == "TEST_DATA"));
    bool use_test_data = ((argc <= 2) || is_test_data_arg);
	
	// Expected checksum is only used to validate UCOMP models, for NCOMP/COMP models find_expected_checksum returns 0
	// Use sim-c in the SDK to validate checksums for NCOMP/COMP models
	unsigned expected_checksum = find_expected_checksum(model, is_test_data_arg, is_unstructured_compression);
	
	// Setup and Initialize io_buffers
	vbx_cnn_io_ptr_t io_buffers[MAX_IO_BUFFERS];
	setup_io_buffers(model, vbx_cnn, io_buffers, model_header, is_unstructured_compression);
	load_input_to_buffer(model, io_buffers, argv[2], use_test_data);

	// Run model inference or load test output from model
	inference_or_load_test_outputs(model, tsnp_model, vbx_cnn, input_offset, io_buffers, is_unstructured_compression); 

	// Load output to output_buffer for postprocessing	
	vbx_cnn_io_ptr_t output_buffer[model_get_num_outputs(model)];	
	write_output_buffer(model, io_buffers, output_buffer, vbx_cnn, pdma_out, pdma_channel);

	// Perform postprocessing on outputs
	if (argc > 3) pprint_post_process(argv[1], argv[3], model, (fix16_t**)(uintptr_t)output_buffer, 1 /*int 8 flag*/, 0 /*display fps*/);
	
	
	// Find and print checksum on outputs
	unsigned checksum = 0;
	#ifdef PDMA
	for (int o = 0; o < (int)model_get_num_outputs(model); o++) {
		int elem_bytes = get_output_element_bytes(model, o);
        int size_bytes = model_get_output_length(model, o) * elem_bytes;
        size_bytes += (size_bytes % sizeof(uint16_t));
		unsigned c = fletcher32((uint16_t*)(io_buffers[model_get_num_inputs(model)+o]),size_bytes/sizeof(uint16_t));
		checksum = (o == 0) ? c : checksum ^ c;
	}	
	#else
	checksum = obtain_buffer_checksum(model, output_buffer);
	#endif

	printf("CHECKSUM = %08x \n", checksum);
	
	
	if ((is_test_data_arg) && (is_unstructured_compression) && (expected_checksum != checksum)) { 
		printf("Checksum mismatch for the model %s: expected = %08x, actual = %08x \n", argv[1], expected_checksum, checksum);
	}
	if ((getenv("WRITE_OUT") != NULL) || (argc <= 3 && (!strcmp(argv[1], "test.vnnx")))) {
		print_json(model, io_buffers, INT8FLAG); 
	}
	return 0;
}
