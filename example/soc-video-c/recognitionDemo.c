#include "recognitionDemo.h"
#include "imageScaler/scaler.h"
#include "warpAffine/warp.h"
#include "frameDrawing/draw_assist.h"
#include "frameDrawing/draw.h"
#include <string.h>
#include <sys/time.h>

extern volatile uint32_t* PROCESSING_FRAME_ADDRESS;
extern volatile uint32_t* PROCESSING_NEXT_FRAME_ADDRESS;
extern volatile uint32_t* PROCESSING_NEXT2_FRAME_ADDRESS;
extern volatile uint32_t* SCALER_BASE_ADDRESS;
extern volatile uint32_t* WARP_BASE_ADDRESS;
extern volatile uint32_t* RED_DDR_FRAME_START_ADDR;
extern volatile uint32_t* GREEN_DDR_FRAME_START_ADDR;
extern volatile uint32_t* BLUE_DDR_FRAME_START_ADDR;

static inline void* virt_to_phys(vbx_cnn_t* vbx_cnn,void* virt){
  return (char*)(virt) + vbx_cnn->dma_phys_trans_offset;
}

extern int fps;
extern uint32_t* overlay_draw_frame;

// Globals Specification
#if VBX_SOC_DRIVER
	#include "pdma/pdma_helpers.h"
	#define MAX_TRACKS 48
	#define DB_LENGTH 32
	extern int delete_embedding_mode;
	extern int add_embedding_mode;
	extern int capture_embedding;
#if PDMA
	extern int8_t* pdma_mmap_t;
	extern uint64_t pdma_out;
	extern int32_t pdma_channel;
#endif	
	
#else
	#define MAX_TRACKS 20
	#define DB_LENGTH 4
	int delete_embedding_mode=0;
	int add_embedding_mode = 0;
	int capture_embedding = 0;
#endif

uint8_t* warp_temp_buffer = NULL;

// Database Embeddings
#define PRESET_DB_LENGTH 4
#define EMBEDDING_LENGTH 128
#define PIPELINED 1
int id_check = 0;
int db_end_idx = 4;
int face_count = 0;
char *db_nameStr[DB_LENGTH] = {
	"Bob",
	"John",
	"Nancy",
	"Tina",
};

int16_t db_embeddings[DB_LENGTH][EMBEDDING_LENGTH] = {
	{8507, 7292, 270, -1620, 6819, 5537, 5199, 5942, 7900, -5199, 8507, -5131, -3241, 8507, -1688, 8507, -5064, 3443, -1688, 1283, -473, -2026, -5334, 8507, -675, 6414, -8710, 1620, 8507, -2363, -743, -4591, 4051, 3714, -7427, 8507, 3106, 1891, -4456, -6144, 8507, -743, 4591, -8575, 8507, 7562, 6077, 8507, 2026, -3579, -473, 6279, -4389, 8507, -8710, 8507, 4389, 4389, -4996, 4861, 7562, 8507, 1013, -3781, -405, -4861, -540, -4456, 7292, -3781, -8710, 8507, 2026, -1823, 1215, 7900, 8507, -3376, 8440, 4186, -4051, -5266, 8507, -8710, 5739, 5874, 2498, -1553, 743, -540, -7360, 270, 8507, -2296, -8710, 8507, -1553, -2701, 135, -2296, -8710, -1891, -8710, 1553, 8507, -8710, 608, -4254, 270, -8710, -3646, -8035, 2836, 6144, 8507, -1080, 5672, -3781, 8507, 8507, 2093, 4456, 8507, -2971, 8507, 5199, 3646, -8710},
	{7925, 6728, 9420, 9420, 2691, 3364, -6429, -9644, -523, -6056, -9644, -6878, -9644, 2841, 75, 1495, -150, -3289, -3738, 9420, -1869, 1719, -2019, 8298, -3962, 972, -9644, 7028, 1944, -2617, -2766, 1794, -5682, 9420, -9420, -9644, -1121, 5757, -2019, -2318, 3289, -374, -3888, 5682, 3663, -2019, -1121, -3065, 1869, -4710, 1271, 6504, 9420, -9644, 1495, 9420, -4261, 4187, 9420, 9420, 6280, 8523, 8822, -9644, -2467, 9420, -150, -3364, -3364, -3065, -5233, -5084, 7551, -374, 9270, 4486, 7401, -7850, 673, 1196, 822, -5308, -1346, 1196, 748, -523, 9420, 150, 9270, -6355, 7028, 6205, 4785, 1794, -8523, 9420, -3364, -1719, 673, -1346, 5607, -1944, -748, -9644, -5233, -972, -6205, -1719, -4560, -9644, 897, -3962, -598, 2019, -2093, -2691, 9420, 9420, -7102, 7999, 5383, -5757, 8224, -5757, 4635, 9420, -4411, -299},
	{-8541, -424, -918, -5365, -5506, 2612, -847, 494, -4306, -9106, 2965, 8683, -2471, -9106, 4165, -9106, 8894, 8894, -3106, 8894, 2612, 8894, -3318, 1624, -4447, 8894, 4306, -3812, 7412, 2541, -4165, -2047, 5083, 8894, -71, 212, 141, -1341, -141, 1694, -847, -2471, -2471, 6565, -2259, 3812, 8259, -2965, -7059, 1412, -4941, 847, -4094, -353, -9106, -7271, 5859, 8894, 8894, -6071, 8894, 4871, -7059, 4730, 1129, -7130, 0, 706, -6424, -5718, -4941, 5365, 1059, -988, 5435, 6000, -7694, -6212, 988, -3953, -2612, 8894, -9106, 1129, -9106, -9106, 1200, -9106, 8753, 3530, -9106, -3671, -282, 2118, 8894, -7553, 7271, 4235, -3741, -2541, -1624, -7341, 8894, -6565, -2541, -1553, 8894, -7977, 8894, 8894, 4659, 8894, 4941, -847, 8894, 4800, 8894, 7059, -8259, -706, -1977, -6424, 5224, 2047, 6918, -706, -6847, -5930},
	{-7403, 3352, -4400, -629, 8799, -8311, -9009, -8869, 7612, -9009, 5727, -6565, -9009, -7333, -2374, 5447, 8799, 2724, 4051, 4889, -1816, -1676, -9009, -279, -8939, -2235, -9009, -3352, -2793, 419, -3841, -9009, -5517, -2724, 8799, -3282, 3212, 6285, -6215, -2724, 7612, 8031, 3632, -2863, -4120, 8799, 3981, -7333, 2584, 1676, -3771, 8799, -9009, 6844, -4470, -5936, -5796, 6425, 2235, 698, 4330, -4889, -9009, 3841, 7822, -5936, -1467, 1117, 4400, -9009, -1397, 8799, -838, -6355, -3771, -6635, -1606, 8799, 5796, -8660, 4958, 7682, 8799, 8799, -6844, -1187, -6006, -6285, 8799, -8660, -629, 8799, -2165, 8799, -70, 2305, -1187, -5377, -6355, 559, -5866, 3981, -1327, -9009, -7123, 1048, 4609, -1536, 349, -2025, 1467, 3422, 5727, -2235, 2444, -2793, -3003, 8799, -7403, 768, 5028, 8799, -5028, 2025, 8799, -5098, -3352, -2305},
};


bool not_duplicate(char* id){
	for(int i = 0; i < db_end_idx;i++){
		if(strcmp(db_nameStr[i],id)==0){
			return false;
		}
	}
	return true;

}

void append_name(char* name_entered){
	if(id_check ==1){
		if(strlen(name_entered)>0){
			strcpy(db_nameStr[db_end_idx-1],name_entered);
		}
		else{
			face_count++;
		}
		printf("Embedding '%s' added to database\n",db_nameStr[db_end_idx-1]);
	}


	id_check = 0;
	add_embedding_mode=0;

}


void print_list(){
	for(int i=0; i <db_end_idx; i++){
		printf("%d: %s\n",i+1,db_nameStr[i]); // index starts at 1
	}
	printf("\n");

}
#if PDMA
static int32_t pdma_ch_transfer(uint64_t output_data_phys, void* source_buffer,int offset,int size,vbx_cnn_t *vbx_cnn,int32_t channel){
	uint64_t srcbuf=0x3000000000 + (uint64_t)(uintptr_t)virt_to_phys(vbx_cnn, source_buffer);
	return pdma_ch_cpy(output_data_phys + offset, srcbuf, size, channel);
}
#endif
void embedding_calc(fix16_t* embedding, struct model_descr_t* recognition_model){
	fix16_t sum = 0;
	fix16_t temp[128];
	int8_t* output_buffer_int8 = (int8_t*)(uintptr_t)recognition_model->model_output_buffer[0];
	int32_t zero_point = model_get_output_zeropoint(recognition_model->model,0);
	fix16_t scale = model_get_output_scale_fix16_value(recognition_model->model,0);
	for(int n = 0; n < recognition_model->model_output_length[0]; n++){
		temp[n] = int8_to_fix16_single(output_buffer_int8[n], scale,  zero_point);
		sum += fix16_sq(temp[n]);
	}
	fix16_t norm = fix16_div(fix16_one, fix16_sqrt(sum));
	for(int n = 0; n < recognition_model->model_output_length[0]; n++){	
		embedding[n] = fix16_mul(temp[n], norm);
	}	
}

void delete_embedding(char* input_buf,struct model_descr_t* models,uint8_t modelIdx){

	int rem_ind = -1;
	size_t len = strlen(input_buf);
	memmove(input_buf, input_buf, len);
	rem_ind = atoi(input_buf) - 1; // index starts at 1 (needed as atoi returns 0 if not an integer)
	if((rem_ind <db_end_idx) && (rem_ind >= 0) && len>1){
		printf("\nEmbedding '%s' removed\n", db_nameStr[rem_ind]);		
		for(int re = rem_ind; re<db_end_idx;re++)
		{	
			for(int i = 0; i< EMBEDDING_LENGTH;i++)
				db_embeddings[re][i] = db_embeddings[re+1][i];	
			db_nameStr[re] = db_nameStr[re+1];
		}
		trackClean(models,modelIdx);
		db_end_idx--;
	} else if (len > 1) {
		printf("Embedding index invalid\n");
	}
	delete_embedding_mode = 0;
	printf("Exiting +/- embedding mode \n");

}


void tracksInit(struct model_descr_t* models){
	int use_plate = 0;
	struct model_descr_t *recognition_model = models;
	if(!strcmp(recognition_model->post_process_type, "LPR"))
		use_plate =1;
	track_t *tracks = (track_t*)calloc(MAX_TRACKS,sizeof(track_t));
	Tracker_t *pTracker = (Tracker_t *)calloc(1,sizeof(Tracker_t));
	track_t** pTracks = (track_t**)calloc(MAX_TRACKS,sizeof(track_t*));
	recognition_model->tracks = tracks;
	recognition_model->pTracker = pTracker;
	recognition_model->pTracks = pTracks;
	trackerInit(pTracker, MAX_TRACKS, pTracks, tracks,use_plate);

}


void trackClean(struct model_descr_t* models, uint8_t modelIdx){

	struct model_descr_t *recognition_model = models + modelIdx;
	free(recognition_model->tracks);
	free(recognition_model->pTracker);
	free(recognition_model->pTracks);
	recognition_model->tracks = NULL;
	recognition_model->pTracker = NULL;
	recognition_model->pTracks = NULL;

}
int facing_front(object_t* object){
	
	int bbox_h = fix16_to_int(object->box[3] - object->box[1]);
	int center_y = fix16_to_int(object->box[3] + object->box[1]) / 2;
	int bbox_h_frontal = bbox_h * 0.5;
	int frontal_bbox_top =      center_y - (bbox_h_frontal/2);
	int frontal_bbox_bottom =   center_y + (bbox_h_frontal/2);
	
	int mouth_left = fix16_to_int(object->points[3][0]);
	int mouth_right = fix16_to_int(object->points[4][0]);
	int nose_x = fix16_to_int(object->points[2][0]);
	int nose_y = fix16_to_int(object->points[2][1]);

	if(nose_x < mouth_left || nose_x > mouth_right || nose_y < frontal_bbox_top || nose_y > frontal_bbox_bottom){
		return 0;
	}
	else{
		return 1;
	}	
}


short recognitionDemoInit(vbx_cnn_t* the_vbx_cnn, struct model_descr_t* models, uint8_t modelIdx, int has_attribute_model, int screen_height, int screen_width, int screen_y_offset, int screen_x_offset) {
	struct model_descr_t *detect_model = models + modelIdx;
	// Allocate memory for Models
    // Allocate Memory needed for the Detect Model buffers
	detect_model->model_io_buffers  = vbx_allocate_dma_buffer(the_vbx_cnn, (1+model_get_num_outputs(detect_model->model))*sizeof(detect_model->model_io_buffers[0]), 0);
	if(!detect_model->model_io_buffers){
		printf("Memory allocation issue for model io buffers.\n");
		return -1;
	}
	// Allocate the buffers for output for Detect Model
    for (unsigned o = 0; o < model_get_num_outputs(detect_model->model); ++o) {
		detect_model->model_output_length[o] = model_get_output_length(detect_model->model, o);
		detect_model->pipelined_output_buffers[0][o] = vbx_allocate_dma_buffer(the_vbx_cnn, model_get_output_length(detect_model->model, o)*sizeof(fix16_t), 0);
		detect_model->pipelined_output_buffers[1][o] = vbx_allocate_dma_buffer(the_vbx_cnn, model_get_output_length(detect_model->model, o)*sizeof(fix16_t), 0);
		detect_model->model_io_buffers[o+1] = (uintptr_t)detect_model->pipelined_output_buffers[0][o];
		if(!detect_model->pipelined_output_buffers[0][o] ||!detect_model->pipelined_output_buffers[1][o] ){
			printf("Memory allocation issue for model output buffers.\n");
			return -1;	
		}		
		
	}
	// Allocate the buffer for input for Detect Model
	detect_model->model_input_buffer = vbx_allocate_dma_buffer(the_vbx_cnn, model_get_input_length(detect_model->model, 0)*sizeof(uint8_t), 0);
	detect_model->model_io_buffers[0] = (uintptr_t)detect_model->model_input_buffer;
	if(!detect_model->model_input_buffer) {
		printf("Memory allocation issue for model input buffers.\n");
		return -1;	
	}
	detect_model->buf_idx = 0;
	detect_model->is_running = 0;

	// Allocate memory for Recognition Model I/Os
	// Specify the input size for Recognition Model
	struct model_descr_t *recognition_model = models + modelIdx + 1;
	recognition_model->coord4 = (fix16_t*)malloc(8*sizeof(fix16_t));

	if(!strcmp(recognition_model->post_process_type, "ARCFACE")){
		recognition_model->coord4[0] = F16(38.2946);
		recognition_model->coord4[1] = F16(51.6963);
		recognition_model->coord4[2] = F16(73.5318);
		recognition_model->coord4[3] = F16(51.5014);
		recognition_model->coord4[4] = F16(56.0252);
		recognition_model->coord4[5] = F16(71.7366);
		recognition_model->coord4[6] = F16(56.1396);
		recognition_model->coord4[7] = F16(92.284805);
	} else if(!strcmp(recognition_model->post_process_type, "SPHEREFACE")){
		recognition_model->coord4[0] = F16(30.2946);
		recognition_model->coord4[1] = F16(51.6963);
		recognition_model->coord4[2] = F16(65.5318);
		recognition_model->coord4[3] = F16(51.5014);
		recognition_model->coord4[4] = F16(48.0252);
		recognition_model->coord4[5] = F16(71.7366);
		recognition_model->coord4[6] = F16(48.1396);
		recognition_model->coord4[7] = F16(92.2848);
	} else if(!strcmp(recognition_model->post_process_type, "LPR")){
		recognition_model->coord4[0] = fix16_from_int(1);  //LT
		recognition_model->coord4[1] = fix16_from_int(1);
		recognition_model->coord4[2] = fix16_from_int(146-1);  //RT
		recognition_model->coord4[3] = fix16_from_int(1);
		recognition_model->coord4[6] = fix16_from_int(1);  //LB
		recognition_model->coord4[7] = fix16_from_int(34-1);
	}
	else {
		printf("Recognition Model does not have an expected input length\n");
		return -1;
	}
	// Allocate the buffer for input for Recognition Model
	recognition_model->model_input_buffer = vbx_allocate_dma_buffer(the_vbx_cnn, model_get_input_length(recognition_model->model, 0)*sizeof(uint8_t), 0);
	if(!recognition_model->model_input_buffer){
		printf("Memory allocation issue with recognition input buffer.\n");
		return -1;
	}
	// Specify the output size for Recognition Model
	recognition_model->model_output_length[0] = model_get_output_length(recognition_model->model, 0);
	// Allocate the buffer for output for Recognition Model
	recognition_model->model_output_buffer[0] = vbx_allocate_dma_buffer(the_vbx_cnn, recognition_model->model_output_length[0]*sizeof(fix16_t), 0);
	if(!recognition_model->model_output_buffer[0]){
		printf("Memory allocation issue with recognition output buffer.\n");
		return -1;
	}
	// Allocate Memory needed for the Recognition Model buffers
	recognition_model->model_io_buffers  = vbx_allocate_dma_buffer(the_vbx_cnn, (1+model_get_num_outputs(recognition_model->model))*sizeof(recognition_model->model_io_buffers[0]), 0);
	if(!recognition_model->model_io_buffers){
		printf("Memory allocation issue with recognition io buffers.\n");
		return -1;
	}
	recognition_model->model_io_buffers[0] = (uintptr_t)recognition_model->model_input_buffer;
	recognition_model->model_io_buffers[1] = (uintptr_t)recognition_model->model_output_buffer[0];

	// Allocate Memory needed for Warp Affine Tranformation
	if (warp_temp_buffer == NULL) warp_temp_buffer = vbx_allocate_dma_buffer(the_vbx_cnn, 224*224*3, 0);


	// Allocate memory for Attribute Model I/Os
	if (has_attribute_model) {
		// Specify the input size for Attribute Model
		struct model_descr_t *attribute_model = models + modelIdx + 2;
		// Allocate the buffer for input for Attribute Model
		attribute_model->model_input_buffer = vbx_allocate_dma_buffer(the_vbx_cnn, model_get_input_length(attribute_model->model, 0)*sizeof(uint8_t), 0);
		if(!attribute_model->model_input_buffer){
			printf("Memory allocation issue with attribute input buffer.\n");
			return -1;
		}
		// Specify the output size for Attribute Model
		attribute_model->model_output_length[0] = model_get_output_length(attribute_model->model, 0); // age output, expecting length 1
		attribute_model->model_output_length[1] = model_get_output_length(attribute_model->model, 1); // gender output, expecting length 2
		// Allocate the buffer for output for Attribute Model
		attribute_model->model_output_buffer[0] = vbx_allocate_dma_buffer(the_vbx_cnn, attribute_model->model_output_length[0]*sizeof(fix16_t), 0);
		attribute_model->model_output_buffer[1] = vbx_allocate_dma_buffer(the_vbx_cnn, attribute_model->model_output_length[1]*sizeof(fix16_t), 0);
		if(!attribute_model->model_output_buffer[0] ||!attribute_model->model_output_buffer[1]){
			printf("Memory allocation issue with attribute output buffers.\n");
			return -1;
		}
		// Allocate Memory needed for the Attribute Model buffers
		attribute_model->model_io_buffers  = vbx_allocate_dma_buffer(the_vbx_cnn, (1+model_get_num_outputs(attribute_model->model))*sizeof(attribute_model->model_io_buffers[0]), 0);
		if(!attribute_model->model_io_buffers ){
			printf("Memory allocation issue with attribute io buffers.\n");
			return -1;
		}
		// I/Os of the attribute model
		attribute_model->model_io_buffers[0] = (uintptr_t)attribute_model->model_input_buffer;
		attribute_model->model_io_buffers[1] = (uintptr_t)attribute_model->model_output_buffer[0];
		attribute_model->model_io_buffers[2] = (uintptr_t)attribute_model->model_output_buffer[1];
	}
	
	return 1;
}


int runRecognitionDemo(struct model_descr_t* models, vbx_cnn_t* the_vbx_cnn, uint8_t modelIdx, int use_attribute_model, int screen_height, int screen_width, int screen_y_offset, int screen_x_offset) {
	int err;
	int screen_stride = 0x2000;
	struct model_descr_t *detect_model = models+modelIdx;
	struct model_descr_t *recognition_model = models+modelIdx+1;
	struct model_descr_t *attribute_model = models+modelIdx+2;
	int colour;
	int use_plate = 0;
	char label[256];
	char gender_char;
	int status;
	uint32_t offset;
	int detectInputH = model_get_input_shape(detect_model->model, 0)[2];
	int detectInputW = model_get_input_shape(detect_model->model, 0)[3];
	//Tracks are initialized if current model has no previous tracks
	if(recognition_model->pTracker == NULL || recognition_model->pTracks == NULL){
		tracksInit(recognition_model);
	}
	// Start processing the network if not already running - 1st pass (frame 0)
	if(!detect_model->is_running) {	
#ifdef HLS_RESIZE
		resize_image_hls(SCALER_BASE_ADDRESS,(uint32_t*)(intptr_t)(*PROCESSING_FRAME_ADDRESS),
				screen_width, screen_height, screen_stride, screen_x_offset, screen_y_offset,
				(uint8_t*)virt_to_phys(the_vbx_cnn, (void*)detect_model->model_input_buffer),
				detectInputW, detectInputH);
#else
		*BLUE_DDR_FRAME_START_ADDR  =  SCALER_FRAME_ADDRESS + (2*detectInputW*detectInputH);
		*GREEN_DDR_FRAME_START_ADDR =  SCALER_FRAME_ADDRESS + (1*detectInputW*detectInputH);
		*RED_DDR_FRAME_START_ADDR   =  SCALER_FRAME_ADDRESS + (0*detectInputW*detectInputH);
		offset = (*PROCESSING_FRAME_ADDRESS) - 0x70000000;
		detect_model->model_input_buffer = (uint8_t*)(uintptr_t)(SCALER_FRAME_ADDRESS + offset);
		detect_model->model_io_buffers[0] = (uintptr_t)detect_model->model_input_buffer - the_vbx_cnn->dma_phys_trans_offset;	
#endif		
		//detect_model->model_io_buffers[0] = (uintptr_t)model_get_test_input(detect_model->model,0);
		// Start Detection model
		err = vbx_cnn_model_start(the_vbx_cnn, detect_model->model, detect_model->model_io_buffers);		
		if(err != 0) return err;
		detect_model->is_running = 1;
	}
	
	status = vbx_cnn_model_poll(the_vbx_cnn); //vbx_cnn_model_wfi(the_vbx_cnn); // Check if model done
	while(status > 0) {
		for(int i =0;i<1000;i++);
		status = vbx_cnn_model_poll(the_vbx_cnn);
	}
	
	if(status < 0) {
		return status;
	} else if (status == 0) { // When  model is completed

		int length=0;
//pdma copy buffers
#if PDMA
	vbx_cnn_io_ptr_t pdma_buffer[model_get_num_outputs(detect_model->model)];
	int output_offset=0;
	for(int o =0; o<(int)model_get_num_outputs(detect_model->model);o++){
		int output_length = model_get_output_length(detect_model->model, o);
		pdma_ch_transfer(pdma_out,(void*)detect_model->pipelined_output_buffers[detect_model->buf_idx][o],output_offset,output_length,the_vbx_cnn,pdma_channel);
		pdma_buffer[o] = (vbx_cnn_io_ptr_t)(pdma_mmap_t + output_offset);
		output_offset+= output_length;
	}
#endif
// Swap pipeline IO
		for (int o = 0; o <  model_get_num_outputs(detect_model->model); o++) {
			detect_model->model_io_buffers[o+1] = (uintptr_t)detect_model->pipelined_output_buffers[!detect_model->buf_idx][o];	
		}	
//
		object_t objects[MAX_TRACKS];
		snprintf(label,sizeof(label),"Face Recognition Demo %dx%d  %d fps",detectInputW,detectInputH,fps);
		if(use_attribute_model)
			snprintf(label,sizeof(label),"Face Recognition + Attribute Demo %dx%d  %d fps",detectInputW,detectInputH,fps);
		if(!strcmp(detect_model->post_process_type, "BLAZEFACE")) {
			// Post Processing BlazeFace output
			int anchor_shift = 1;
			if (detectInputH == 256 && detectInputW == 256) anchor_shift = 0;

			length = post_process_blazeface(objects, detect_model->pipelined_output_buffers[detect_model->buf_idx][0], detect_model->pipelined_output_buffers[detect_model->buf_idx][1],
					detect_model->model_output_length[0], MAX_TRACKS, fix16_from_int(1)>>anchor_shift);

		}
		else if (!strcmp(detect_model->post_process_type, "RETINAFACE")) {
			// Post Processing RetinaFace output
			fix16_t confidence_threshold=F16(0.76);
			fix16_t nms_threshold=F16(0.34);
			// ( 0 1 2 3 4 5 6 7 8) -> (5 4 3 8 7 6 2 1 0)
			fix16_t *outputs[9];

			fix16_t** output_buffers = detect_model->pipelined_output_buffers[detect_model->buf_idx];

			outputs[0]=(fix16_t*)(uintptr_t)output_buffers[5];
			outputs[1]=(fix16_t*)(uintptr_t)output_buffers[4];
			outputs[2]=(fix16_t*)(uintptr_t)output_buffers[3];
			outputs[3]=(fix16_t*)(uintptr_t)output_buffers[8];
			outputs[4]=(fix16_t*)(uintptr_t)output_buffers[7];
			outputs[5]=(fix16_t*)(uintptr_t)output_buffers[6];
			outputs[6]=(fix16_t*)(uintptr_t)output_buffers[2];
			outputs[7]=(fix16_t*)(uintptr_t)output_buffers[1];
			outputs[8]=(fix16_t*)(uintptr_t)output_buffers[0];

			length = post_process_retinaface(objects, MAX_TRACKS, outputs, detectInputW, detectInputH,
					confidence_threshold, nms_threshold);
		}
		else if (!strcmp(detect_model->post_process_type, "SCRFD")) {
			// Post Processing SCRFD output
			fix16_t confidence_threshold=F16(0.8);
			fix16_t nms_threshold=F16(0.34);
			//( 0 1 2 3 4 5 6 7 8)->(2 5 8 1 4 7 0 3 6)
#if PDMA
			fix16_t** output_buffers = (fix16_t**)pdma_buffer;
#else			
			fix16_t** output_buffers = detect_model->pipelined_output_buffers[detect_model->buf_idx];
#endif

			//fix16_t* fix16_buffers[9];
			int8_t* output_buffer_int8[9];
			int zero_points[9];
			fix16_t scale_outs[9];
			for(int o=0; o<model_get_num_outputs(detect_model->model); o++){
				int *output_shape = model_get_output_shape(detect_model->model,o);
				int ind = (output_shape[1]/8)*3 + (2-(output_shape[2]/18)); //first dim should be {2,8,20} second dim should be {9,18,36}
				output_buffer_int8[ind]= (int8_t*)(uintptr_t)output_buffers[o];
				zero_points[ind]=model_get_output_zeropoint(detect_model->model,o);
				scale_outs[ind]=model_get_output_scale_fix16_value(detect_model->model,o);
			}		
			length = post_process_scrfd_int8(objects, MAX_TRACKS, output_buffer_int8, zero_points, scale_outs, detectInputW, detectInputH,
				confidence_threshold,nms_threshold,detect_model->model);

		}
		else if (!strcmp(detect_model->post_process_type, "LPD")) {
			// Post Processing LPD output
			use_plate = 1;
			fix16_t confidence_threshold=F16(0.55);
			fix16_t nms_threshold=F16(0.2);
			int num_outputs = model_get_num_outputs(detect_model->model);
			int8_t* output_buffer_int8[9];
			int zero_points[9];
			fix16_t scale_outs[9];
			fix16_t** output_buffers = detect_model->pipelined_output_buffers[detect_model->buf_idx];
			for (int o = 0; o < num_outputs; o++) {				
				int *output_shape = model_get_output_shape(detect_model->model,o);
				int ind = 2*(output_shape[2]/18) + (output_shape[1]/6); 
				output_buffer_int8[ind]= (int8_t*)(uintptr_t)output_buffers[o];
				zero_points[ind]=model_get_output_zeropoint(detect_model->model,o);
				scale_outs[ind]=model_get_output_scale_fix16_value(detect_model->model,o);
			}

			length = post_process_lpd_int8(objects, MAX_TRACKS, output_buffer_int8, detectInputW, detectInputH,
					confidence_threshold,nms_threshold, num_outputs, zero_points, scale_outs);
							
			snprintf(label,sizeof(label),"Plate Recognition Demo %dx%d  %d fps",detectInputW,detectInputH,fps);
		}
		
		draw_label(label,20,2,overlay_draw_frame,2048,1080,WHITE);

		int tracks[length];
		// If objects are detected			
		if (length > 0) {
			fix16_t confidence;
			char* name="";
			int is_frontal_view ;

			fix16_t x_ratio = fix16_div(screen_width, detectInputW);
			fix16_t y_ratio = fix16_div(screen_height, detectInputH);

			for(int f = 0; f < length; f++) {
				objects[f].box[0] = fix16_mul(objects[f].box[0], x_ratio);
				objects[f].box[1] = fix16_mul(objects[f].box[1], y_ratio);
				objects[f].box[2] = fix16_mul(objects[f].box[2], x_ratio);
				objects[f].box[3] = fix16_mul(objects[f].box[3], y_ratio);
				for(int p = 0; p < 5; p++) {
					objects[f].points[p][0] =fix16_mul(objects[f].points[p][0],x_ratio);
					objects[f].points[p][1] =fix16_mul(objects[f].points[p][1],y_ratio);
				}
			}
			if (add_embedding_mode) {
				object_t* object = &(objects[0]);
				for(int i=0;i<length;i++){
					int new_w = fix16_to_int(objects[i].box[2]) - fix16_to_int(objects[i].box[0]);
					int new_h = fix16_to_int(objects[i].box[3]) - fix16_to_int(objects[i].box[1]);
					if(new_h > (fix16_to_int(object->box[3]) - fix16_to_int(object->box[1])) && (new_w>fix16_to_int(object->box[2]) - fix16_to_int(object->box[0])))
						object = &(objects[i]);				
				}
				recognizeObject(the_vbx_cnn, recognition_model, object, detect_model->post_process_type,
						screen_height, screen_width, screen_stride, screen_y_offset, screen_x_offset);

				// Start Recognition model
				err = vbx_cnn_model_start(the_vbx_cnn, recognition_model->model, recognition_model->model_io_buffers);
				if(err != 0) return err;

				err = vbx_cnn_model_poll(the_vbx_cnn); // Wait for the Recognition model
				while(err > 0) {
					for(int i =0;i<1000;i++);
					err = vbx_cnn_model_poll(the_vbx_cnn);
				}
				if(err < 0) return err;
				fix16_t embedding[128] = {0};
				embedding_calc(embedding,recognition_model);

				int box_thickness=5;
				// Compute the offsets
				int x = fix16_to_int(object->box[0]) + screen_x_offset;
				int y = fix16_to_int(object->box[1]) + screen_y_offset;
				int w = fix16_to_int(object->box[2]) - fix16_to_int(object->box[0]);
				int h = fix16_to_int(object->box[3]) - fix16_to_int(object->box[1]);
				colour = GET_COLOUR(0, 0, 255, 255); //R,G,B,A
				snprintf(label,sizeof(label),"Strong Embedding");
				int frontal_view=facing_front(object);
				if(!frontal_view || (h<112)){
					colour = GET_COLOUR(255,255,0,255);
					snprintf(label,sizeof(label),"Weak Embedding");
				}
				if( x > 0 &&  y > 0 && w > 0 && h > 0) {	
					draw_label(label,x,y+h+screen_y_offset+box_thickness,overlay_draw_frame,2048,1080,WHITE);
					draw_box(x,y,w,h,box_thickness,colour,overlay_draw_frame,2048,1080);
					// Draw the points of detected objects
					for(int p = 0; p < 5; p++) {
						draw_rectangle(fix16_to_int(object->points[p][0])+screen_x_offset,
								fix16_to_int(object->points[p][1])+screen_y_offset, 4, 4, colour,
								overlay_draw_frame, 2048,1080);
					}
				}

				if (capture_embedding) {
					matchEmbedding(embedding,&confidence,&name);
					if(db_end_idx < DB_LENGTH){
						for (int e = 0; e < EMBEDDING_LENGTH; e++) {
							db_embeddings[db_end_idx][e] = (int16_t)embedding[e];
						}
						db_nameStr[db_end_idx] = (char*)malloc(TRACK_NAME_LENGTH);
						sprintf(db_nameStr[db_end_idx], "FACE_%03d", face_count);

						if(fix16_to_int(100*confidence)>40) printf("Warning: Similar embedding already exists (%s)\n\n",name);
						printf("Enter the id of the captured face (default: %s)\n", db_nameStr[db_end_idx]);
						id_check = 1;
						db_end_idx++;
					}
				}
				
			} else {
				// Match detected objects to tracks
				matchTracks(objects, length, recognition_model->pTracker, MAX_TRACKS, recognition_model->pTracks, tracks,use_plate);
				// Run Recognition if there is a tracked object
				if(recognition_model->pTracker->recognitionTrackInd < 0) {
					printf("Obj not tracked\n");
					detect_model->is_running = 0;
					detect_model->buf_idx = !detect_model->buf_idx;
					return 0;
				}
				
				object_t* object = recognition_model->pTracks[recognition_model->pTracker->recognitionTrackInd]->object;
				// Warp the tracked objects							
				recognizeObject(the_vbx_cnn, recognition_model, object, detect_model->post_process_type,
					screen_height, screen_width, screen_stride, screen_y_offset, screen_x_offset);
				if(use_attribute_model){
					// GENDER+AGE ATTRIBUTE 
					// get region within object bbox and see if nose keypoint within region for determining a "frontal view"
					// if frontal view, then perform genderage prediction and adjust track
					is_frontal_view = facing_front(object);
					int bbox_w = fix16_to_int(object->box[2] - object->box[0]);
					int bbox_h = fix16_to_int(object->box[3] - object->box[1]);

					// Resize detected objects to genderage input size
					resize_image_hls(SCALER_BASE_ADDRESS,
						(uint32_t*)(intptr_t)(*PROCESSING_FRAME_ADDRESS), bbox_w, bbox_h, screen_stride, fix16_to_int(object->box[0]), fix16_to_int(object->box[1]),
						(uint8_t*)virt_to_phys(the_vbx_cnn, (void*)attribute_model->model_input_buffer),
						model_get_input_shape(detect_model->model, 0)[3], model_get_input_shape(detect_model->model, 0)[2]);
				}

				// Start Recognition model
				err = vbx_cnn_model_start(the_vbx_cnn, recognition_model->model, recognition_model->model_io_buffers);
				if(err != 0) return err;

				// Update kalman filters
				updateFilters(objects, length, recognition_model->pTracker, recognition_model->pTracks, tracks);

				err = vbx_cnn_model_poll(the_vbx_cnn); // Wait for the Recognition model	
				while(err > 0) {
					for(int i =0;i<1000;i++);
					err = vbx_cnn_model_poll(the_vbx_cnn);
				}
				if(err < 0) return err;


				if(use_attribute_model) {
					// Start attribute model
					vbx_cnn_model_start(the_vbx_cnn, attribute_model->model, attribute_model->model_io_buffers);
				}

			
				if (!strcmp(recognition_model->post_process_type, "LPR")) {
					char tmp[20]="";
					name = (char*)tmp;
					int8_t* output_buffer_int8 = (int8_t*)(uintptr_t)recognition_model->model_output_buffer[0];
					confidence = post_process_lpr_int8(output_buffer_int8, recognition_model->model, name);
				} else {
					// normalize the recognition output embedding
					
					fix16_t embedding[128]={0};
					embedding_calc(embedding,recognition_model);						
					// Match the recognized objects with the ones in the database
					matchEmbedding(embedding,&confidence,&name);
				}
				// Filter recognition output
				updateRecognition(recognition_model->pTracks, recognition_model->pTracker->recognitionTrackInd, confidence, name, recognition_model->pTracker);
				if(use_attribute_model) {
					err = vbx_cnn_model_poll(the_vbx_cnn); // Wait for the attribute model
					while(err > 0) {
						for(int i =0;i<1000;i++);
						err = vbx_cnn_model_poll(the_vbx_cnn);
					}
					if(err < 0) return err;
					// Update gender+age of object tracks
					fix16_t age = 100*attribute_model->model_output_buffer[0][0];
					fix16_t gender = attribute_model->model_output_buffer[1][0];
					updateAttribution(recognition_model->pTracks[recognition_model->pTracker->recognitionTrackInd],gender,age,recognition_model->pTracker, is_frontal_view);
				}

				// Draw boxes for detected and label the recognized
				for(int t = 0; t < recognition_model->pTracker->tracksLength; t++) {
					track_t* track = recognition_model->pTracks[t];
					if(track->object == NULL)
						continue;
					fix16_t box[4];
					int box_thickness=5;
					// Calculate the box coordinates of the tracked object
					boxCoordinates(box, &track->filter);
					// Compute the offsets
					int x = fix16_to_int(box[0]) + screen_x_offset;
					int y = fix16_to_int(box[1]) + screen_y_offset;
					int w = fix16_to_int(box[2]) - fix16_to_int(box[0]);
					int h = fix16_to_int(box[3]) - fix16_to_int(box[1]);
					// Adding labels to the recognized
					if(strlen(track->name) > 0 && fix16_to_int(100*track->similarity)>40) {
						label_colour_e text_color= GREEN;
						if(fix16_to_int(100*track->similarity)>60){					
							colour = GET_COLOUR(0, 250, 0, 255);
							text_color = GREEN;
						}
						else{
							colour = GET_COLOUR(250, 250, 0, 255);
							text_color = WHITE;
						}
						if (!strcmp(recognition_model->post_process_type, "LPR")) {
							snprintf(label,sizeof(label),"%s", track->name);
						} else {
							snprintf(label,sizeof(label),"%s  (%d) ", track->name, fix16_to_int(100*track->similarity));
						}
						
						draw_label(label,x,fix16_to_int(box[3])+screen_y_offset+box_thickness,overlay_draw_frame,2048,1080,text_color);
						if(use_attribute_model){
							if(track->gender > F16(0.2)){
								gender_char = 'F';
							} else if(track->gender < F16(-0.6)){
								gender_char = 'M';
							} else{
								gender_char = '?';
							}
							snprintf(label,sizeof(label),"%c %d", gender_char, fix16_to_int(track->age));
							draw_label(label,x,fix16_to_int(box[1])-32,overlay_draw_frame,2048,1080,text_color);
						}
					} else {
						colour = GET_COLOUR(250, 250, 250,255);
						if(use_attribute_model){
							if(track->gender > F16(0.2)){
								gender_char = 'F';
							} else if(track->gender < F16(-0.6)){
								gender_char = 'M';
							} else{
								gender_char = '?';
							}
							snprintf(label,sizeof(label),"%c %d", gender_char, fix16_to_int(track->age));
							draw_label(label,x,fix16_to_int(box[1])-32,overlay_draw_frame,2048,1080,WHITE);
						}
					}

					if( x > 0 &&  y > 0 && w > 0 && h > 0) {
						draw_box(x,y,w,h,box_thickness,colour,overlay_draw_frame,2048,1080);
						if (!strcmp(recognition_model->post_process_type, "LPR")) {
						} else {
							// Draw the points of detected objects
							for(int p = 0; p < 5; p++) {
								draw_rectangle(fix16_to_int(track->object->points[p][0])+screen_x_offset,
										fix16_to_int(track->object->points[p][1])+screen_y_offset, 4, 4, colour,
										overlay_draw_frame, 2048,1080);
							}
						}
					}
				}
			}
		} else {
			matchTracks(objects, length, recognition_model->pTracker, MAX_TRACKS, recognition_model->pTracks, tracks,use_plate);
			if(capture_embedding){
				printf("No valid face embeddings captured\n");
				capture_embedding = 0;
			}
		}
		detect_model->is_running = 0;
		detect_model->buf_idx = !detect_model->buf_idx;
	}
	return 0;
}

void matchEmbedding(fix16_t embedding[],fix16_t* similarity, char** name) {
	// Match the detected objects with the objects in database
	*similarity = fix16_minimum;
	for(int d = 0; d < db_end_idx; d++){
		fix16_t dotProd = 0;
		for(int n = 0; n < EMBEDDING_LENGTH; n++)
			dotProd += (embedding[n] * db_embeddings[d][n])>>16;
		if(dotProd > *similarity){
			*similarity = dotProd;
			*name = db_nameStr[d];
		}
	}
}

void recognizeObject(vbx_cnn_t* the_vbx_cnn, struct model_descr_t* model, object_t* object, const char* post_process_type, int screen_height, int screen_width, int screen_stride, int screen_y_offset, int screen_x_offset) {
	fix16_t xy[6], ref[6];

	if(!strcmp(post_process_type,"LPD")) {
		xy[0] = object->box[0];
		xy[1] = object->box[1];
		xy[2] = object->box[2];
		xy[3] = object->box[1];
		xy[4] = object->box[0];
		xy[5] = object->box[3];
	} else if(!strcmp(post_process_type,"RETINAFACE") || !strcmp(post_process_type, "SCRFD")) {
		xy[0] = object->points[0][0];
		xy[1] = object->points[0][1];
		xy[2] = object->points[1][0];
		xy[3] = object->points[1][1];
		// Mean of mouth points of Retinaface
		xy[4] = (object->points[3][0] + object->points[4][0])/2;
		xy[5] = (object->points[3][1] + object->points[4][1])/2;
	} else{
		xy[0] = object->points[0][0];
		xy[1] = object->points[0][1];
		xy[2] = object->points[1][0];
		xy[3] = object->points[1][1];
		xy[4] = object->points[3][0];
		xy[5] = object->points[3][1];
	}
	if (screen_x_offset > 0) {
		xy[0] += fix16_from_int(screen_x_offset);
		xy[2] += fix16_from_int(screen_x_offset);
		xy[4] += fix16_from_int(screen_x_offset);
	}
	if (screen_y_offset > 0) {
		xy[1] += fix16_from_int(screen_y_offset);
		xy[3] += fix16_from_int(screen_y_offset);
		xy[5] += fix16_from_int(screen_y_offset);
	}
	// Model reference coordinates
	ref[0] = model->coord4[0];
	ref[1] = model->coord4[1];
	ref[2] = model->coord4[2];
	ref[3] = model->coord4[3];
	ref[4] = model->coord4[6];
	ref[5] = model->coord4[7];
	warp_image_with_points(SCALER_BASE_ADDRESS,
		WARP_BASE_ADDRESS,
		(uint32_t*)(intptr_t)(*PROCESSING_FRAME_ADDRESS),
		(uint8_t*)virt_to_phys(the_vbx_cnn, (void*)model->model_input_buffer),
		(uint8_t*)virt_to_phys(the_vbx_cnn, (void*)warp_temp_buffer),
		xy, ref,
		screen_width, screen_height, screen_stride,
		model_get_input_shape(model->model, 0)[3],model_get_input_shape(model->model, 0)[2]);
}
