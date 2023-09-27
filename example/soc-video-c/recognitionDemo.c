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
static inline void* virt_to_phys(vbx_cnn_t* vbx_cnn,void* virt){
  return (char*)(virt) + vbx_cnn->dma_phys_trans_offset;
}
extern int fps;
extern uint32_t* loop_draw_frame;

// Globals Specification
#if VBX_SOC_DRIVER
	#define MAX_TRACKS 48
	#define DB_LENGTH 32
	extern int delete_embedding_mode;
	extern int add_embedding_mode;
	extern int capture_embedding;
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
{7885,3501,-860,-1888,4510,4596,-2735,4597,8906,566,10390,-3417,-3020,10061,3402,11746,-6467,1488,-1247,-1394,745,-1244,-3412,-186,-2223,-1019,-6732,2135,9692,-4734,1015,-5168,7147,407,-4256,7437,9844,267,-2570,-2622,9554,-268,7854,-4912,5446,767,7180,5580,7675,-851,-1363,5377,466,8983,-6302,11044,2922,-1063,-4158,12,8243,3252,1358,-5579,4307,-481,2134,-4501,8222,-481,-8227,8569,5734,-941,3820,-1084,4764,-755,6349,690,-1357,-1005,-1257,-14082,7217,13145,-3583,-3986,3114,2243,-5409,1083,16126,640,-3178,10545,-2815,3805,-6053,-1494,-7010,2572,-6821,7532,7836,-6884,-5788,-4347,2803,-13064,-5841,-3991,-1987,3062,10889,3079,5809,-568,8409,10218,-223,766,387,-6324,5643,5620,4686,-9639,},
{8128,9650,8024,10271,595,-3457,-5450,-7820,-3846,1089,-16355,-67,-8641,2434,-971,4955,-3036,-2068,-4194,4212,-434,-259,-3465,4126,-5702,-2921,-7690,9565,1708,-3749,-4619,-476,-2376,7805,-8085,-5222,-626,6996,-5681,-1212,2876,-2932,-1059,3919,384,-5252,-3305,-9207,3960,-1739,182,9293,10323,-11492,-670,8075,-4298,569,10264,13077,98,6445,3196,-6998,-1669,7045,-3396,-5498,-2800,-3163,-4438,-3083,10062,881,7607,2185,5061,-1472,-580,1891,-1941,-8115,-3540,593,3858,-3430,9248,-1354,7588,876,2468,1166,8317,1942,-1000,11178,-6242,4328,-5953,-2209,6329,-2059,1391,-4689,-5046,-2839,2539,-4417,301,-8322,-1498,-1944,-1974,-4408,1138,-275,15111,10642,-9381,7236,665,-7366,9163,-3209,2183,13691,-2680,2390,},
{-3082,3475,-2912,-6377,-4705,5972,-2953,1291,-2323,-4631,5680,8635,-4564,-4285,4307,-5858,9624,6597,3177,8171,-973,10949,-3601,2462,-6471,8368,1035,312,6164,-3,-3753,-1448,9905,8293,-720,4239,-1545,-4046,-3882,-275,-4305,-2749,4000,6783,-977,-57,4794,-3550,-1477,2001,-3131,5629,-1260,-1585,-5175,-3941,3578,12496,3009,-5725,1994,9204,-1237,-731,-3601,-7838,-3969,679,-5528,-9455,-3126,2954,2273,438,613,-760,-1669,-4552,4035,-4743,-7296,10410,-974,-1590,-14714,-19516,1499,-10311,5630,6577,-11681,-4432,6828,-2194,8479,-3151,9892,3171,-2328,496,-1996,-3947,11098,-8156,-5079,6356,4486,-2183,6050,7414,3462,12720,742,2287,9559,10608,5742,5703,-1399,1044,-6428,-6192,5150,2672,2796,-5876,1158,-4817,},
{-4214,-912,-3213,-3425,6701,-6203,-13317,-5947,9098,-9949,5140,-4410,-13009,-881,-977,6725,11932,2380,5463,1974,-906,2796,-10204,-361,-5977,-2043,-2719,-6203,-4111,-4939,-3553,-15949,-5108,-10438,14381,-3592,4347,3015,-10308,-2142,6282,4222,4834,-650,-5860,3923,4360,-6488,4922,2955,-4819,7810,-9800,1223,-3533,351,-7534,7538,-1181,312,5479,-3857,-10134,-236,3156,-6432,-5089,182,4317,-6388,-7935,8513,983,-5515,-240,-3909,-860,7179,7254,-6626,3281,4025,8764,2630,-7627,-3646,-1825,-7864,8979,-5086,1631,5719,-398,9897,-2002,258,-772,-1966,-7602,163,-58,6132,-3697,-7671,-7379,2740,-261,-1442,1003,1297,-4014,1760,4949,-1843,4602,5459,-4744,6584,-5246,5474,-1328,8447,-6693,2299,4020,-4995,-6726,-4583,},
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


short recognitionDemoInit(vbx_cnn_t* the_vbx_cnn, struct model_descr_t* models, uint8_t modelIdx, int has_attribute_model, int screen_height, int screen_width, int screen_y_offset, int screen_x_offset) {
	struct model_descr_t *detect_model = models + modelIdx;
	// Allocate memory for Models

	// Allocate the buffer for input for Detect Model


	// Allocate Memory needed for the Detect Model buffers
	detect_model->model_io_buffers  = vbx_allocate_dma_buffer(the_vbx_cnn, (1+model_get_num_outputs(detect_model->model))*sizeof(detect_model->model_io_buffers[0]), 0);
	if(!detect_model->model_io_buffers){
		printf("Memory allocation issue for model io buffers.\n");
		return -1;
	}
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



	detect_model->pipelined_input_buffer[0] = vbx_allocate_dma_buffer(the_vbx_cnn, model_get_input_length(detect_model->model, 0)*sizeof(uint8_t), 0);
	detect_model->pipelined_input_buffer[1] = vbx_allocate_dma_buffer(the_vbx_cnn, model_get_input_length(detect_model->model, 0)*sizeof(uint8_t), 0);
	detect_model->model_io_buffers[0] = (uintptr_t)detect_model->pipelined_input_buffer[0];
	if(!detect_model->pipelined_input_buffer[0] ||!detect_model->pipelined_input_buffer[1]){
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
	//Tracks are initialized if current model has no previous tracks
	if(recognition_model->pTracker == NULL || recognition_model->pTracks == NULL){
		tracksInit(recognition_model);
	}
	// Start processing the network if not already running - 1st pass (frame 0)
	if(!detect_model->is_running) {
		// Start Scaling the frame
		resize_image_hls(SCALER_BASE_ADDRESS,(uint32_t*)(intptr_t)(*PROCESSING_FRAME_ADDRESS),
				screen_width, screen_height, screen_stride, screen_x_offset, screen_y_offset,
				(uint8_t*)virt_to_phys(the_vbx_cnn, (void*)detect_model->pipelined_input_buffer[detect_model->buf_idx]),
				model_get_input_dims(detect_model->model, 0)[2], model_get_input_dims(detect_model->model, 0)[1]);

		// Start Detection model
		err = vbx_cnn_model_start(the_vbx_cnn, detect_model->model, detect_model->model_io_buffers);
		resize_image_hls_start(SCALER_BASE_ADDRESS,(uint32_t*)(intptr_t)(*PROCESSING_NEXT_FRAME_ADDRESS),
				screen_width, screen_height, screen_stride, screen_x_offset, screen_y_offset,
				(uint8_t*)virt_to_phys(the_vbx_cnn, (void*)detect_model->pipelined_input_buffer[!detect_model->buf_idx]),
				model_get_input_dims(detect_model->model, 0)[2], model_get_input_dims(detect_model->model, 0)[1]);

		detect_model->is_running = 1;
		if(err != 0) return err;
		err = vbx_cnn_model_poll(the_vbx_cnn);
		// Poll for next detection to be done
		while(err > 0) {
			for(int i =0;i<1000;i++);
			err = vbx_cnn_model_poll(the_vbx_cnn);
		}
	}

	
	
	if (detect_model->is_running){ // Assuming demo is running

		int length;
		int detectInputH = model_get_input_dims(detect_model->model, 0)[1];
		int detectInputW = model_get_input_dims(detect_model->model, 0)[2];

		// Swap which set of pipelined buffers is used as model IO
		detect_model->model_io_buffers[0] = (uintptr_t)detect_model->pipelined_input_buffer[!detect_model->buf_idx];
		for (int o = 0; o <  model_get_num_outputs(detect_model->model); o++) {
			detect_model->model_io_buffers[o+1] = (uintptr_t)detect_model->pipelined_output_buffers[!detect_model->buf_idx][o];
		}

		// Start Detection model on next frame
		
		err = vbx_cnn_model_start(the_vbx_cnn, detect_model->model, detect_model->model_io_buffers);
		if(err != 0) return err;
		
		//Wait for next frame scaling to finish
		resize_image_hls_wait(SCALER_BASE_ADDRESS);

		// Start Scaling the next frame for detection
		resize_image_hls_start(SCALER_BASE_ADDRESS,(uint32_t*)(intptr_t)(*PROCESSING_NEXT2_FRAME_ADDRESS),
					screen_width, screen_height, screen_stride, screen_x_offset, screen_y_offset,
					(uint8_t*)virt_to_phys(the_vbx_cnn, (void*)detect_model->pipelined_input_buffer[detect_model->buf_idx]),
					model_get_input_dims(detect_model->model, 0)[2], model_get_input_dims(detect_model->model, 0)[1]);	

		
		object_t objects[MAX_TRACKS];
		snprintf(label,sizeof(label),"Running Face Recognition Demo %d fps",fps);
		if(use_attribute_model)
			snprintf(label,sizeof(label),"Running Face Recognition + Attribute Demo %d fps",fps);
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
			fix16_t confidence_threshold=F16(0.0);
			fix16_t nms_threshold=F16(0.34);
			//( 0 1 2 3 4 5 6 7 8)->(2 5 8 1 4 7 0 3 6)
			fix16_t *outputs[9];
			fix16_t** output_buffers = detect_model->pipelined_output_buffers[detect_model->buf_idx];
			outputs[0]=(fix16_t*)(uintptr_t)output_buffers[2];
			outputs[1]=(fix16_t*)(uintptr_t)output_buffers[5];
			outputs[2]=(fix16_t*)(uintptr_t)output_buffers[8];
			outputs[3]=(fix16_t*)(uintptr_t)output_buffers[1];
			outputs[4]=(fix16_t*)(uintptr_t)output_buffers[4];
			outputs[5]=(fix16_t*)(uintptr_t)output_buffers[7];
			outputs[6]=(fix16_t*)(uintptr_t)output_buffers[0];
			outputs[7]=(fix16_t*)(uintptr_t)output_buffers[3];
			outputs[8]=(fix16_t*)(uintptr_t)output_buffers[6];


			length = post_process_scrfd(objects, MAX_TRACKS, outputs, detectInputW, detectInputH,
					confidence_threshold, nms_threshold);
		}
		else if (!strcmp(detect_model->post_process_type, "LPD")) {
			// Post Processing SCRFD output
			use_plate = 1;
			fix16_t confidence_threshold=F16(0.55);
			fix16_t nms_threshold=F16(0.2);
			int num_outputs = model_get_num_outputs(detect_model->model);
			length = post_process_lpd(objects, MAX_TRACKS, detect_model->pipelined_output_buffers[detect_model->buf_idx], detectInputW, detectInputH,
					confidence_threshold, nms_threshold, num_outputs);
			snprintf(label,sizeof(label),"Running Plate Recognition Demo %d fps",fps);
		}

		draw_label(label,20,2,loop_draw_frame,2048,1080,WHITE);

		int tracks[length];
		// If objects are detected
		if (length > 0) {
			fix16_t confidence;
			char* name;
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

				// Poll for next detection to be done
				err = vbx_cnn_model_poll(the_vbx_cnn);
				while(err > 0) {
					for(int i =0;i<1000;i++);
					err = vbx_cnn_model_poll(the_vbx_cnn);
				}
				if(err != 0) return err;


				// Start Recognition model
				err = vbx_cnn_model_start(the_vbx_cnn, recognition_model->model, recognition_model->model_io_buffers);
				if(err != 0) return err;


				err = vbx_cnn_model_poll(the_vbx_cnn); // Wait for the Recognition model
				while(err > 0) {
					for(int i =0;i<1000;i++);
					err = vbx_cnn_model_poll(the_vbx_cnn);
				}
				if(err < 0) return err;

				fix16_t sum = 0;
				fix16_t embedding[128];
				for(int n = 0; n < recognition_model->model_output_length[0]; n++)
					sum += fix16_sq(recognition_model->model_output_buffer[0][n]);
				fix16_t norm = fix16_div(fix16_one, fix16_sqrt(sum));
				for(int n = 0; n < recognition_model->model_output_length[0]; n++)
					embedding[n] = fix16_mul(recognition_model->model_output_buffer[0][n], norm);

				int box_thickness=5;
				// Compute the offsets
				int x = fix16_to_int(object->box[0]) + screen_x_offset;
				int y = fix16_to_int(object->box[1]) + screen_y_offset;
				int w = fix16_to_int(object->box[2]) - fix16_to_int(object->box[0]);
				int h = fix16_to_int(object->box[3]) - fix16_to_int(object->box[1]);
				colour = GET_COLOUR(255, 0, 0, 255);
				if( x > 0 &&  y > 0 && w > 0 && h > 0) {
					draw_box(x,y,w,h,box_thickness,colour,loop_draw_frame,2048,1080);
					// Draw the points of detected objects
					for(int p = 0; p < 5; p++) {
						draw_rectangle(fix16_to_int(object->points[p][0])+screen_x_offset,
								fix16_to_int(object->points[p][1])+screen_y_offset, 4, 4, colour,
								loop_draw_frame, 2048,1080);
					}
				}


				if (capture_embedding) {
					matchEmbedding(embedding,&confidence,&name);

					if(db_end_idx < DB_LENGTH){
						for (int e = 0; e < EMBEDDING_LENGTH; e++) {
							db_embeddings[db_end_idx][e] = (int16_t)embedding[e];
						}
						db_nameStr[db_end_idx] = (char*)malloc(16);
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
				err = vbx_cnn_model_poll(the_vbx_cnn);
				// Poll for next detection to be done
				while(err > 0) {
					for(int i =0;i<1000;i++);
					err = vbx_cnn_model_poll(the_vbx_cnn);
				}
				detect_model->is_running = 1;
				detect_model->buf_idx = !detect_model->buf_idx;
				return 0;
			}
			object_t* object = recognition_model->pTracks[recognition_model->pTracker->recognitionTrackInd]->object;

			// Warp the tracked objects
			recognizeObject(the_vbx_cnn, recognition_model, object, detect_model->post_process_type,
					screen_height, screen_width, screen_stride, screen_y_offset, screen_x_offset);

			if(use_attribute_model){
				/* GENDER+AGE ATTRIBUTE */
				// get region within object bbox and see if nose keypoint within region for determining a "frontal view"
				// if frontal view, then perform genderage prediction and adjust track
				int bbox_w = fix16_to_int(object->box[2] - object->box[0]);
				int bbox_h = fix16_to_int(object->box[3] - object->box[1]);
				int center_x = fix16_to_int(object->box[2] + object->box[0]) / 2;
				int center_y = fix16_to_int(object->box[3] + object->box[1]) / 2;
				int bbox_w_frontal = bbox_w * 0.5;
				int bbox_h_frontal = bbox_h * 0.5;
				int frontal_bbox_left =     center_x - (bbox_w_frontal/2);
				int frontal_bbox_right =    center_x + (bbox_w_frontal/2);
				int frontal_bbox_top =      center_y - (bbox_h_frontal/2);
				int frontal_bbox_bottom =   center_y + (bbox_h_frontal/2);
				int nose_x = fix16_to_int(object->points[2][0]);
				int nose_y = fix16_to_int(object->points[2][1]);

				if(nose_x < frontal_bbox_left || nose_x > frontal_bbox_right || nose_y < frontal_bbox_top || nose_y > frontal_bbox_bottom){
					is_frontal_view = 0;
				}
				else{
					is_frontal_view = 1;
				}

				// Resize detected objects to genderage input size
				resize_image_hls(SCALER_BASE_ADDRESS,
						(uint32_t*)(intptr_t)(*PROCESSING_FRAME_ADDRESS), bbox_w, bbox_h, screen_stride, fix16_to_int(object->box[0]), fix16_to_int(object->box[1]),
						(uint8_t*)virt_to_phys(the_vbx_cnn, (void*)attribute_model->model_input_buffer),
						model_get_input_dims(attribute_model->model, 0)[2], model_get_input_dims(attribute_model->model, 0)[1]);

			}

			err = vbx_cnn_model_poll(the_vbx_cnn);
			// Poll for next detection to be done
			while(err > 0) {
				for(int i =0;i<1000;i++);
				err = vbx_cnn_model_poll(the_vbx_cnn);
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


			if(use_attribute_model){
				// Start attribute model
				vbx_cnn_model_start(the_vbx_cnn, attribute_model->model, attribute_model->model_io_buffers);


			}



			
			if (!strcmp(recognition_model->post_process_type, "LPR")) {
				char tmp[20];
				name = (char*)tmp;
				confidence = post_process_lpr(recognition_model->model_output_buffer[0], recognition_model->model_output_length[0], name);
			} else {
				// normalize the recognition output embedding
				fix16_t sum = 0;
				fix16_t embedding[128];
				for(int n = 0; n < recognition_model->model_output_length[0]; n++)
					sum += fix16_sq(recognition_model->model_output_buffer[0][n]);
				fix16_t norm = fix16_div(fix16_one, fix16_sqrt(sum));
				for(int n = 0; n < recognition_model->model_output_length[0]; n++)
					embedding[n] = fix16_mul(recognition_model->model_output_buffer[0][n], norm);

				// Match the recognized objects with the ones in the database
				matchEmbedding(embedding,&confidence,&name);
			}

			// Filter recognition output
			updateRecognition(recognition_model->pTracks, recognition_model->pTracker->recognitionTrackInd, confidence, name, recognition_model->pTracker);

			if(use_attribute_model){

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
				if(strlen(track->name) > 0) {
					colour = GET_COLOUR(0, 250, 0, 255);
					if (!strcmp(recognition_model->post_process_type, "LPR")) {
						snprintf(label,sizeof(label),"%s", track->name);
					} else {
						snprintf(label,sizeof(label),"%s  (%d%%)", track->name, fix16_to_int(100*track->similarity));
					}
					draw_label(label,x,fix16_to_int(box[3])+screen_y_offset+box_thickness,loop_draw_frame,2048,1080,GREEN);
					if(use_attribute_model){
						if(track->gender > F16(0.2)){
							gender_char = 'F';
						} else if(track->gender < F16(-0.6)){
							gender_char = 'M';
						} else{
							gender_char = '?';
						}
						snprintf(label,sizeof(label),"%c %d", gender_char, fix16_to_int(track->age));
						draw_label(label,x,fix16_to_int(box[1])-32,loop_draw_frame,2048,1080,GREEN);
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
						draw_label(label,x,fix16_to_int(box[1])-32,loop_draw_frame,2048,1080,WHITE);
					}
				}

				if( x > 0 &&  y > 0 && w > 0 && h > 0) {
					draw_box(x,y,w,h,box_thickness,colour,loop_draw_frame,2048,1080);
					if (!strcmp(recognition_model->post_process_type, "LPR")) {
					} else {
						// Draw the points of detected objects
						for(int p = 0; p < 5; p++) {
							draw_rectangle(fix16_to_int(track->object->points[p][0])+screen_x_offset,
									fix16_to_int(track->object->points[p][1])+screen_y_offset, 4, 4, colour,
									loop_draw_frame, 2048,1080);
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
			err = vbx_cnn_model_poll(the_vbx_cnn);
			// Poll for next detection to be done
			while(err > 0) {
				for(int i =0;i<1000;i++);
				err = vbx_cnn_model_poll(the_vbx_cnn);
			}


		}
		detect_model->is_running = 1;
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
			model_get_input_dims(model->model, 0)[2], model_get_input_dims(model->model, 0)[1]);
}
