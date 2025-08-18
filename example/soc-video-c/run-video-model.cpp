#include "postprocess.h"
#include "vbx_cnn_api.h"
#include <stdio.h>
#include <string.h>
#include <string>
#include <sys/time.h>
#include <inttypes.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include "pdma/pdma_helpers.h"
#include <cassert>

#include "frameDrawing/draw_assist.h"
#include "frameDrawing/draw.h"
#include "imageScaler/scaler.h"
#include "warpAffine/warp.h"
#include "recognitionDemo.h"
#include "detectionDemo.h"


struct model_descr_t models[] = {
		{"Yolo v8n", "/home/root/samples_V1000_2.0.3/yolov8n_512x288_argmax.vnnx", 0, "ULTRALYTICS"},		
		{"SCRFD", "/home/root/samples_V1000_2.0.3/scrfd_500m_bnkps.vnnx", 0, "SCRFD"},
		{"mobileface-arcface", "/home/root/samples_V1000_2.0.3/mobilefacenet-arcface.vnnx", 0, "ARCFACE"},
		{"MobileNet V2", "/home/root/samples_V1000_2.0.3/mobilenet-v2.vnnx", 0, "CLASSIFY"},	
		{"Yolov8 Pose", "/home/root/samples_V1000_2.0.3/yolov8n-pose_512x288_split.vnnx", 0, "ULTRALYTICS_POSE"},	
		{"FFNet 122NS", "/home/root/samples_V1000_2.0.3/FFNet-122NS-LowRes_512x288.vnnx", 0, "PIXEL"},	
		{"Midas V2", "/home/root/samples_V1000_2.0.3/Midas-V2-Quantized.vnnx", 0, "PIXEL"},
};

#define UIO_DMA_LIMIT 512*1024*1024*2
int total_fsize=0;
short demo_setup = 0;
int use_attribute_model = 0;
int fps = 0;
int add_embedding_mode = 0;
int capture_embedding = 0;
int delete_embedding_mode=0;
char id_entered[128];

#ifdef HLS_RESIZE
    int set_screen_width = 1920;//1920;
    int set_screen_height = 1080;//1080;
    int set_screen_y_offset = 0;
    int set_screen_x_offset = 0;
    int set_screen_stride = 0x2000;
#endif

void enable_interrupt(vbx_cnn_t *vbx_cnn){
	uint32_t reenable = 1;
	ssize_t writeSize = write(vbx_cnn->fd, &reenable, sizeof(uint32_t));
	if(writeSize < 0) {
		close(vbx_cnn->fd);
	}
};
static inline void* virt_to_phys(vbx_cnn_t* vbx_cnn,void* virt){
	return (char*)(virt) + vbx_cnn->dma_phys_trans_offset;
}
#if PDMA
int8_t* pdma_mmap_t;
uint64_t pdma_out;
int32_t pdma_channel;
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
#endif
static int swap_model(int model_idx) {
	model_idx = (model_idx + 1) % (int)(sizeof(models)/sizeof(*models));
	
	while((!strcmp(models[model_idx].post_process_type, "ARCFACE") || !strcmp(models[model_idx].post_process_type, "GENDERAGE") || 
		!strcmp(models[model_idx].post_process_type, "LPR"))) {
		
		model_idx = (model_idx + 1) % (int)(sizeof(models)/sizeof(*models));
	}
	models[model_idx].is_running = 0;
	if (!strcmp(models[model_idx].post_process_type, "ARCFACE") && (use_attribute_model == 1)) {
		model_idx -=1;
		use_attribute_model =0;
		printf("Face Recognition + GenderAge\n");
	} else if ((!strcmp(models[model_idx].post_process_type, "RETINAFACE") || 
				!strcmp(models[model_idx].post_process_type, "SCRFD"))) {
		
		if(use_attribute_model == 1) {
			use_attribute_model = 0;
		}
		printf("Face Recognition\n");
	} else if (!strcmp(models[model_idx].post_process_type, "LPD")) {
		trackClean(models,model_idx+1);
		printf("Plate Recognition\n");
	} else {
		printf("%s\n", models[model_idx].name);
	}
	return model_idx;
}

void *read_ascii_file(vbx_cnn_t *vbx_cnn, const char *filename) {
	FILE *fd = fopen(filename, "r");
	if (fd == NULL) {
		return NULL;
	}
	fseek(fd, 0, SEEK_END);
	int file_size = ftell(fd);
	fseek(fd, 0, SEEK_SET);
	void *ascii = malloc(file_size);
	int size_read = fread(ascii, 1, file_size, fd);
	if (size_read != file_size) {
		fprintf(stderr, "Error reading full file %s\n", filename);
		return NULL;
	}
	void *dma_ascii = vbx_allocate_dma_buffer(vbx_cnn, file_size, 0);
	if (dma_ascii) {
		memcpy(dma_ascii, ascii, file_size);
	}
	free(ascii);
	return dma_ascii;
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
	int size_read = fread(model, 1, file_size, model_file);
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
	total_fsize += model_allocate_size;
	if(total_fsize > UIO_DMA_LIMIT){ //hardcoded for now, will update
		printf("%s exceeds max model allocate size limit by %d MB\nExiting...\n",filename,(total_fsize-UIO_DMA_LIMIT)/(1024*1024));
		exit(1);
	}
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


static uint64_t u64_from_attribute(const char* filename){
	FILE* fd = fopen(filename,"r");
	uint64_t ret;
	fscanf(fd,"0x%" PRIx64,&ret);
	fclose(fd);
	return ret;
}


static int find_uio_dev_num_by_addr(void *target_addr){
	char buf[4096];
	for(int n=0; ; n++) {
		snprintf(buf,4096,"/sys/class/uio/uio%d/name",n);
		int name_fd = open(buf,O_RDONLY);
		if (name_fd >= 0) {
			read(name_fd,buf,4096);
			close(name_fd);
			//if target_addr is not NULL, make sure that matches as well.
			snprintf(buf,4096,"/sys/class/uio/uio%d/maps/map0/addr",n);
			uintptr_t addr =  u64_from_attribute(buf);
			if(target_addr && (uintptr_t)target_addr != addr){
				//name matched but address didn't, skip this device.
				continue;
			}
			return n;
		} else {
			break;
		}
	}
	return -1;
}

static void* uio_mmap(int fd, int dev_num,int map_num){
    char filename[64];
    snprintf(filename,sizeof(filename),"/sys/class/uio/uio%d/maps/map%d/size",dev_num,map_num);
    int64_t size=u64_from_attribute(filename);
    void* _ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED,
                    fd, map_num * sysconf(_SC_PAGESIZE));
    return _ptr;
}


static void* uio_mmap_from_addr(void* target_addr){
    int dev_num =  find_uio_dev_num_by_addr(target_addr);
    if (dev_num != -1) {
        char filename[64];
        snprintf(filename,sizeof(filename),"/dev/uio%d",dev_num);
        int fd = open(filename, O_RDWR);
        if (fd>=0){
            return uio_mmap(fd, dev_num,0);
        }
    }
    return NULL;
}

#define FRAME_BASE          0x40001000
#define SCALE_BASE          0x60040000
#define WARP_BASE           0x60050000
#define DRAW_BASE           0x60060000
#define	OVERLAY_FRAME_ADDR  0x78000000

volatile uint32_t* SCALER_BASE_ADDRESS;
volatile uint32_t* WARP_BASE_ADDRESS;
volatile uint32_t* PROCESSING_FRAME_ADDRESS;
volatile uint32_t* PROCESSING_NEXT_FRAME_ADDRESS;
volatile uint32_t* PROCESSING_NEXT2_FRAME_ADDRESS;
volatile uint32_t* SAVED_FRAME_SWAP;
volatile uint32_t* HORZ_RES_IN_ADDR;
volatile uint32_t* VERT_RES_IN_ADDR;
volatile uint32_t* HORZ_RES_OUT_ADDR;
volatile uint32_t* VERT_RES_OUT_ADDR;
volatile uint32_t* SCALE_FACTOR_HORZ_ADDR;
volatile uint32_t* SCALE_FACTOR_VERT_ADDR;
volatile uint32_t* LINE_GAP_ADDR;
volatile uint32_t* RED_DDR_FRAME_START_ADDR;
volatile uint32_t* GREEN_DDR_FRAME_START_ADDR;
volatile uint32_t* BLUE_DDR_FRAME_START_ADDR;

volatile uint32_t* ALPHA_BLEND_EN_ADDR;
volatile uint32_t* FB_FRAME_START_ADDR;
volatile uint32_t* FRAME_BLANKER_EN_ADDR;
volatile uint32_t* FRAME_BLANKER_DONE_ADDR;
volatile uint32_t* OVERLAY_DRAW_ADDR;
volatile uint32_t* OVERLAY_DISPLAY_ADDR;
volatile uint32_t* OVERLAY_BLANK_ADDR;
volatile uint32_t* MIN_LATENCY_SEL_ADDR;

volatile void* draw_assist_base_address;
void* ascii_characters_base_address;
uint32_t *loop_draw_frame;
uint32_t *overlay_draw_frame;
uint32_t *linux_draw_frame;
uint32_t offset_overlay;
int update_Classifier = 1;

const int CLASSIFIER_FPS = 10;					  
const int FPS_LIM = 60;
uint32_t* swap_draw_frame(){
	draw_wait_for_draw();
	*SAVED_FRAME_SWAP=1;
	return (uint32_t*)(intptr_t)(*PROCESSING_FRAME_ADDRESS);
}


int main(int argc, char **argv) {
	

	volatile uint32_t* frame_reg = (volatile uint32_t*)uio_mmap_from_addr((void*)FRAME_BASE);
	volatile uint32_t* scale_reg = (volatile uint32_t*)uio_mmap_from_addr((void*)SCALE_BASE);
	volatile uint32_t* warp_reg = (volatile uint32_t*)uio_mmap_from_addr((void*)WARP_BASE);
	volatile uint32_t* draw_reg = (volatile uint32_t*)uio_mmap_from_addr((void*)DRAW_BASE);

	if (scale_reg != NULL && warp_reg != NULL) {
		SCALER_BASE_ADDRESS = scale_reg;
		WARP_BASE_ADDRESS = warp_reg;
	} else {
        fprintf(stderr, "Unable to setup scaler and warps. Exiting\n");
        exit(1);
	}
	if (frame_reg != NULL) {
        PROCESSING_FRAME_ADDRESS 		= frame_reg + 0x44;
		PROCESSING_NEXT_FRAME_ADDRESS 	= frame_reg + 0x45;
        PROCESSING_NEXT2_FRAME_ADDRESS 	= frame_reg + 0x46;
        SAVED_FRAME_SWAP 				= frame_reg + 0x44;
		HORZ_RES_IN_ADDR 	   			= frame_reg + 0x47;
		VERT_RES_IN_ADDR 	   			= frame_reg + 0x48;
		HORZ_RES_OUT_ADDR 	   			= frame_reg + 0x49;
		VERT_RES_OUT_ADDR 	   			= frame_reg + 0x4a;
		SCALE_FACTOR_HORZ_ADDR 			= frame_reg + 0x4b;
		SCALE_FACTOR_VERT_ADDR 			= frame_reg + 0x4c;
		LINE_GAP_ADDR 					= frame_reg + 0x4d;
		RED_DDR_FRAME_START_ADDR 		= frame_reg + 0x4e;
		GREEN_DDR_FRAME_START_ADDR 		= frame_reg + 0x4f;
		BLUE_DDR_FRAME_START_ADDR 		= frame_reg + 0x50;
		FRAME_BLANKER_EN_ADDR 			= frame_reg + 0x51;
        FRAME_BLANKER_DONE_ADDR 		= frame_reg + 0x52;
        OVERLAY_DRAW_ADDR 				= frame_reg + 0x53;
        OVERLAY_DISPLAY_ADDR			= frame_reg + 0x54;
        OVERLAY_BLANK_ADDR 				= frame_reg + 0x55;
        ALPHA_BLEND_EN_ADDR 			= frame_reg + 0x56;
        FB_FRAME_START_ADDR 			= frame_reg + 0x57;
		MIN_LATENCY_SEL_ADDR            = frame_reg + 0x58;
		
        if (draw_reg != NULL){
            draw_assist_base_address = (volatile void*)draw_reg;
			for (int i = 0; i < 3; i++) loop_draw_frame=swap_draw_frame();
		} else {
			fprintf(stderr, "Unable to setup draw assist. Exiting\n");
			exit(1);
		}
    } else {
		fprintf(stderr, "Unable to setup frame swapping. Exiting\n");
		exit(1);
	}

	*FB_FRAME_START_ADDR   = 0x00000078; // @TODO remove. not used in the FPGA
	*ALPHA_BLEND_EN_ADDR   = 1;
	*FRAME_BLANKER_EN_ADDR = 1;
	*MIN_LATENCY_SEL_ADDR  = 0;
	overlay_draw_frame 	   = (uint32_t*)(intptr_t)(*OVERLAY_DRAW_ADDR);
	
	vbx_cnn_t *vbx_cnn = vbx_cnn_init(NULL);
    if (!vbx_cnn) {
        fprintf(stderr, "Unable to initialize vbx_cnn. Exiting\n");
        exit(1);
    }

	void *ascii_characters = read_ascii_file(vbx_cnn, "./frameDrawing/ascii_characters.bin");
    if (!ascii_characters) {
        fprintf(stderr, "Unable to correctly read %s. Exiting\n", "./ascii_characters.bin");
        exit(1);
    }
	ascii_characters_base_address = virt_to_phys(vbx_cnn,(void*)ascii_characters);

    int fd = fileno(stdin);
    int flags = fcntl(fd, F_GETFL, 0);
    if (fcntl(fd, F_SETFL, flags|O_NONBLOCK) != 0) {
        fprintf(stderr, "Unable to setup stdin. Exiting\n");
        exit(1);
    }
    
	char input_buf[128]="";
#if PDMA	
	int total_size = 32*1024*1024; //#TODO Check limit size in comparison
	

	pdma_out = pdma_mmap(total_size);	
	pdma_channel = pdma_ch_open();
#endif
	//Setup Models
	for (int i = 0; i < (int)(sizeof(models)/sizeof(*models)); i++) {
		model_t *model = read_model_file(vbx_cnn, models[i].fname);
		if (!model) {
			fprintf(stderr, "Unable to correctly read %s. Exiting\n", models[i].fname);
			exit(1);
		}

		if (model_check_sanity(model) != 0) {
			printf("Model %s is not sane\n", models[i].fname);
		}
		models[i].model = model;
		models[i].modelSetup_done = 1;
	}
	
	//Initialize models	
	for(int i = 0; i < (int)(sizeof(models)/sizeof(*models));i++){
		if(!strcmp(models[i].post_process_type, "RETINAFACE") || !strcmp(models[i].post_process_type, "SCRFD")) {
			demo_setup = recognitionDemoInit(vbx_cnn,models, i, 0, 1080, 1920, 0, 0);
			i+=1;
		} else if(!strcmp(models[i].post_process_type, "LPD")) {
			demo_setup = recognitionDemoInit(vbx_cnn,models, i, 0, 1080, 1920, 0, 0);
			i+=1;
		} else {
			demo_setup = detectionDemoInit(vbx_cnn, models,i);
		}
		if (demo_setup < 0) {
			printf("Error setting up %s demo\n",models[i].name);
		}
	}
  
    int mode = 0;
	int name_input = 0;
	int embedding_modify = 0;
	int* input_dims;
    short privacy = 0;
	int use_attribute_model = 0;
	int x_offset = 0;
	int y_offset = 0;

	input_dims = model_get_input_shape(models[mode].model, 0);

	int img_h = input_dims[2];
	int img_w = input_dims[3];
	
	*HORZ_RES_IN_ADDR 			= 1920;
	*VERT_RES_IN_ADDR 			= 1080;
	*HORZ_RES_OUT_ADDR 			= img_w;
	*VERT_RES_OUT_ADDR 			= img_h;
	*SCALE_FACTOR_HORZ_ADDR 	= ((1920 - 1)*1024)/img_w;
	*SCALE_FACTOR_VERT_ADDR 	= ((1080 - 1)*1024)/img_h;
	*LINE_GAP_ADDR 				= input_dims[3];
	*BLUE_DDR_FRAME_START_ADDR  = SCALER_FRAME_ADDRESS + (2*img_w*img_h);
	*GREEN_DDR_FRAME_START_ADDR = SCALER_FRAME_ADDRESS + (1*img_w*img_h);
	*RED_DDR_FRAME_START_ADDR   = SCALER_FRAME_ADDRESS + (0*img_w*img_h);
	
#if VBX_SOC_DRIVER
	if (mode > -1) {
		enable_interrupt(vbx_cnn);
    }
#endif
	
	static struct timeval tv1, tv2,prev_timestamp;
	gettimeofday(&prev_timestamp, NULL); 
	printf("Starting Demo\n");
    while(1) {
		gettimeofday(&tv1, NULL);
		int status = 1;
		while(status > 0) {
			if (privacy)
				if (!strcmp(models[mode].post_process_type, "POSENET") || !strcmp(models[mode].post_process_type, "ULTRALYTICS_POSE")){ //blank screen
					int split = 1;
					privacy_draw(split);
				}		
			if(!strcmp(models[mode].post_process_type, "RETINAFACE") || !strcmp(models[mode].post_process_type, "SCRFD") || !strcmp(models[mode].post_process_type, "LPD")) {
				status = runRecognitionDemo(models, vbx_cnn, mode, use_attribute_model, 1080, 1920, y_offset, x_offset);
			} else {
				status = runDetectionDemo(models, vbx_cnn, mode);
			}			
		}
		
		if (status < 0) {
			printf("Error running mode %d\n", mode);
			printf("control_reg = %x\n", vbx_cnn->ctrl_reg[0]);
			printf("error code: %d",vbx_cnn_get_error_val(vbx_cnn));
			printf("state = %d\n", vbx_cnn_get_state(vbx_cnn));
			while(1);
		}
		if (gettimediff_us(prev_timestamp,tv1) > 1500*1000/CLASSIFIER_FPS){
			update_Classifier = 1;
			gettimeofday(&prev_timestamp, NULL);
		}
		else{
			update_Classifier = 0;
		}													 

		gettimeofday(&tv2, NULL);
		fps = 1000/ (gettimediff_us(tv1, tv2) / 1000);
		loop_draw_frame = swap_draw_frame();
		overlay_draw_frame = (uint32_t*)(intptr_t)(*OVERLAY_DRAW_ADDR);
		
        capture_embedding = 0;
		if (fgets(input_buf, sizeof(input_buf), stdin)) {
			embedding_modify = !strcmp(models[mode].post_process_type, "RETINAFACE") || !strcmp(models[mode].post_process_type, "SCRFD");
			//Embedding capture mode enabled
			if(add_embedding_mode){
				if (tolower(input_buf[0]) == 'a' && name_input == 0) {
					name_input = 1;
					capture_embedding = 1;
				} else if (name_input ==1) {
					memset(id_entered,0,sizeof(id_entered));
					input_buf[strlen(input_buf)-1] = '\0';
					snprintf(id_entered,sizeof(id_entered),"%s",input_buf);
					if(not_duplicate(id_entered)){
						append_name(id_entered);
						name_input = 0;
						printf("Exiting +/- embedding mode \n");
						add_embedding_mode = 0;
					} else {
						printf("Embedding '%s' already exists, please select a new ID\n",id_entered);
					}
				} else {
					printf("Exiting +/- embedding mode \n");
					add_embedding_mode = 0;
				}
			}
		
			//Deletion mode enabled
			else if (delete_embedding_mode) {
				delete_embedding(input_buf,models,mode+1);
			} 	
			else if(tolower(input_buf[0])=='b'){
				if (!strcmp(models[mode].post_process_type, "POSENET") || !strcmp(models[mode].post_process_type, "ULTRALYTICS_POSE")){ //blank screen
					privacy = !privacy;
				}
			}
			else if (tolower(input_buf[0]) == 'q') {
				*ALPHA_BLEND_EN_ADDR  = 0;
                *MIN_LATENCY_SEL_ADDR = 1;			
				printf("Exiting demo\n");
				break;
			}
			//Entering deletion/addition mode for embeddings
			else if (tolower(input_buf[0]) == 'd' && embedding_modify) {
				delete_embedding_mode=1;
				print_list();
				printf("Enter the index of the embedding to be removed (default: NONE)\n");
			} else if (tolower(input_buf[0]) == 'a' && embedding_modify) {
				add_embedding_mode =1;
				printf("Enter 'a' to add the highlighted face\n");
			} else { //swap models and display  to UART 
				while(vbx_cnn_model_poll(vbx_cnn) > 0);
				mode = swap_model(mode);
				input_dims = model_get_input_shape(models[mode].model, 0);
				int img_h = input_dims[2];
				int img_w = input_dims[3];
				*HORZ_RES_IN_ADDR 			= 1920;
				*VERT_RES_IN_ADDR 			= 1080;
				*HORZ_RES_OUT_ADDR 			= img_w;
				*VERT_RES_OUT_ADDR 			= img_h;
				*SCALE_FACTOR_HORZ_ADDR 	= ((1920 - 1)*1024)/img_w;
				*SCALE_FACTOR_VERT_ADDR 	= ((1080 - 1)*1024)/img_h;
				*LINE_GAP_ADDR 				= input_dims[3];
				*BLUE_DDR_FRAME_START_ADDR  = SCALER_FRAME_ADDRESS + (2*img_w*img_h);
				*GREEN_DDR_FRAME_START_ADDR = SCALER_FRAME_ADDRESS + (1*img_w*img_h);
				*RED_DDR_FRAME_START_ADDR   = SCALER_FRAME_ADDRESS + (0*img_w*img_h);
			  #if VBX_SOC_DRIVER
				if (mode > -1) {
					enable_interrupt(vbx_cnn);
				}
			  #endif
			}
		}
	}

    return 0;
}
