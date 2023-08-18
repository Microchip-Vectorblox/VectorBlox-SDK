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


#include "frameDrawing/draw_assist.h"
#include "frameDrawing/draw.h"
#include "imageScaler/scaler.h"
#include "warpAffine/warp.h"
#include "recognitionDemo.h"
#include "detectionDemo.h"


struct model_descr_t models[] = {
        {"SCRFD", "/home/root/samples_V1000_1.4.4/scrfd_500m_bnkps.vnnx", 0, "SCRFD"},
        {"ArcFace", "/home/root/samples_V1000_1.4.4/mobilefacenet-arcface.vnnx", 0, "ARCFACE"},
        {"GenderAge", "/home/root/samples_V1000_1.4.4/genderage.vnnx", 0, "GENDERAGE"},
        {"LPD", "/home/root/samples_V1000_1.4.4/lpd_eu_v42.vnnx", 0, "LPD"},
        {"LPR", "/home/root/samples_V1000_1.4.4/lpr_eu_v3.vnnx", 0, "LPR"},
        {"MobileNet V2", "/home/root/samples_V1000_1.4.4/mobilenet-v2.vnnx", 0, "IMAGENET"},
        {"Yolo V5 Nano", "/home/root/samples_V1000_1.4.4/ultralytics.yolov5n.relu.vnnx", 0, "YOLOV5"},
        {"Tiny Yolo V4 COCO", "/home/root/samples_V1000_1.4.4/yolo-v4-tiny-tf.vnnx", 0, "YOLOV4"},
};

short demo_setup = 0;
int use_attribute_model = 0;
int fps = 0;
int add_embedding_mode = 0;
int capture_embedding = 0;
int delete_embedding_mode=0;
char id_entered[128];

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
  //printf("READ SYSFS attribute %s = 0x%"PRIx64"\n",filename,ret);
  return ret;
}


static int find_uio_dev_num_by_addr(void *target_addr){
  char buf[4096];
  for(int n=0;;n++){
    snprintf(buf,4096,"/sys/class/uio/uio%d/name",n);
    int name_fd = open(buf,O_RDONLY);
    if (name_fd>=0){
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
    }else{
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

#define FRAME_BASE         0x40001000
#define SCALE_BASE         0x60040000
#define WARP_BASE          0x60050000
#define DRAW_BASE          0x60060000
volatile uint32_t* SCALER_BASE_ADDRESS;
volatile uint32_t* WARP_BASE_ADDRESS;
volatile uint32_t* PROCESSING_FRAME_ADDRESS;
volatile uint32_t* PROCESSING_NEXT_FRAME_ADDRESS;
volatile uint32_t* PROCESSING_NEXT2_FRAME_ADDRESS;
volatile uint32_t* SAVED_FRAME_SWAP;
volatile void* draw_assist_base_address;
void* ascii_characters_base_address;
uint32_t *loop_draw_frame;
int update_Classifier = 1;

const int CLASSIFIER_FPS = 10;					  


uint32_t* swap_draw_frame(){

        draw_wait_for_draw();
        *SAVED_FRAME_SWAP=1;

        return (uint32_t*)(intptr_t)(*PROCESSING_FRAME_ADDRESS);
}


int main(int argc, char **argv) {

        void *firmware_instr = read_firmware_file("../../fw/firmware.bin");
        if (!firmware_instr) {
                fprintf(stderr, "Unable to correctly read %s. Exiting\n", argv[1]);
                exit(1);
        }

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
                PROCESSING_FRAME_ADDRESS = frame_reg + 0x44;
                PROCESSING_NEXT_FRAME_ADDRESS = frame_reg + 0x45;
                PROCESSING_NEXT2_FRAME_ADDRESS = frame_reg + 0x46;
                SAVED_FRAME_SWAP = frame_reg + 0x44;
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

        vbx_cnn_t *vbx_cnn = vbx_cnn_init(NULL, firmware_instr);
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
        if (fcntl(fd, F_SETFL, flags|O_NONBLOCK) != 0)
        {
                fprintf(stderr, "Unable to setup stdin. Exiting\n");
                exit(1);
        }
        char input_buf[128];


        for (int i = 0; i < 8; i++) {
                model_t *model = read_model_file(vbx_cnn, models[i].fname);
                if (!model) {
                        fprintf(stderr, "Unable to correctly read %s. Exiting\n", models[i].fname);
                        exit(1);
                }

                if (model_check_sanity(model) != 0) {
                        printf("Model %s is not sane\n", models[i].fname);
                };
                models[i].model = model;
                models[i].modelSetup_done = 1;
        }

        demo_setup = recognitionDemoInit(vbx_cnn, (struct model_descr_t*)models, 0, 1, 1080, 1920, 0, 0);
        if (demo_setup < 0) {
                printf("faceDemoSetup error: %d\n", demo_setup);
                exit(1);
        }
		tracksInit(models+1);
        demo_setup = recognitionDemoInit(vbx_cnn, (struct model_descr_t*)models, 3, 0, 540, 1920, 540, 0);
        if (demo_setup < 0) {
                printf("plateDemoSetup error: %d\n", demo_setup);
                exit(1);
        }
        for (int i = 0; i < 3; i++) {
                demo_setup = detectionDemoInit(vbx_cnn, (struct model_descr_t*)models, 5+i);
                if (demo_setup < 0) {
                        printf("detectionDemoSetup error: %d\n", demo_setup);
                        exit(1);
                }
        }

        int mode = 0;
		int name_input = 0;
		int embedding_modify = 0;
		if(mode>2){
			enable_interrupt(vbx_cnn);
        }
	static struct timeval tv1, tv2,prev_timestamp;
	gettimeofday(&prev_timestamp, NULL); 
	printf("Starting Demo\n");
        while(1) {
		gettimeofday(&tv1, NULL);
		int status = 1;
		while(status > 0) {
			if (mode == 0) {
				use_attribute_model = 0;
				status = runRecognitionDemo(models, vbx_cnn, 0, use_attribute_model, 1080, 1920, 0, 0);
			} else if (mode == 1){
				use_attribute_model = 1;
				status = runRecognitionDemo(models, vbx_cnn, 0, use_attribute_model, 1080, 1920, 0, 0);
			} else if (mode == 2){
				status = runRecognitionDemo(models, vbx_cnn, 3, 0, 540, 1920, 540, 0);
			} else if (mode == 3){
				status = runDetectionDemo(models, vbx_cnn, 5);
			} else if (mode == 4){
				status = runDetectionDemo(models, vbx_cnn, 6);
			} else {
				status = runDetectionDemo(models, vbx_cnn, 7);
			}
		}
		if (status < 0) {
			printf("Error running mode %d\n", mode);
			printf("control_reg = %x\n", vbx_cnn->ctrl_reg[0]);
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

	        capture_embedding = 0;
                if (fgets(input_buf, sizeof(input_buf), stdin)) {
                         			
				embedding_modify = mode <=1;
				//Embedding capture mode enabled
				if(add_embedding_mode){
					if (tolower(input_buf[0]) == 'a') {
						name_input = 1;
						capture_embedding = 1;
					} else if (name_input ==1) {
						memset(id_entered,0,sizeof(id_entered));
						strncpy(id_entered, input_buf,strlen(input_buf)-1);
						if(not_duplicate(id_entered)){
							append_name(id_entered);
							
							
							name_input = 0;
							printf("Exiting +/- embedding mode \n");
							add_embedding_mode = 0;
						} else
							printf("Embedding '%s' already exists, please select a new ID\n",id_entered);
					} else {
							printf("Exiting +/- embedding mode \n");
							add_embedding_mode = 0;
					}
				}
				
				//Deletion mode enabled
				else if (delete_embedding_mode){
					delete_embedding(input_buf,models,1);
				} 
				else if (tolower(input_buf[0]) == 'q') {
                                printf("Exiting demo\n");
                                break;
				}
				//Entering deletion/addition mode for embeddings
				else if (tolower(input_buf[0]) == 'd' && embedding_modify){
					delete_embedding_mode=1;
					print_list();
					printf("Enter the index of the embedding to be removed (default: NONE)\n");
				} else if (tolower(input_buf[0]) == 'a' && embedding_modify){
					add_embedding_mode =1;
					printf("Enter 'a' to add the highlighted face\n");

				} else { //swap models and display  to UART 
					while(vbx_cnn_model_poll(vbx_cnn) > 0);

					resize_image_hls_wait(SCALER_BASE_ADDRESS);

					mode = (mode + 1) % 6;
					
					if (mode == 0) {
							models[0].is_running = 0;
							printf("Face Recognition\n");
					} else if (mode == 1){
						models[0].is_running = 0;
						printf("Face Recognition + GenderAge\n");
					} else if (mode == 2){
						trackClean(models,4);
						models[3].is_running = 0;
						printf("Plate Recognition\n");
					} else {
						enable_interrupt(vbx_cnn);
						models[mode+2].is_running = 0;
						printf("%s\n", models[mode+2].name);
					}
				}
			
			 
			} 
        }

        return 0;
}
