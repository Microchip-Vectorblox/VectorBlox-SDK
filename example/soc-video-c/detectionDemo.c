#include "imageScaler/scaler.h"
#include "detectionDemo.h"
#include "frameDrawing/draw_assist.h"
#include "frameDrawing/draw.h"
#include <sys/time.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include "pdma/pdma_helpers.h"

//#define DRAW_SCALER
#define PDMA 1
extern int update_Classifier;
extern volatile uint32_t*  PROCESSING_FRAME_ADDRESS;
extern volatile uint32_t*  PROCESSING_NEXT_FRAME_ADDRESS;

extern volatile uint32_t*  SAVED_FRAME_SWAP;
extern int fps;

extern uint32_t* overlay_draw_frame;

extern volatile uint32_t* RED_DDR_FRAME_START_ADDR;
extern volatile uint32_t* GREEN_DDR_FRAME_START_ADDR;
extern volatile uint32_t* BLUE_DDR_FRAME_START_ADDR;

static inline void* virt_to_phys(vbx_cnn_t* vbx_cnn,void* virt){
	return (char*)(virt) + vbx_cnn->dma_phys_trans_offset;
}
#ifdef DRAW_SCALER
static void* mmap_buffer(uint32_t start_address){
    int fd = open("/dev/mem", O_RDWR);
    size_t size = 128*1024*1024;
    void* _ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0x3000000000 + start_address);
    close(fd);
    return _ptr;
}


void copy_planer_image(uint32_t* alpha_frame,uint8_t *planer_image,int x_offset,int y_offset,int width,int height){
    int plane_size = width*height;
								
	for(int y=0;y<height;++y){
        for(int x=0;x<width;++x){
			uint32_t pixel = (0x00 | (planer_image[(2*plane_size)+(width*y+x)]<<16) |
									 (planer_image[(1*plane_size)+(width*y+x)]<<8) |
									 (planer_image[(0*plane_size)+(width*y+x)]<<0));			


			alpha_frame[(y+y_offset)*(0x2000/4)+x+x_offset]=pixel;
        }
    }
}

uint32_t* scaler_draw_frame;
uint32_t* scaler_frame_addr[8];

static inline uint32_t* virt_loop() {
  uint32_t offset = (uint32_t)overlay_draw_frame - 0x70000000;
  return (uint32_t*)((char*)(scaler_draw_frame) + offset);
}
#endif

static int gettimediff_us_2(struct timeval start, struct timeval end) {
	int sec = end.tv_sec - start.tv_sec;
	int usec = end.tv_usec - start.tv_usec;
	return sec * 1000000 + usec;
}

extern int8_t* pdma_mmap_t;
extern uint64_t pdma_out;
extern int32_t pdma_channel;

int32_t pdma_ch_transfer(uint64_t output_data_phys, void* source_buffer,int offset,int size,vbx_cnn_t *vbx_cnn,int32_t channel){
	uint64_t srcbuf=0x3000000000 + (uint64_t)(uintptr_t)virt_to_phys(vbx_cnn, source_buffer);
	return pdma_ch_cpy(output_data_phys + offset, srcbuf, size, channel);
}

#if VBX_SOC_DRIVER
extern uint32_t *linux_draw_frame;
static inline uint32_t* virt_loop_linux(uint32_t* loop) {
	uint32_t offset = ((uint32_t)(uintptr_t)overlay_draw_frame) - 0x70000000;
	return (uint32_t*)((char*)(linux_draw_frame) + offset);
}
#endif

#define POSE 1
#define BOXES_LEN 1024
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
static fix16_t fix16_two = F16(2);
// Globals Specification
const int topk = 4;
int16_t indexes[4] = {0};
int16_t display_index[4] = {0};
int32_t scores[4] = {0};

fix16_box boxes[BOXES_LEN];
int valid_boxes = 0;
int m_run_fps=0;;



void draw_post_process(struct model_descr_t *model, const int fps,int int8_flag,vbx_cnn_t* vbx_cnn,vbx_cnn_io_ptr_t *pdma_buffer) {
	char label[256];
	int *in_dims = model_get_input_shape(model->model,0);
	int input_h = in_dims[2];
	int input_w = in_dims[3];	
	const char* pptype = model->post_process_type;

#if PDMA
	fix16_t** output_buffers = (fix16_t**)pdma_buffer;
#else	
	fix16_t** output_buffers = model->pipelined_output_buffers[model->buf_idx];

#endif
	// Run Post Processing of previous frame's outputs
	if (!strcmp(pptype, "YOLOV2") ||
			!strcmp(pptype, "YOLOV3") ||
			!strcmp(pptype, "YOLOV4") ||
			!strcmp(pptype, "YOLOV5") ||
			!strcmp(pptype, "ULTRALYTICS") ||
			!strcmp(pptype, "ULTRALYTICS_CUT") ||
			!strcmp(pptype, "SSDV2")){
		int max_boxes = 100;
		fix16_t thresh = F16(0.5);
		fix16_t iou = F16(0.4);
		int num_outputs;
		yolo_info_t cfg[3];
		fix16_t *outputs[3];
		char** class_names = coco_classes;
		char *is_tiny = strstr(model->name, "Tiny");

		if(!strcmp(pptype, "YOLOV2")) {
			num_outputs=1;
			outputs[0] = (fix16_t*)(uintptr_t)output_buffers[0];
			int output_length = (int)model_get_output_length(model->model, 0);

			if(output_length == 125*13*13){ // yolo v2 voc
				class_names = voc_classes;
				static fix16_t tiny_anchors[] ={F16(1.08),F16(1.19),F16(3.42),F16(4.41),F16(6.63),F16(11.38),F16(9.42),F16(5.11),F16(16.620001),F16(10.52)};
				static fix16_t anchors[] = {F16(1.3221),F16(1.73145),F16(3.19275),F16(4.00944),F16(5.05587),F16(8.09892),F16(9.47112),F16(4.84053),F16(11.2364),F16(10.0071)};
				yolo_info_t cfg_0 = {
					.version = 2,
					.input_dims = {3, input_h, input_w},
					.output_dims = {125, 13, 13},
					.coords = 4,
					.classes = 20,
					.num = 5,
					.anchors_length = 10,
					.anchors = is_tiny ? tiny_anchors : anchors,
				};
				cfg[0] = cfg_0;
				valid_boxes = post_process_yolo(outputs, num_outputs, cfg, thresh, iou, boxes, max_boxes);
			}
			else if (output_length ==  425*13*13 || output_length == 425*19*19){ // yolo v2 coco
				class_names = coco_classes;
				int w, h, i;
				if (output_length == 425*13*13) {
					i = 416; h = 13; w = 13;
				} else {
					i = 608; h = 19; w = 19;
				}
				fix16_t anchors[] ={F16(0.57273),F16(0.677385),F16(1.87446),F16(2.06253),F16(3.33843),F16(5.47434),F16(7.88282),F16(3.52778),F16(9.77052),F16(9.16828)};
				yolo_info_t cfg_0 = {
					.version = 2,
					.input_dims = {3, i, i},
					.output_dims = {425, h, w},
					.coords = 4,
					.classes = 80,
					.num = 5,
					.anchors_length = 10,
					.anchors = anchors,
				};
				cfg[0] = cfg_0;
				valid_boxes = post_process_yolo(outputs, num_outputs, cfg, thresh, iou, boxes, max_boxes);
			}
		} else if (!strcmp(pptype, "ULTRALYTICS")){
			class_names = coco_classes;			
			

			fix16_t f16_scale = (fix16_t)model_get_output_scale_fix16_value(model->model,0); // get output scale
			int32_t zero_point = model_get_output_zeropoint(model->model,0); // get output zero
			if(int8_flag){
				int8_t* output_int8 =(int8_t*)(uintptr_t)output_buffers[0];
				//int8_t* output_int8=(int8_t*)(uintptr_t)model_get_test_output(model->model,0);
				valid_boxes = post_process_ultra_nms_int8(output_int8, 8400, input_h, input_w,f16_scale,zero_point, thresh, iou, boxes, max_boxes);
			}
			else{
				fix16_t* output=(fix16_t*)(uintptr_t)output_buffers[0];
				valid_boxes = post_process_ultra_nms(output,8400, input_h, input_w, thresh, iou, boxes, max_boxes);
			}

		} else if (!strcmp(pptype, "ULTRALYTICS_CUT")){
			class_names = coco_classes;
			int* outputs_shape[6];
			int8_t *outputs_int8[6];
			int zero_points[6];
			fix16_t scale_outs[6];

			// put outputs in this order
			// type:  {class_stride8, box_stride8,   class_stride16,  box_stride16,    class_stride32,  box_stride32}
			// shape: {[1,80,H/8,W/8],[1,64,H/8,W/8],[1,80,H/16,W/16],[1,64,H/16,W/16],[1,80,H/32,W/32],[1,64,H/32,W/32]}
			int32_t w_min = 0x7FFFFFFF;	// minimum width must be stride32
			int32_t w_max = 0;			// maximum width must be stride8
			int* shapes[6];			
			for(int n=0; n<6; n++){
				shapes[n] = model_get_output_shape(model->model,n);
				w_min = MIN(shapes[n][3], w_min);
				w_max = MAX(shapes[n][3], w_max);
			}
			for(int i=0;i<6;i++){
				int o; //proper order
				if(shapes[i][3]==w_min) o=4;				//stride 8
				else if (shapes[i][3]==w_max) o=0;			//stride 32
				else o=2;									//stride 16
				if(shapes[i][1]==64) o+=1;					//box (otherwise class)
				outputs_shape[o] = shapes[i];
				outputs_int8[o] = (int8_t*)(uintptr_t)output_buffers[i];
				zero_points[o] = model_get_output_zeropoint(model->model,i);
				scale_outs[o]=model_get_output_scale_fix16_value(model->model,i);	
			}
			const int max_detections = 200;
			fix16_t post_buffer[max_detections*84];
			int post_len;
			post_len = post_process_ultra_int8(outputs_int8, outputs_shape, post_buffer, thresh, zero_points, scale_outs, max_detections);
			valid_boxes = post_process_ultra_nms(post_buffer, post_len, input_h, input_w, thresh, iou, boxes, BOXES_LEN);
			

		}

		else if (!strcmp(pptype, "YOLOV3") || !strcmp(pptype, "YOLOV4")) {
			class_names = coco_classes;
			if (is_tiny) {
				num_outputs = 2;
				int output_sizes[2] = {255*13*13, 255*26*26};
				for (int o = 0; o < num_outputs; o++) {
					for (int i = 0; i < num_outputs; i++) {
						if (model_get_output_length(model->model,i) == output_sizes[o]) {
							outputs[o] = (fix16_t*)(uintptr_t)output_buffers[i];
						}
					}
				}
				static fix16_t tiny_anchors[] = {F16(10),F16(14),F16(23),F16(27),F16(37),F16(58),F16(81),F16(82),F16(135),F16(169),F16(344),F16(319)}; // 2*num
				static int mask_0[] = {3,4,5};
				static int mask_1[] = {1,2,3}; 

				yolo_info_t cfg_0 = {
					.version = 3,
					.input_dims = {3, input_h, input_w},
					.output_dims = {255, 13, 13},
					.coords = 4,
					.classes = 80,
					.num = 6,
					.anchors_length = 12,
					.anchors = tiny_anchors,
					.mask_length = 3,
					.mask = mask_0,
				};

				yolo_info_t cfg_1 = {
					.version = 3,
					.input_dims = {3, input_h, input_w},
					.output_dims = {255, 26, 26},
					.coords = 4,
					.classes = 80,
					.num = 6,
					.anchors_length = 12,
					.anchors = tiny_anchors,
					.mask_length = 3,
					.mask = mask_1,
				};

				yolo_info_t cfg[] = {cfg_0, cfg_1};
				valid_boxes = post_process_yolo(outputs, num_outputs, cfg, thresh, iou, boxes, max_boxes);
			} else {
				num_outputs = 3;
				int output_sizes[3] = {255*19*19, 255*38*38, 255*76*76};
				for (int o = 0; o < num_outputs; o++) {
					for (int i = 0; i < num_outputs; i++) {
						if (model_get_output_length(model->model,i) == output_sizes[o]) {
							outputs[o] = (fix16_t*)(uintptr_t)output_buffers[i];
						}
					}
				}
				static fix16_t anchors[] = {F16(10),F16(13),F16(16),F16(30),F16(33),F16(23),F16(30),F16(61),F16(62),F16(45),F16(59),F16(119),F16(116),F16(90),F16(156),F16(198),F16(373),F16(326)};
				static int mask_0[] = {6,7,8};
				static int mask_1[] = {3,4,5};
				static int mask_2[] = {0,1,2};

				yolo_info_t cfg_0 = {
					.version = 3,
					.input_dims = {3, input_h, input_w},
					.output_dims = {255, 19, 19},
					.coords = 4,
					.classes = 80,
					.num = 9,
					.anchors_length = 18,
					.anchors = anchors,
					.mask_length = 3,
					.mask = mask_0,
				};
				yolo_info_t cfg_1 = {
					.version = 3,
					.input_dims = {3, input_h, input_w},
					.output_dims = {255, 38, 38},
					.coords = 4,
					.classes = 80,
					.num = 9,
					.anchors_length = 18,
					.anchors = anchors,
					.mask_length = 3,
					.mask = mask_1,
				};
				yolo_info_t cfg_2 = {
					.version = 3,
					.input_dims = {3, input_h, input_w},
					.output_dims = {255, 76, 76},
					.coords = 4,
					.classes = 80,
					.num = 9,
					.anchors_length = 18,
					.anchors = anchors,
					.mask_length = 3,
					.mask = mask_2,
				};

				yolo_info_t cfg[] = {cfg_0, cfg_1, cfg_2};
				valid_boxes = post_process_yolo(outputs, num_outputs, cfg, thresh, iou, boxes, max_boxes);
			}
		}
		else if (!strcmp(pptype, "YOLOV5")){ //ultralytics

			class_names = coco_classes;
			int num_outputs = 3;
			thresh =F16(.25);
			fix16_t *outputs[3];
			int8_t *outputs_int8[3];
			int zero_points[3];
			fix16_t scale_outs[3];
			int output_sizes[3] = {255*13*13, 255*26*26, 255*52*52};
			for (int o = 0; o < num_outputs; o++) {
			  for (int i = 0; i < num_outputs; i++) {
			    if (model_get_output_length(model->model,i) == output_sizes[o]) {
				outputs[o] = (fix16_t*)(uintptr_t)output_buffers[i];
				outputs_int8[o] = (int8_t*)(uintptr_t)output_buffers[i];
				zero_points[o] = model_get_output_zeropoint(model->model,i);
				scale_outs[o]=model_get_output_scale_fix16_value(model->model,i);
			    }
			  }
			}			

			fix16_t anchors[] = {F16(10),F16(13),F16(16),F16(30),F16(33),F16(23),F16(30),F16(61),F16(62),F16(45),F16(59),F16(119),F16(116),F16(90),F16(156),F16(198),F16(373),F16(326)};
			int mask_0[] = {6,7,8};
			int mask_1[] = {3,4,5};
			int mask_2[] = {0,1,2};

			yolo_info_t cfg_0 = {
				.version = 5,
				.input_dims = {3, input_h, input_w},
				.output_dims = {255, 13, 13},
				.coords = 4,
				.classes = 80,
				.num = 9,
				.anchors_length = 18,
				.anchors = anchors,
				.mask_length = 3,
				.mask = mask_0,
			};

			yolo_info_t cfg_1 = {
				.version = 5,
				.input_dims = {3, input_h, input_w},
				.output_dims = {255, 26, 26},
				.coords = 4,
				.classes = 80,
				.num = 9,
				.anchors_length = 18,
				.anchors = anchors,
				.mask_length = 3,
				.mask = mask_1,
			};
			yolo_info_t cfg_2 = {
				.version = 5,
				.input_dims = {3, input_h, input_w},
				.output_dims = {255, 52, 52},
				.coords = 4,
				.classes = 80,
				.num = 9,
				.anchors_length = 18,
				.anchors = anchors,
				.mask_length = 3,
				.mask = mask_2,
			};

			yolo_info_t cfg[] = {cfg_0, cfg_1, cfg_2};

			if(int8_flag){
				valid_boxes = post_process_yolo_int8(outputs_int8, num_outputs, zero_points, scale_outs, cfg, thresh, iou, boxes, max_boxes);
			}
			else{
				valid_boxes = post_process_yolo(outputs, num_outputs, cfg, thresh, iou, boxes, max_boxes);
			}

		}
		else if (!strcmp(pptype, "SSDV2")){
			fix16_t* ssdv2_outputs[12];
			char *is_vehicle = strstr(model->name, "vehicle");
			char *is_torch = strstr(model->name,"torch");
			if (is_vehicle == NULL) is_vehicle = strstr(model->name, "EHICLE");

			fix16_t confidence_threshold=F16(0.5);
			fix16_t nms_threshold=F16(0.4);
			int8_t* output_buffers_int8[12];
			fix16_t f16_scale[12];
			int32_t zero_point[12];
			if(is_torch){
				for(int o=0; o<12;++o){
				    int* oshape = model_get_output_shape(model->model,o);
				    int idx;
					//idx = 12 - 2*(oshape[2] - (oshape[2]/5) - 3*(oshape[2]/10) - 4*(oshape[2]/20)) + oshape[1]/546;
					if (oshape[2] == 1) {
					    idx = 5*2;   
				    } else if (oshape[2] == 2) {
					    idx = 4*2;   
				    } else if (oshape[2] == 3) {
					    idx = 3*2;   
				    } else if (oshape[2] == 5) {
					    idx = 2*2;   
				    } else if (oshape[2] == 10) {
					    idx = 1*2;    
				    } else {
					    idx = 0*2;
				    }
				    if (oshape[1] == 546) {
					    idx += 1;
				    }		
					ssdv2_outputs[idx]=(fix16_t*)(uintptr_t)output_buffers[o];
				    output_buffers_int8[idx] = (int8_t*)(uintptr_t)output_buffers[o];
				    f16_scale[idx] = model_get_output_scale_fix16_value(model->model,o);
				    zero_point[idx] = model_get_output_zeropoint(model->model,o);	
				}
				if(int8_flag){	
					valid_boxes = post_process_ssd_torch_int8(boxes,max_boxes,output_buffers_int8,f16_scale,zero_point, 91,confidence_threshold,nms_threshold);
				} else{
					valid_boxes = post_process_ssd_torch(boxes,max_boxes,ssdv2_outputs,91,confidence_threshold,nms_threshold);
				}
				class_names = coco91_classes;				
			}


			else if (is_vehicle) {
				for(int o=0;o<6;++o){
					ssdv2_outputs[2*o]=(fix16_t*)(uintptr_t)output_buffers[(6-1-o)*2];
					ssdv2_outputs[2*o+1]=(fix16_t*)(uintptr_t)output_buffers[(6-1-o)*2+1];
				}
				valid_boxes = post_process_vehicles(boxes,max_boxes,ssdv2_outputs,3,confidence_threshold,nms_threshold);
				class_names = vehicle_classes;

			} else {
				for(int o=0;o<12;++o){
					ssdv2_outputs[o]=(fix16_t*)(uintptr_t)output_buffers[(11-o)];
				}
				valid_boxes = post_process_ssdv2(boxes,max_boxes,ssdv2_outputs,91,confidence_threshold,nms_threshold);
				class_names = coco91_classes;
			}
		}

		// Assign names to valid objects detected
		for(int b =0 ;b<valid_boxes;++b) {
			const char* name = class_names[boxes[b].class_id];
			boxes[b].class_name = name;
		}
	} else if(!strcmp(pptype, "IMAGENET")) {
		
		fix16_t f16_scale = (fix16_t)model_get_output_scale_fix16_value(model->model,0); // get output scale

		int32_t zero_point = model_get_output_zeropoint(model->model,0); // get output zero
		int output_length = (int)model_get_output_length(model->model, 0);
		//int8_t* output_buffers_u8 = (int8_t*)model_get_test_output(model->model,0);
		int8_t* output_buffers_u8 = (int8_t*)(uintptr_t)output_buffers[0];// output_buffers[0];
		//int8_t* output_buffers_u8 = (int8_t*)(uintptr_t)model->model_io_buffers[1];// output_buffers[0];
		post_process_classifier_int8(output_buffers_u8,output_length,indexes,topk);
		
		//post_process_classifier(output_buffers[0],output_length,indexes,topk);
		if(update_Classifier)
		for(int i=0;i<topk;++i) {
			int idx = indexes[i];
			display_index[i]=indexes[i];
			scores[i] = fix16_mul(fix16_from_int((int32_t)(output_buffers_u8[idx])-zero_point),f16_scale);
			
		}
	} else if (!strcmp(pptype, "SCRFD")) {
		int length = 0;
		const int MAX_FACES=24;
		object_t faces[MAX_FACES];
		fix16_t confidence_threshold=F16(0.8);
		fix16_t nms_threshold=F16(0.4);
		fix16_t* fix16_buffers[9];
		int8_t* output_buffer_int8[9];
		int zero_points[9];
		fix16_t scale_outs[9];
		
		for(int o=0; o<model_get_num_outputs(model->model); o++){
			int *output_shape = model_get_output_shape(model->model,o);
			int ind = (output_shape[1]/8)*3 + (2-(output_shape[2]/18)); //first dim should be {2,8,20} second dim should be {9,18,36}
			fix16_buffers[ind]=(fix16_t*)(uintptr_t)output_buffers[o]; //assigns output buffers by first dim ascending, second descending
			output_buffer_int8[ind]= (int8_t*)(uintptr_t)output_buffers[o];
			zero_points[ind]=model_get_output_zeropoint(model->model,o);
			scale_outs[ind]=model_get_output_scale_fix16_value(model->model,o);
		}
		if(int8_flag){
			
			length = post_process_scrfd_int8(faces,MAX_FACES,output_buffer_int8, zero_points, scale_outs, input_w, input_h,
				confidence_threshold,nms_threshold,model->model);
		}	

		else{			
			length = post_process_scrfd(faces, MAX_FACES, fix16_buffers, input_w, input_h,
				confidence_threshold,nms_threshold);
		}

		fix16_t hratio = fix16_div(fix16_from_int(1080),fix16_from_int(input_h));
		fix16_t wratio = fix16_div(fix16_from_int(1920),fix16_from_int(input_w));
		for(int f=0;f<length;f++){
			object_t* face = faces+f;
			fix16_t x = face->box[0];
			fix16_t y = face->box[1];
			fix16_t w = face->box[2] - face->box[0];
			fix16_t h = face->box[3] - face->box[1];
			if( x > 0 &&  y > 0 && w > 0 && h > 0) {
				x = fix16_mul(x, wratio);
				y = fix16_mul(y, hratio);
				w = fix16_mul(w, wratio);
				h = fix16_mul(h, hratio);
				draw_box(fix16_to_int(x),
						fix16_to_int(y),
						fix16_to_int(w),
						fix16_to_int(h),
						5,
						get_colour_modulo(0),
						overlay_draw_frame,2048,1080);
			}
		}
	} else if (!strcmp(pptype, "LPD")) {
		const int MAX_PLATES=24;
		object_t plates[MAX_PLATES];
		fix16_t confidence_threshold=F16(0.55);
		fix16_t nms_threshold=F16(0.2);

		int image_w = 1024;
		int image_h = 288;
		int length = post_process_lpd(plates,MAX_PLATES, output_buffers, image_w, image_h,
				confidence_threshold,nms_threshold, model_get_num_outputs(model->model));
		fix16_t hratio = fix16_div(fix16_from_int(1080/2), fix16_from_int(image_h));
		fix16_t wratio = fix16_div(fix16_from_int(1920), fix16_from_int(image_w));
		for(int f=0;f<length;f++){
			object_t* plate = plates+f;
			fix16_t x = plate->box[0];
			fix16_t y = plate->box[1];
			fix16_t w = plate->box[2];
			fix16_t h = plate->box[3];
			x = fix16_sub(x, fix16_div(w, fix16_two));
			y = fix16_sub(y, fix16_div(h, fix16_two));
			if( x > 0 &&  y > 0 && w > 0 && h > 0) {
				x = fix16_mul(x, wratio);
				y = fix16_mul(y, hratio);
				w = fix16_mul(w, wratio);
				h = fix16_mul(h, hratio);
				draw_box(fix16_to_int(x),
						fix16_to_int(y)+540,
						fix16_to_int(w),
						fix16_to_int(h),
						5,
						get_colour_modulo(0),
						overlay_draw_frame,2048,1080);
			}
		}
	} 
#if POSE	
	else if (!strcmp(pptype, "POSENET")) {
		const int MAX_TOTALPOSE=5;
		const int NUM_KEYPOINTS=17;
		
		poses_t r_poses[MAX_TOTALPOSE];		
		int *output_dims = model_get_output_shape(model->model,1);
		int poseScoresH = output_dims[2]; 
		int poseScoresW = output_dims[3]; 		

	
		int outputStride = 16;
		int nmsRadius = 20;
		int pose_count = 0;
		fix16_t minPoseScore = F16(.25);
		fix16_t scoreThreshold = F16(0.5);
		fix16_t score = 0;
		if (int8_flag) {
			int zero_points[4];
			fix16_t scale_outs[4];
			int8_t* scores_8, *offsets_8, *displacementsFwd_8, *displacementsBwd_8;
			scores_8 = (int8_t*)(uintptr_t)output_buffers[1];
			offsets_8 = (int8_t*)(uintptr_t)output_buffers[0];
			displacementsFwd_8 = (int8_t*)(uintptr_t)output_buffers[2];
			displacementsBwd_8 = (int8_t*)(uintptr_t)output_buffers[3];
			for(int o=0; o<model_get_num_outputs(model->model); o++){
				zero_points[o] = model_get_output_zeropoint(model->model,o);
				scale_outs[o]=model_get_output_scale_fix16_value(model->model,o);
			}
			
			pose_count = decodeMultiplePoses_int8(r_poses,scores_8,offsets_8,displacementsFwd_8,displacementsBwd_8, outputStride, MAX_TOTALPOSE, scoreThreshold, nmsRadius, minPoseScore,poseScoresH,poseScoresW,zero_points,scale_outs); //actualpostprocess code			
		}
		else{
			fix16_t* scores, *offsets, *displacementsFwd, *displacementsBwd;
			scores = (fix16_t*)(uintptr_t)output_buffers[1];
			offsets = (fix16_t*)(uintptr_t)output_buffers[0];
			displacementsFwd = (fix16_t*)(uintptr_t)output_buffers[2];
			displacementsBwd = (fix16_t*)(uintptr_t)output_buffers[3];
			pose_count = decodeMultiplePoses(r_poses,scores,offsets,displacementsFwd,displacementsBwd, outputStride, MAX_TOTALPOSE, scoreThreshold, nmsRadius, minPoseScore,poseScoresH,poseScoresW); //actualpostprocess code
		}
		
		
		int imageH, imageW;
		imageH = 1080; //default img feed dims
		imageW = 1920; //default img feed dims
		
		int *model_dims = model_get_input_shape(model->model,0);
		int modelInputH = model_dims[2];
		int modelInputW = model_dims[3];
		fix16_t scale_Y = fix16_div(fix16_from_int(imageH),fix16_from_int(modelInputH));
		fix16_t scale_X = fix16_div(fix16_from_int(imageW),fix16_from_int(modelInputW));
		
		//scales up the image
		for(int i = 0; i < pose_count; i++) {
			for(int j = 0; j < NUM_KEYPOINTS; j++) {
				r_poses[i].keypoints[j][0] = fix16_mul(r_poses[i].keypoints[j][0],scale_Y);
				r_poses[i].keypoints[j][1] = fix16_mul(r_poses[i].keypoints[j][1],scale_X);
			}       
		}
		
		//begin draw
		int radius = 6;
		//int skeleton [19][2] = {{15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12}, {5, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10}, {0, 1}, {0, 2}, {1, 3}, {2, 4}, {1, 2}, {3, 5}, {4, 6}};
		int color;
		for(int i =0; i < pose_count; i++) {
			//Draw lines/edges
			/* Disabled for now since draw line doesn't work yet
			for(int z=0; z<16; z++) { 
				int y_source = fix16_to_int(r_poses[i].keypoints[skeleton[z][0]][0]); //source y
				int x_source = fix16_to_int(r_poses[i].keypoints[skeleton[z][0]][1]); //source x
				int y_target = fix16_to_int(r_poses[i].keypoints[skeleton[z][1]][0]); //target y
				int x_target = fix16_to_int(r_poses[i].keypoints[skeleton[z][1]][1]); //target y
				
				if(r_poses[i].scores[skeleton[z][0]]>0 && r_poses[i].scores[skeleton[z][1]]>0) {
					if(r_poses[i].scores[skeleton[z][0]]>F16(1) && r_poses[i].scores[skeleton[z][1]]>F16(1)) {
						color = GET_COLOUR(0, 255, 0, 255); //green
					} else {
						color = GET_COLOUR(255,255,0,255);  //yellow
					}
				  #if VBX_SOC_DRIVER					
					uint32_t *line = virt_loop(overlay_draw_frame);
					draw_assist_draw_line_pp(x_source, y_source, x_target, y_target,
											line, 2048,color,1,100);
				  #else
					draw_line(x_source, y_source, x_target, y_target,color, overlay_draw_frame, 2048,1080, 2,0);
				  #endif
				}
			}*/
			//Draw out points
			for(int j = 0; j < NUM_KEYPOINTS; j++){
				int y = fix16_to_int(r_poses[i].keypoints[j][0]);
				int x = fix16_to_int(r_poses[i].keypoints[j][1]);
				score = r_poses[i].scores[j];
				if(score > F16(1)) {
					color = GET_COLOUR(0, 0, 255, 255);  //blue
					draw_box(x,y,3,3,radius, color, overlay_draw_frame,2048,1080);
				} else if (score > 0) {
					color = GET_COLOUR(255,0,0,255);  //red
					draw_box(x,y,3,3,radius, color, overlay_draw_frame,2048,1080);
				}
			}
		}
	}
#endif
	//Draw outputs from last frame

	snprintf(label,sizeof(label),"%s %dx%d  %d fps",model->name,input_w,input_h, fps);
	
	draw_label(label,20,2,overlay_draw_frame,2048,1080,WHITE);

	if (!strcmp(pptype, "YOLOV2") ||
			!strcmp(pptype, "YOLOV3") ||
			!strcmp(pptype, "YOLOV4") ||
			!strcmp(pptype, "YOLOV5") ||
			!strcmp(pptype, "ULTRALYTICS") ||
			!strcmp(pptype, "ULTRALYTICS_CUT") ||
			!strcmp(pptype, "SSDV2")) {
		//int* input_shape = model_get_input_shape(model->model,0);
		for(int i=0;i<valid_boxes;++i) {
			if(boxes[i].confidence == 0) {
				continue;
			}
			int x = boxes[i].xmin,y=boxes[i].ymin;
			int w = boxes[i].xmax-boxes[i].xmin;
			int h = boxes[i].ymax-boxes[i].ymin;
			x = x*1920/input_w;
			w = w*1920/input_w;
			y = y*1080/input_h;
			h = h*1080/input_h;
			if(x<0 || y<0 || w<=0 || h<=0) {
				continue;
			}
			draw_box(x,y,w,h,5,get_colour_modulo(boxes[i].class_id),
					overlay_draw_frame,2048,1080);
			draw_label(boxes[i].class_name,x+5,y+5, overlay_draw_frame,2048,1080,WHITE);
		}

	} else if(!strcmp(pptype, "IMAGENET")) {
		int idx, output_length;
		int32_t score;
		char* classifier_name="";
		for(int i=0;i<topk;++i) {
			idx = display_index[i];
			score = scores[i];
			output_length = (int)model_get_output_length(model->model, 0);
			if(output_length == 1001 || output_length == 1000){ // imagenet
				classifier_name = imagenet_classes[idx];
				if(output_length==1001){
					//some imagenet networks have a null catagory, account for that
					classifier_name =  imagenet_classes[idx-1];
				}
			} 
			snprintf(label,sizeof(label),"%d %s %d%%",i,classifier_name,(score*100)>>16);
			draw_label(label,20,36+i*34,overlay_draw_frame,2048,1080,WHITE);
		}
	}

}

short detectionDemoInit(vbx_cnn_t* the_vbx_cnn, struct model_descr_t* models, uint8_t modelIdx) {
	struct model_descr_t *object_model = models+modelIdx;
	object_model->buf_idx=0;
	object_model->is_running = 0;
	object_model->model_io_buffers  = vbx_allocate_dma_buffer(the_vbx_cnn, (1+model_get_num_outputs(object_model->model))*sizeof(object_model->model_io_buffers), 0);
	if(!object_model->model_io_buffers){
		printf("Memory allocation issue for model io buffers.\n");
		return -1;
	}

	size_t input_length = 0;
	input_length = model_get_input_length(object_model->model, 0) *
		((int)model_get_input_datatype(object_model->model, 0) + 1);
	object_model->model_input_buffer = vbx_allocate_dma_buffer(the_vbx_cnn, input_length, 0);
	object_model->model_io_buffers[0] = (uintptr_t)object_model->model_input_buffer;
	if(!object_model->model_input_buffer){
		printf("Memory allocation issue for model input buffers.\n");
		return -1;	
	}

	// Determine the input and output lengths of the Object Model, and re-order output buffers if applicable
	int num_outputs = model_get_num_outputs(object_model->model);
	for(int output = 0; output < num_outputs; output++) {
		int output_length = model_get_output_length(object_model->model, output) * 4; // fix16 outputs are 4 bytes per element?
		object_model->pipelined_output_buffers[0][output] = vbx_allocate_dma_buffer(the_vbx_cnn, output_length, 0);
		object_model->pipelined_output_buffers[1][output] = vbx_allocate_dma_buffer(the_vbx_cnn, output_length, 0);
		object_model->model_io_buffers[output+1] = (uintptr_t)object_model->pipelined_output_buffers[0][output];
		if(!object_model->pipelined_output_buffers[0][output] || !object_model->pipelined_output_buffers[1][output]){
			printf("Memory allocation issue for model output buffers.\n");
			return -1;	
		}
	}


#ifdef DRAW_SCALER
	scaler_draw_frame       = (uint32_t *)mmap_buffer(0x70000000);
	scaler_frame_addr[0]    = (uint32_t *)mmap_buffer(0x68000000);
	scaler_frame_addr[1]	= (uint32_t *)mmap_buffer(0x69000000);
	scaler_frame_addr[2]	= (uint32_t *)mmap_buffer(0x6a000000);
	scaler_frame_addr[3]	= (uint32_t *)mmap_buffer(0x6b000000);
	scaler_frame_addr[4]	= (uint32_t *)mmap_buffer(0x6c000000);
	scaler_frame_addr[5]	= (uint32_t *)mmap_buffer(0x6d000000);
	scaler_frame_addr[6]	= (uint32_t *)mmap_buffer(0x6e000000);
	scaler_frame_addr[7]	= (uint32_t *)mmap_buffer(0x6f000000);
#endif
	return 1;
}

int runDetectionDemo(struct model_descr_t* models, vbx_cnn_t* the_vbx_cnn, uint8_t modelIdx){
	int err, status;
	struct timeval m_run1, m_run2;
	struct model_descr_t *object_model = models+modelIdx;
	int* input_dims = model_get_input_shape(object_model->model, 0);	
	uint32_t offset;
	//Start processing the network if not already running - 1st pass only (frame 0 )
	if(!object_model->is_running) {		
		*BLUE_DDR_FRAME_START_ADDR  = (2*input_dims[3]*input_dims[2]);
		*GREEN_DDR_FRAME_START_ADDR = (1*input_dims[3]*input_dims[2]);
		*RED_DDR_FRAME_START_ADDR   = (0*input_dims[3]*input_dims[2]); 
		offset = (*PROCESSING_FRAME_ADDRESS) - 0x70000000;
		object_model->model_input_buffer = (uint8_t*)(uintptr_t)(SCALER_FRAME_ADDRESS + offset);
#ifdef DRAW_SCALER
		uint32_t *scaler_frame = virt_loop();
		if ( offset == 0) {
		  copy_planer_image(scaler_frame, (uint8_t*)scaler_frame_addr[0], 100, 100, input_dims[3],input_dims[2]);
		} else if (offset == 0x1000000) {
		  copy_planer_image(scaler_frame, (uint8_t*)scaler_frame_addr[1], 100, 100, input_dims[3],input_dims[2]);
		} else if (offset == 0x2000000) {
		  copy_planer_image(scaler_frame, (uint8_t*)scaler_frame_addr[2], 100, 100, input_dims[3],input_dims[2]);
		} else if (offset == 0x3000000) {
		  copy_planer_image(scaler_frame, (uint8_t*)scaler_frame_addr[3], 100, 100, input_dims[3],input_dims[2]);
		} else if (offset == 0x4000000) {
		  copy_planer_image(scaler_frame, (uint8_t*)scaler_frame_addr[4], 100, 100, input_dims[3],input_dims[2]);
		} else if (offset == 0x5000000) {
		  copy_planer_image(scaler_frame, (uint8_t*)scaler_frame_addr[5], 100, 100, input_dims[3],input_dims[2]);
		} else if (offset == 0x6000000) {
		  copy_planer_image(scaler_frame, (uint8_t*)scaler_frame_addr[6], 100, 100, input_dims[3],input_dims[2]);
		} else if (offset == 0x7000000) {
		  copy_planer_image(scaler_frame, (uint8_t*)scaler_frame_addr[7], 100, 100, input_dims[3],input_dims[2]);
		}
#endif
		object_model->model_io_buffers[0] = (uintptr_t)object_model->model_input_buffer - the_vbx_cnn->dma_phys_trans_offset;	
//		object_model->model_io_buffers[0] = (uintptr_t)model_get_test_input(object_model->model,0);
		
		// Start model
		gettimeofday(&m_run1, NULL);
		err = vbx_cnn_model_start(the_vbx_cnn, object_model->model, object_model->model_io_buffers); 
		if(err != 0) return err;
		object_model->is_running = 1;
	}

	status = vbx_cnn_model_wfi(the_vbx_cnn); // Check if model done
	
	if(status < 0) {
		return status;
	} else if (status == 0) { // When  model is completed
	
		//Swap set of pipelined output buffers
		offset = (*PROCESSING_NEXT_FRAME_ADDRESS) - 0x70000000;
		object_model->model_input_buffer = (uint8_t*)(uintptr_t)(SCALER_FRAME_ADDRESS + offset);

		int num_outputs = model_get_num_outputs(object_model->model);
		for (int o = 0; o < num_outputs; o++) {			
			object_model->model_io_buffers[o+1] = (uintptr_t)object_model->pipelined_output_buffers[!object_model->buf_idx][o];
		}	
		//Start model inference
		
		err = vbx_cnn_model_start(the_vbx_cnn, object_model->model, object_model->model_io_buffers); 
		if(err != 0) return err;
		object_model->is_running = 1;
	
		
		gettimeofday(&m_run2, NULL);
		m_run_fps = 1000/ (gettimediff_us_2(m_run1, m_run2) / 1000);



#if PDMA
	vbx_cnn_io_ptr_t pdma_buffer[model_get_num_outputs(object_model->model)];
	int output_offset=0;
	for(int o =0; o<(int)model_get_num_outputs(object_model->model);o++){
		int output_length = model_get_output_length(object_model->model, o);
		pdma_ch_transfer(pdma_out,(void*)object_model->pipelined_output_buffers[object_model->buf_idx][o],output_offset,model_get_output_length(object_model->model, o),the_vbx_cnn,pdma_channel);
		pdma_buffer[o] = (vbx_cnn_io_ptr_t)(pdma_mmap_t + output_offset);
		output_offset+= output_length;
	}
	draw_post_process(object_model, fps,1,the_vbx_cnn,(vbx_cnn_io_ptr_t*)pdma_buffer); // Post process and draw previous 
#else
	draw_post_process(object_model, fps,1,the_vbx_cnn,(vbx_cnn_io_ptr_t*)object_model->pipelined_output_buffers[object_model->buf_idx]);
#endif	
	object_model->buf_idx = !object_model->buf_idx;
		
	}
	return status;
}