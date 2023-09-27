#include "imageScaler/scaler.h"
#include "detectionDemo.h"
#include "frameDrawing/draw_assist.h"
#include "frameDrawing/draw.h"
#include <sys/time.h>

extern int update_Classifier;
extern volatile uint32_t*  PROCESSING_FRAME_ADDRESS;
extern volatile uint32_t*  PROCESSING_NEXT_FRAME_ADDRESS;
extern volatile uint32_t*  SCALER_BASE_ADDRESS;
extern volatile uint32_t*  SAVED_FRAME_SWAP;
extern int fps;
static inline void* virt_to_phys(vbx_cnn_t* vbx_cnn,void* virt){
	return (char*)(virt) + vbx_cnn->dma_phys_trans_offset;
}
extern uint32_t* loop_draw_frame;
static fix16_t fix16_two = F16(2);

// Globals Specification
const int topk = 4;
int16_t indexes[4] = {0};
int16_t display_index[4] = {0};
int32_t scores[4] = {0};
fix16_box boxes[100];
int valid_boxes = 0;
//const int CLASSIFIER_LABEL_FPS = 10;


void draw_post_process(struct model_descr_t *model, const int fps) {
	char label[256];
	const char* pptype = model->post_process_type;
	fix16_t** output_buffers = model->pipelined_output_buffers[model->buf_idx];

	// Run Post Processing of previous frame's outputs
	if (!strcmp(pptype, "YOLOV2") ||
			!strcmp(pptype, "YOLOV3") ||
			!strcmp(pptype, "YOLOV4") ||
			!strcmp(pptype, "YOLOV5") ||
			!strcmp(pptype, "SSDV2")){
		int max_boxes = 100;
		fix16_t thresh = F16(0.5);
		fix16_t iou = F16(0.4);
		int num_outputs;
		yolo_info_t cfg[3];
		fix16_t *outputs[3];
		char** class_names;
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
					.input_dims = {3, 416, 416},
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
					.input_dims = {3, 416, 416},
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
					.input_dims = {3, 416, 416},
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
					.input_dims = {3, 608, 608},
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
					.input_dims = {3, 608, 608},
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
					.input_dims = {3, 608, 608},
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
		else if(!strcmp(pptype, "YOLOV5")) {
			class_names = coco_classes;
			num_outputs = 3;
			int output_sizes[3] = {255*13*13, 255*26*26, 255*52*52};
			for (int o = 0; o < num_outputs; o++) {
				for (int i = 0; i < num_outputs; i++) {
					if (model_get_output_length(model->model,i) == output_sizes[o]) {
						outputs[o] = output_buffers[i];
					}
				}
			}

			static fix16_t anchors[] = {F16(10),F16(13),F16(16),F16(30),F16(33),F16(23),F16(30),F16(61),F16(62),F16(45),F16(59),F16(119),F16(116),F16(90),F16(156),F16(198),F16(373),F16(326)};
			static int mask_0[] = {6,7,8};
			static int mask_1[] = {3,4,5};
			static int mask_2[] = {0,1,2};

			yolo_info_t cfg_0 = {
				.version = 5,
				.input_dims = {3, 416, 416},
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
				.input_dims = {3, 416, 416},
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
				.input_dims = {3, 416, 416},
				.output_dims = {255, 52, 52},
				.coords = 4,
				.classes = 80,
				.num = 9,
				.anchors_length = 18,
				.anchors = anchors,
				.mask_length = 3,
				.mask = mask_2,
			};

			cfg[0] = cfg_0;
			cfg[1] = cfg_1;
			cfg[2] = cfg_2;
			valid_boxes = post_process_yolo(outputs, num_outputs, cfg, thresh, iou, boxes, max_boxes);
		}
		else if (!strcmp(pptype, "SSDV2")){
			fix16_t* ssdv2_outputs[12];
			char *is_vehicle = strstr(model->name, "vehicle");
			fix16_t confidence_threshold=F16(0.5);
			fix16_t nms_threshold=F16(0.4);
			if (is_vehicle) {
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
		int output_length = (int)model_get_output_length(model->model, 0);
		post_process_classifier(output_buffers[0],output_length,indexes,topk);
		if(update_Classifier)
		for(int i=0;i<topk;++i) {
			int idx = indexes[i];
			display_index[i]=indexes[i];
			scores[i] = output_buffers[0][idx];
		}
	} else if (!strcmp(pptype, "SCRFD")) {
		const int MAX_FACES=24;
		object_t faces[MAX_FACES];
		fix16_t confidence_threshold=F16(0.8);
		fix16_t nms_threshold=F16(0.4);

		//( 0 1 2 3 4 5 6 7 8)->(2 5 8 1 4 7 0 3 6)
		fix16_t *outputs[9];
		outputs[0]=(fix16_t*)(uintptr_t)output_buffers[2];
		outputs[1]=(fix16_t*)(uintptr_t)output_buffers[5];
		outputs[2]=(fix16_t*)(uintptr_t)output_buffers[8];
		outputs[3]=(fix16_t*)(uintptr_t)output_buffers[1];
		outputs[4]=(fix16_t*)(uintptr_t)output_buffers[4];
		outputs[5]=(fix16_t*)(uintptr_t)output_buffers[7];
		outputs[6]=(fix16_t*)(uintptr_t)output_buffers[0];
		outputs[7]=(fix16_t*)(uintptr_t)output_buffers[3];
		outputs[8]=(fix16_t*)(uintptr_t)output_buffers[6];


		int *input_dims = model_get_input_dims(model->model, 0);
		int image_w = input_dims[1];
		int image_h = input_dims[2];


		int length = post_process_scrfd(faces, MAX_FACES, outputs, image_w, image_h,
				confidence_threshold,nms_threshold);

		fix16_t hratio = fix16_div(fix16_from_int(1080),fix16_from_int(image_h));
		fix16_t wratio = fix16_div(fix16_from_int(1920),fix16_from_int(image_w));
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
						loop_draw_frame,2048,1080);
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
						loop_draw_frame,2048,1080);
			}
		}
	}

	//Draw outputs from last frame
	snprintf(label,sizeof(label),"%s %d fps",model->name, fps);
	draw_label(label,20,2,loop_draw_frame,2048,1080,WHITE);

	if (!strcmp(pptype, "YOLOV2") ||
			!strcmp(pptype, "YOLOV3") ||
			!strcmp(pptype, "YOLOV4") ||
			!strcmp(pptype, "YOLOV5") ||
			!strcmp(pptype, "SSDV2")) {
		int* input_length = model_get_input_dims(model->model,0);
		for(int i=0;i<valid_boxes;++i) {
			if(boxes[i].confidence == 0) {
				continue;
			}
			int x = boxes[i].xmin,y=boxes[i].ymin;
			int w = boxes[i].xmax-boxes[i].xmin;
			int h = boxes[i].ymax-boxes[i].ymin;
			x = x*1920/input_length[2];
			w = w*1920/input_length[2];
			y = y*1080/input_length[2];
			h = h*1080/input_length[2];
			if(x<0 || y<0 || w<=0 || h<=0) {
				continue;
			}
			draw_box(x,y,w,h,5,get_colour_modulo(boxes[i].class_id),
					loop_draw_frame,2048,1080);
			draw_label(boxes[i].class_name,x+5,y+5, loop_draw_frame,2048,1080,WHITE);
		}

	} else if(!strcmp(pptype, "IMAGENET")) {
		int idx, output_length;
		int32_t score;
		char* class_name;
		for(int i=0;i<topk;++i) {
			idx = display_index[i];
			score = scores[i];
			output_length = (int)model_get_output_length(model->model, 0);
			if(output_length == 1001 || output_length == 1000){ // imagenet
				class_name = imagenet_classes[idx];
				if(output_length==1001){
					//some imagenet networks have a null catagory, account for that
					class_name =  imagenet_classes[idx-1];
				}
			} 
			snprintf(label,sizeof(label),"%d %s %d%%",i,class_name,(score*100)>>16);
			draw_label(label,20,36+i*34,loop_draw_frame,2048,1080,WHITE);
		}
	}

}


short detectionDemoInit(vbx_cnn_t* the_vbx_cnn, struct model_descr_t* models, uint8_t modelIdx) {
	struct model_descr_t *object_model = models+modelIdx;
	size_t input_length     	= 0;
	size_t output_length     	= 0;

	object_model->buf_idx = 0;
	object_model->is_running = 0;
	object_model->model_io_buffers  = vbx_allocate_dma_buffer(the_vbx_cnn, (1+model_get_num_outputs(object_model->model))*sizeof(object_model->model_io_buffers), 0);
	if(!object_model->model_io_buffers){
		printf("Memory allocation issue for model io buffers.\n");
		return -1;
	}

	// Determine the input and output lengths of the Object Model, and re-order output buffers if applicable
	int num_outputs = model_get_num_outputs(object_model->model);
	for(int output = 0; output < num_outputs; output++) {
		output_length = model_get_output_length(object_model->model, output) * 4; // fix16 outputs are 4 bytes per element
		object_model->pipelined_output_buffers[0][output] = vbx_allocate_dma_buffer(the_vbx_cnn, output_length, 0);
		object_model->pipelined_output_buffers[1][output] = vbx_allocate_dma_buffer(the_vbx_cnn, output_length, 0);
		object_model->model_io_buffers[output+1] = (uintptr_t)object_model->pipelined_output_buffers[0][output];
		if(!object_model->pipelined_output_buffers[0][output] || !object_model->pipelined_output_buffers[1][output]){
			printf("Memory allocation issue for model output buffers.\n");
			return -1;	
		}
	}


	input_length = model_get_input_length(object_model->model, 0) *
		((int)model_get_input_datatype(object_model->model, 0) + 1);
	object_model->pipelined_input_buffer[0] = vbx_allocate_dma_buffer(the_vbx_cnn, input_length, 0);
	object_model->pipelined_input_buffer[1] = vbx_allocate_dma_buffer(the_vbx_cnn, input_length, 0);
	object_model->model_io_buffers[0] =(uintptr_t)object_model->pipelined_input_buffer[0];
	if(!object_model->pipelined_input_buffer[0] ||!object_model->pipelined_input_buffer[1]){
			printf("Memory allocation issue for model input buffers.\n");
			return -1;	
	}
	return 1;
}


int runDetectionDemo(struct model_descr_t* models, vbx_cnn_t* the_vbx_cnn, uint8_t modelIdx){
	int err, status;
	int screen_width = 1920;
	int screen_height = 1080;
	int screen_y_offset = 0;
	int screen_x_offset = 0;
	int screen_stride = 0x2000;

	struct model_descr_t *object_model = models+modelIdx;
	int* input_dims = model_get_input_dims(object_model->model, 0);

	//Start processing the network if not already running - 1st pass only (frame 0 )
	if(!object_model->is_running) {
		// Scale the current input frame
		resize_image_hls(SCALER_BASE_ADDRESS,(uint32_t*)(intptr_t)(*PROCESSING_FRAME_ADDRESS),
				screen_width, screen_height, screen_stride, 
				screen_x_offset, screen_y_offset,
				(uint8_t*)virt_to_phys(the_vbx_cnn, (void*)object_model->pipelined_input_buffer[object_model->buf_idx]),
				input_dims[2],input_dims[1]);

		err = vbx_cnn_model_start(the_vbx_cnn,object_model->model,object_model->model_io_buffers); // Start model
													  
		// Scaling the next input frame
		resize_image_hls_start(SCALER_BASE_ADDRESS,(uint32_t*)(intptr_t)(*PROCESSING_NEXT_FRAME_ADDRESS),
				screen_width, screen_height, screen_stride, 
				screen_x_offset, screen_y_offset,
				(uint8_t*)virt_to_phys(the_vbx_cnn, (void*)object_model->pipelined_input_buffer[!object_model->buf_idx]),
				input_dims[2], input_dims[1]);

		if(err != 0) return err;
		object_model->is_running = 1;

	}

	status = vbx_cnn_model_wfi(the_vbx_cnn); // Check if model done
	if(status < 0) {
		return status;
	} else if (status == 0) { // When  model is completed
		object_model->is_running = 0;

		// Swap which set of pipelined buffers is used as model IO
		object_model->model_io_buffers[0] = (uintptr_t)object_model->pipelined_input_buffer[!object_model->buf_idx];
		int num_outputs = model_get_num_outputs(object_model->model);
		for (int o = 0; o < num_outputs; o++) {
			object_model->model_io_buffers[o+1] = (uintptr_t)object_model->pipelined_output_buffers[!object_model->buf_idx][o];
		}

		// Wait for next frame scaling to finish, if it hasn't already
		resize_image_hls_wait(SCALER_BASE_ADDRESS);

		// Start model inference
		err = vbx_cnn_model_start(the_vbx_cnn,object_model->model,object_model->model_io_buffers);
		if(err != 0) return err;
		object_model->is_running = 1;

		// Scale the next input frame
		resize_image_hls_start(SCALER_BASE_ADDRESS,(uint32_t*)(intptr_t)(*PROCESSING_NEXT_FRAME_ADDRESS),
				screen_width, screen_height, screen_stride, 
				screen_x_offset, screen_y_offset,
				(uint8_t*)virt_to_phys(the_vbx_cnn, (void*)object_model->pipelined_input_buffer[object_model->buf_idx]),
				input_dims[2],input_dims[1]);

		draw_post_process(object_model, fps); // Post process and draw previous results
						    
		object_model->buf_idx = !object_model->buf_idx; // Toggle double-buffering (0->1 or 1->0)
	}

	return status;
}
