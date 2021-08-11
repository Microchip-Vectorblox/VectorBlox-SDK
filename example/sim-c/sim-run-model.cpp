#include <stdio.h>
#include <string>
#include "vbx_cnn_api.h"
#include "postprocess.h"

extern "C" int read_JPEG_file (const char * filename,int* width,int* height,unsigned char **image);
extern "C" int resize_image(uint8_t* image_in,int in_w,int in_h,
				 uint8_t* image_out,int out_w,int out_h);
void* read_image(const char* filename,int size,int data_type){
  unsigned char* image;
  int h,w;
  read_JPEG_file (filename,&w,&h,&image);
  unsigned char* planer_img = (unsigned char*)malloc(w*h*3);
  for(int r=0;r<h;r++){
	for(int c=0;c<w;c++){
	  planer_img[r*w+c] = image[(r*w+c)*3+2];
	  planer_img[w*h+r*w+c] = image[(r*w+c)*3+1];
	  planer_img[2*w*h+r*w+c] = image[(r*w+c)*3+0];
	}
  }

  free(image);
  //return planer_img;
  unsigned char* resized_img = (unsigned char*)malloc(size*size*3);
  resize_image((uint8_t*)planer_img  ,w,h,
			   (uint8_t*)resized_img,size,size);
  resize_image((uint8_t*)planer_img  +w*h,w,h,
			   (uint8_t*)resized_img+size*size,size,size);
  resize_image((uint8_t*)planer_img  +2*w*h,w,h,
			   (uint8_t*)resized_img+2*size*size,size,size);

  free(planer_img);
  return resized_img;

}




static uint32_t fletcher32(const uint16_t *data, size_t len)
{
        uint32_t c0, c1;
        unsigned int i;

        for (c0 = c1 = 0; len >= 360; len -= 360) {
                for (i = 0; i < 360; ++i) {
                        c0 = c0 + *data++;
                        c1 = c1 + c0;
                }
                c0 = c0 % 65535;
                c1 = c1 % 65535;
        }
        for (i = 0; i < len; ++i) {
                c0 = c0 + *data++;
                c1 = c1 + c0;
        }
        c0 = c0 % 65535;
        c1 = c1 % 65535;
        return (c1 << 16 | c0);
}

int main(int argc, char** argv){

	//On hardware these two variables would be set with real values
	//because this is for the simulator, we use NULL
	void* ctrl_reg_addr = NULL;
	void* firmware_blob = NULL;
	vbx_cnn_t* vbx_cnn = vbx_cnn_init(ctrl_reg_addr,firmware_blob);

	if(argc < 4){
		fprintf(stderr,
                "Usage: %s MODEL_FILE IMAGE.jpg NETWORK_TYPE\n"
                "   NETWORK_TYPE controls what post processing routines to use\n"
                "      must be one of CLASSIFY, YOLOV2, TINYYOLOV2, TINYYOLOV3, BLAZEFACE,RETINAFACE, or SSDV2\n",
                argv[0]);
		return 1;
	}

    std::string post_process_str(argv[3]);
	FILE* model_file = fopen(argv[1],"r");
	if(model_file == NULL){
		printf("Unable to open file %s\n", argv[1]);
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
		fprintf(stderr,"Error model file is not correct size%s\n",argv[1]);
	}
	int model_allocate_size = model_get_allocate_bytes(model);
	model = (model_t*)realloc(model,model_allocate_size);
	//For this example we are assuming we only have
	//one input and one output buffer for the models,
	//(or two output buffers for yolo v3)
	uint8_t* input_buffer=NULL;
	void* read_buffer=NULL;
	if(std::string(argv[2]) != "TEST_DATA"){
	  int input_datatype = model_get_input_datatype(model,0);
	  int input_length = model_get_input_length(model,0);
	  int side = 1;
	  while(side*side*3 < input_length)side++;
	  read_buffer = read_image(argv[2],side,input_datatype);
	  input_buffer = (uint8_t*)read_buffer;
	}else{
	  input_buffer = (uint8_t*)model_get_test_input(model,0);
	}
    vbx_cnn_io_ptr_t* io_buffers= (vbx_cnn_io_ptr_t*)malloc(sizeof(vbx_cnn_io_ptr_t)*(1+model_get_num_outputs(model)));
    io_buffers[0] = (uintptr_t)input_buffer;
    for (unsigned o = 0; o < model_get_num_outputs(model); ++o) {
      int output_length = model_get_output_length(model, o);
	  io_buffers[o+1] = (uintptr_t)malloc(output_length*sizeof(fix16_t));
    }

	//buffers are now setup,
	//we can run the model.

	vbx_cnn_model_start(vbx_cnn, model, io_buffers);
	int err=1;
	while (err>0) {
		err = vbx_cnn_model_poll(vbx_cnn);
	}
	if (err<0) {
		printf("Model Run failed with error code: %d\n",err);
	}
	//data should be available int the output buffers now.
    if (post_process_str == "BLAZEFACE"){
        const int MAX_FACES=24;
        face_t faces[MAX_FACES];
        fix16_t* output_buffer0=(fix16_t*)(uintptr_t)io_buffers[1];
        fix16_t* output_buffer1=(fix16_t*)(uintptr_t)io_buffers[2];
        int output_length0 = model_get_output_length(model, 0);
        int output_length1 = model_get_output_length(model, 1);

	int facesLength = 0;
	if (output_length0 < output_length1) {
		facesLength = post_process_blazeface(faces,output_buffer0,output_buffer1,output_length0,
				MAX_FACES,fix16_from_int(1));
	} else {
		facesLength = post_process_blazeface(faces,output_buffer1,output_buffer0,output_length1,
				MAX_FACES,fix16_from_int(1));
	}
        for(int f=0;f<facesLength;f++){
            face_t* face = faces+f;
            fix16_t x = face->box[0];
            fix16_t y = face->box[1];
            fix16_t w = face->box[2] - face->box[0];
            fix16_t h = face->box[3] - face->box[1];
            printf("face %d found at (x,y,w,h) %f %f %f %f\n",f,
                   fix16_to_float(x), fix16_to_float(y),
                   fix16_to_float(w), fix16_to_float(h));
        }
    }else if (post_process_str=="RETINAFACE"){
        const int MAX_FACES=24;
        face_t faces[MAX_FACES];
        fix16_t confidence_threshold=fix16_from_float(0.95);
        fix16_t nms_threshold=fix16_from_float(0.4);

        fix16_t* output_buffers[9];
        for(int o=0;o<9;++o){
            output_buffers[o]=(fix16_t*)(uintptr_t)io_buffers[1+o];
        }
        int input_length = model_get_input_length(model,0);
	int size = 640;
	if (input_length == (3*320*320)) size = 320;
        int facesLength = post_process_retinaface(faces,MAX_FACES,output_buffers, size, size,
                                                  confidence_threshold,nms_threshold);

        for(int f=0;f<facesLength;f++){
            face_t* face = faces+f;
            fix16_t x = face->box[0];
            fix16_t y = face->box[1];
            fix16_t w = face->box[2] - face->box[0];
            fix16_t h = face->box[3] - face->box[1];
            printf("face %d found at (x,y,w,h) %d %d %d %d\n",f,
                   fix16_to_int(x), fix16_to_int(y),
                   fix16_to_int(w), fix16_to_int(h));
            printf("landmarks: ");
            for(int l =0;l<5;++l){
                printf("%d,%d ",
                       fix16_to_int(face->points[l][0]),
                       fix16_to_int(face->points[l][1]));
                fflush(stdout);
            }printf("\n");


        }
    }else if (post_process_str=="CLASSIFY") {
	  const int topk=10;
	  int16_t indices[topk];
      int output_length = model_get_output_length(model, 0);
      fix16_t* output_buffer0=(fix16_t*)(uintptr_t)io_buffers[1];
	  post_process_classifier(output_buffer0,output_length,indices,topk);
	  for(int i = 0;i < topk; ++i){
		int idx = indices[i];
        char* class_name = imagenet_classes[idx];
        if(output_length==1001){
          //some imagenet networks have a null catagory, account for that
          class_name =  imagenet_classes[idx-1];
        }
		int score = output_buffer0[idx];
		printf("%d, %d, %s, %d.%03d\n", i, idx,class_name, score>>16, (score*1000)>>16);
	  }
	} else if (post_process_str == "TINYYOLOV2" || post_process_str == "YOLOV2" || post_process_str == "TINYYOLOV3" || post_process_str == "SSDV2"){
		char **class_names = NULL;
		int valid_boxes = 0;
		fix16_box boxes[1024];
		int max_boxes = 100;
		float thresh = 0.3;
		float iou = 0.4;

		if(post_process_str == "TINYYOLOV2"){ //tiny yolo v2 VOC
			class_names = voc_classes;
			int num_outputs = 1;
			fix16_t *outputs[] = {(fix16_t*)io_buffers[1]};
			float anchors[] ={1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.620001, 10.52};

			yolo_info_t cfg_0 = {
				.version = 2,
				.input_dims = {3, 416, 416},
				.output_dims = {125, 13, 13},
				.coords = 4,
				.classes = 20,
				.num = 5,
				.anchors_length = 10,
				.anchors = anchors,
			};
			yolo_info_t cfg[] = {cfg_0};

			valid_boxes = post_process_yolo(outputs, num_outputs, cfg, thresh, iou, boxes, max_boxes);

		} else if (post_process_str == "YOLOV2"){ //yolo v2 VOC
			class_names = voc_classes;
			int num_outputs = 1;
            fix16_t *outputs[] = {(fix16_t*)io_buffers[1]};
			float anchors[] = {1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071};

			yolo_info_t cfg_0 = {
				.version = 2,
				.input_dims = {3, 416, 416},
				.output_dims = {125, 13, 13},
				.coords = 4,
				.classes = 20,
				.num = 5,
				.anchors_length = 10,
				.anchors = anchors,
			};
			yolo_info_t cfg[] = {cfg_0};

			valid_boxes = post_process_yolo(outputs, num_outputs, cfg, thresh, iou, boxes, max_boxes);

		} else if (post_process_str == "TINYYOLOV3"){ //tiny yolo v3 COCO
			class_names = coco_classes;
			int num_outputs = 2;
			fix16_t *outputs[] = {(fix16_t*)io_buffers[1],
                                  (fix16_t*)io_buffers[2]};
			float anchors[] = {10,14,23,27,37,58,81,82,135,169,344,319}; // 2*num
			int mask_0[] = {3,4,5};
			int mask_1[] = {1,2,3};

			yolo_info_t cfg_0 = {
				.version = 3,
				.input_dims = {3, 416, 416},
				.output_dims = {255, 13, 13},
				.coords = 4,
				.classes = 80,
				.num = 6,
				.anchors_length = 12,
				.anchors = anchors,
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
				.anchors = anchors,
				.mask_length = 3,
				.mask = mask_1,
			};

			yolo_info_t cfg[] = {cfg_0, cfg_1};

			valid_boxes = post_process_yolo(outputs, num_outputs, cfg, thresh, iou, boxes, max_boxes);
		}
        else if (post_process_str == "SSDV2"){
        fix16_t* output_buffers[12] = {(fix16_t*)io_buffers[1],
                                       (fix16_t*)io_buffers[2],
                                       (fix16_t*)io_buffers[3],
                                       (fix16_t*)io_buffers[4],
                                       (fix16_t*)io_buffers[5],
                                       (fix16_t*)io_buffers[6],
                                       (fix16_t*)io_buffers[7],
                                       (fix16_t*)io_buffers[8],
                                       (fix16_t*)io_buffers[9],
                                       (fix16_t*)io_buffers[10],
                                       (fix16_t*)io_buffers[11],
                                       (fix16_t*)io_buffers[12]
        };
        fix16_t confidence_threshold=fix16_from_float(0.6);
        fix16_t nms_threshold=fix16_from_float(0.5);
        valid_boxes = post_process_ssdv2(boxes,max_boxes,output_buffers,91,confidence_threshold,nms_threshold);
        class_names = coco91_classes;
    }

		char class_str[50];
		for(int i=0;i<valid_boxes;++i){
			if(boxes[i].confidence == 0){
				continue;
			}

			if (class_names) { //class_names must be set, or prints the class id
				boxes[i].class_name = class_names[boxes[i].class_id];
				sprintf(class_str, "%s", boxes[i].class_name);
			} else {
				sprintf(class_str, "%d", boxes[i].class_id);
			}

			printf("%s\t%.2f\t(%d, %d, %d, %d)\n",
					class_str,
					fix16_to_float(boxes[i].confidence),
					boxes[i].xmin,boxes[i].xmax,
					boxes[i].ymin,boxes[i].ymax);
		}
	} else {
		printf("Unknown post processing type %s, skipping post process\n",
				post_process_str.c_str());
	}

	int32_t checksum = fletcher32((uint16_t*)io_buffers[1],  model_get_output_length(model, 0));
    for(unsigned o =1;o<model_get_num_outputs(model);++o){
	  checksum ^= fletcher32((uint16_t*)io_buffers[1+o], model_get_output_length(model, o)*2);
    }
	printf("CHECKSUM = 0x%08x\n",checksum);
	if (read_buffer) {
		free(read_buffer);
	}
	free(model);

	return 0;
}
