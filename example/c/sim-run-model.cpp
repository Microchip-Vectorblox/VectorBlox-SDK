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
	volatile void* ctrl_reg_addr = NULL;
	void* firmware_blob = NULL;
	vbx_cnn_t* vbx_cnn = vbx_cnn_init(ctrl_reg_addr,firmware_blob);

	if(argc < 4){
		printf("Usage %s MODEL_FILE IMAGE.jpg (CLASSIFY|YOLOV2|TINYYOLOV2|TINYYOLOV3)\n",argv[0]);
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
	if(argc >2){
	  int input_datatype = model_get_input_datatype(model,0);
	  int input_length = model_get_input_length(model,0);
	  int side = 1;
	  while(side*side*3 < input_length)side++;
	  read_buffer = read_image(argv[2],side,input_datatype);
	  input_buffer = (uint8_t*)read_buffer;
	}else{
	  input_buffer = (uint8_t*)model_get_test_input(model,0);
	}
	int output_length = model_get_output_length(model,0);
	int output_length1=0;
	fix16_t* output_buffer0 = (fix16_t*)malloc(output_length*sizeof(fix16_t));
	fix16_t* output_buffer1 = NULL;
	if(model_get_num_outputs(model)==2){
		output_length1 = model_get_output_length(model,1);
		output_buffer1 = (fix16_t*)malloc(output_length1*sizeof(fix16_t));
	}

	vbx_cnn_io_ptr_t io_buffers[3] = {(uintptr_t)input_buffer,
	                                  (uintptr_t)output_buffer0,
					  (uintptr_t)output_buffer1};
	//buffers are now setup,
	//we can run the model.

	vbx_cnn_model_start(vbx_cnn,model,io_buffers);
	int err=1;
	while(err>0){
		err = vbx_cnn_model_poll(vbx_cnn);
	}
	if(err<0){
		printf("Model Run failed with error code: %d\n",err);
	}
    //data should be available int the output buffers now.
	if(post_process_str=="CLASSIFY"){
	  const int topk=10;
	  int16_t indices[topk];
	  post_process_classifier(output_buffer0,output_length,indices,topk);
	  for(int i=0;i<topk;++i){
		int idx = indices[i];
		int score = output_buffer0[idx];
		printf("%d, %d, %s, %d.%03d\n",
			   i,idx,imagenet_classes[idx],score>>16,(score*1000)>>16);
	  }
	} else if(post_process_str == "TINYYOLOV2" || post_process_str == "YOLOV2"){
      // tiny yolov2 voc
		int valid_boxes;
		fix16_box boxes[13*13*5];
        if(post_process_str == "TINYYOLOV2"){
          post_process_tiny_yolov2_voc(output_buffer0, &valid_boxes, boxes,13*13*5);
        }else{
          post_process_yolov2_voc(output_buffer0, &valid_boxes, boxes,13*13*5);
        }
		for(int i=0;i<valid_boxes;++i){
			if(boxes[i].confidence == 0){
				continue;
			}
			printf("%s %.2f box:(%d,%d) (%d,%d)\n",
			       boxes[i].class_name,
			       fix16_to_float(boxes[i].confidence),
			       boxes[i].xmin,boxes[i].ymin,
			       boxes[i].xmax,boxes[i].ymax);
		}
	} else if(post_process_str == "TINYYOLOV3"){
		int valid_boxes;
		fix16_box boxes[100];
		post_process_tiny_yolov3_coco(output_buffer0, output_buffer1, &valid_boxes, boxes,100);
		for(int i=0;i<valid_boxes;++i){
			if(boxes[i].confidence == 0){
				continue;
			}
			printf("%s %.2f box:(%d,%d) (%d,%d)\n",
			       boxes[i].class_name,
			       fix16_to_float(boxes[i].confidence),
			       boxes[i].xmin,boxes[i].ymin,
			       boxes[i].xmax,boxes[i].ymax);
		}
	}else{
      printf("Unknown post processing type %s, skipping post process\n",
             post_process_str.c_str());
    }

	int32_t checksum = fletcher32((uint16_t*)io_buffers[1], output_length*2);
	if(io_buffers[2]){
	  checksum ^= fletcher32((uint16_t*)io_buffers[2], output_length1*2);
	}
	printf("CHECKSUM = 0x%08x\n",checksum);
	if(read_buffer){
		free(read_buffer);
	}
	free(model);



	return 0;

}
