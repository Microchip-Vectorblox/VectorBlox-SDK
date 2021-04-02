#include "postprocess.h"
#include "vbx_cnn_api.h"
#include <stdio.h>
#include <string.h>
#include <string>
#include <sys/time.h>

#define debug(a) printf("%s:%d %s=%d\n",__FILE__,__LINE__,#a,(int)(uintptr_t)(a))
#define debugx(a) printf("%s:%d %s=0x%08x\n",__FILE__,__LINE__,#a,(unsigned)(uintptr_t)(a))
#define debugp(a) printf("%s:%d %s=%p\n",__FILE__,__LINE__,#a,(void*)(a))
#define debugs(a) printf("%s:%d %s=%s\n",__FILE__,__LINE__,#a,(a))

extern "C" int read_JPEG_file(const char *filename, int *width, int *height,
                              unsigned char **image);
extern "C" int resize_image(uint8_t *image_in, int in_w, int in_h,
                            uint8_t *image_out, int out_w, int out_h);
void *read_image(const char *filename, int size) {
  unsigned char *image;
  int h, w;
  read_JPEG_file(filename, &w, &h, &image);
  unsigned char *planer_img = (unsigned char *)malloc(w * h * 3);
  for (int r = 0; r < h; r++) {
    for (int c = 0; c < w; c++) {
      planer_img[r * w + c] = image[(r * w + c) * 3 + 2];
      planer_img[w * h + r * w + c] = image[(r * w + c) * 3 + 1];
      planer_img[2 * w * h + r * w + c] = image[(r * w + c) * 3 + 0];
    }
  }

  free(image);
  // return planer_img;
  unsigned char *resized_img = (unsigned char *)malloc(size * size * 3);
  resize_image((uint8_t *)planer_img, w, h, (uint8_t *)resized_img, size, size);
  resize_image((uint8_t *)planer_img + w * h, w, h,
               (uint8_t *)resized_img + size * size, size, size);
  resize_image((uint8_t *)planer_img + 2 * w * h, w, h,
               (uint8_t *)resized_img + 2 * size * size, size, size);

  free(planer_img);
  return resized_img;
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

// Not an official API
void check_debug_prints(vbx_cnn_t *vbx_cnn) {
  char debug_str[32];
  while (1) {
    int nread = vbx_cnn_get_debug_prints(vbx_cnn, debug_str, 32);
    for (int c = 0; c < nread; ++c) {
      putchar(debug_str[c]);
    }
    if (nread == 0) {
      return;
    }
  }
}
int gettimediff_us(struct timeval start, struct timeval end) {

  int sec = end.tv_sec - start.tv_sec;
  int usec = end.tv_usec - start.tv_usec;
  return sec * 1000000 + usec;
}
int main(int argc, char **argv) {

  if (argc < 3) {
    printf("Usage %s FIRMWARE.bin Model.vnnx [IMAGE.jpeg] "
           "[CLASSIFY|YOLOV2|TINYYOLOV2)\n",
           argv[0]);
    return 1;
  }
  void *firmware_instr = read_firmware_file(argv[1]);
  if (!firmware_instr) {
    fprintf(stderr, "Unable to correctly read %s. Exiting\n", argv[1]);
    exit(1);
  }
  vbx_cnn_t *vbx_cnn = vbx_cnn_init(NULL, firmware_instr);
  if (!vbx_cnn) {
    fprintf(stderr, "Unable to initialize vbx_cnn. Exiting\n");
    exit(1);
  }
  model_t *model = read_model_file(vbx_cnn, argv[2]);
  if (!model) {
    fprintf(stderr, "Unable to correctly read %s. Exiting\n", argv[2]);
    exit(1);
  }
  if (model_check_sanity(model) != 0) {
    printf("Model %s is not sane\n", argv[2]);
  };

  uint8_t *input_buffer = (uint8_t *)vbx_allocate_dma_buffer(
      vbx_cnn, model_get_input_length(model, 0) * sizeof(uint8_t), 1);
  int input_length = model_get_input_length(model, 0);
  if (argc > 3) {
    int side = 1;
    while (side * side * 3 < input_length)
      side++;
    printf("Reading %s\n", argv[3]);
    void *read_buffer = read_image(argv[3], side);
    memcpy(input_buffer, read_buffer, input_length);
  } else {
    memcpy(input_buffer, (uint8_t *)model_get_test_input(model, 0),
           input_length);
  }
  std::string post_process_str(argc > 4 ? argv[4] : "");

  vbx_cnn_io_ptr_t io_buffers[3];
  io_buffers[0] = (vbx_cnn_io_ptr_t)input_buffer;
  for (unsigned o = 0; o < model_get_num_outputs(model); ++o) {
    io_buffers[1 + o] = (vbx_cnn_io_ptr_t)vbx_allocate_dma_buffer(
        vbx_cnn, model_get_output_length(model, o) * sizeof(uint32_t), 0);
  }
  fix16_t *output_buffers[2] = {(fix16_t *)io_buffers[1],
                                (fix16_t *)io_buffers[2]};
  struct timeval tv1, tv2;
  gettimeofday(&tv1, NULL);
  vbx_cnn_model_start(vbx_cnn, model, io_buffers);
  int pol_val;
  while ((pol_val = vbx_cnn_model_poll(vbx_cnn)) > 0) {
    check_debug_prints(vbx_cnn);
    gettimeofday(&tv2, NULL);
    if (gettimediff_us(tv1, tv2) > 60E6) {
      fprintf(stderr, "ERROR: Network taking more than 10s. ABORTING\n");
      exit(-1);
    }
  }
  printf("Done model\n");
  if (pol_val < 0) {
    printf("Model failed with error %d\n", vbx_cnn_get_error_val(vbx_cnn));
  }
  check_debug_prints(vbx_cnn);

  printf("network took %d ms\n", gettimediff_us(tv1, tv2) / 1000);

  if (post_process_str == "CLASSIFY") {
    const int topk = 10;
    int16_t indices[topk];
    int output_length = model_get_output_length(model, 0);
    post_process_classifier(output_buffers[0], output_length, indices, topk);
    for (int i = 0; i < topk; ++i) {
      int idx = indices[i];
      int score = output_buffers[0][idx];
      printf("%d, %d, %s, %d.%03d\n", i, idx, imagenet_classes[idx],
             score >> 16, (score * 1000) >> 16);
    }
  } else if (post_process_str == "TINYYOLOV2" || post_process_str == "YOLOV2" ||
             post_process_str == "TINYYOLOV3") {
    char **class_names = NULL;
    int valid_boxes = 0;
    fix16_box boxes[1024];
    int max_boxes = 100;
    float thresh = 0.3;
    float iou = 0.4;

    if (post_process_str == "TINYYOLOV2") { // tiny yolo v2 VOC
      class_names = voc_classes;
      int num_outputs = 1;
      fix16_t *outputs[] = {output_buffers[0]};
      float anchors[] = {1.08,  1.19, 3.42, 4.41,      6.63,
                         11.38, 9.42, 5.11, 16.620001, 10.52};

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

      valid_boxes = post_process_yolo(outputs, num_outputs, cfg, thresh, iou,
                                      boxes, max_boxes);

    }
    else if (post_process_str == "TINYYOLOV2") { // yolo v2 VOC
      class_names = voc_classes;
      int num_outputs = 1;
      fix16_t *outputs[] = {output_buffers[0]};
      float anchors[] = {1.3221,  1.73145, 3.19275, 4.00944, 5.05587,
                         8.09892, 9.47112, 4.84053, 11.2364, 10.0071};

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

      valid_boxes = post_process_yolo(outputs, num_outputs, cfg, thresh, iou,
                                      boxes, max_boxes);

    }
    else if (post_process_str == "TINYYOLOV3") { // tiny yolo v3 COCO
      class_names = coco_classes;
      int num_outputs = 2;
      fix16_t *outputs[] = {output_buffers[0], output_buffers[1]};
      float anchors[] = {10, 14, 23,  27,  37,  58,
                         81, 82, 135, 169, 344, 319}; // 2*num
      int mask_0[] = {3, 4, 5};
      int mask_1[] = {1, 2, 3};

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

      valid_boxes = post_process_yolo(outputs, num_outputs, cfg, thresh, iou,
                                      boxes, max_boxes);
    }

    char class_str[50];
    for (int i = 0; i < valid_boxes; ++i) {
      if (boxes[i].confidence == 0) {
        continue;
      }

      if (class_names) { // class_names must be set, or prints the class id
        boxes[i].class_name = class_names[boxes[i].class_id];
        sprintf(class_str, "%s", boxes[i].class_name);
      } else {
        sprintf(class_str, "%d", boxes[i].class_id);
      }

      printf("%s\t%.2f\t(%d, %d, %d, %d)\n", class_str,
             fix16_to_float(boxes[i].confidence), boxes[i].xmin, boxes[i].xmax,
             boxes[i].ymin, boxes[i].ymax);
    }
  } else if (post_process_str == "") {
  } else {
    printf("Unknown post processing type %s, skipping post process\n",
           post_process_str.c_str());
  }

  return 0;
}
