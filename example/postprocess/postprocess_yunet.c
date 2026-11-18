// YuNet (face_detection_yunet_2023mar) postprocess.
#include "postprocess.h"
#include <stdio.h>

// Note: this postprocessing script will not work without cutting the transpose and final sigmoid
// out of the model
// bbox is (dx, dy, log_w, log_h) about the cell center
// network_outputs is ordered cls_8/16/32, bbox_8/16/32, obj_8/16/32, kps_8/16/32.
// the score is found by sqrt(sigmoid(cls) * sigmoid(obj)).  obj is the discriminative term --

#define YUNET_MAX_PRE_DETECTS 32

static void yunet_map_geometry(int image_width, int image_height, int mapSizes[3][2], int mapPixels[3]){

    // feat map with stride = 32
    int h32 = image_height/32;  
    int w32 = image_width/32;   

    // feat map with stride = 16
    int h16 = h32*2, w16 = w32*2;

    // feat map with stride = 8
    int h8  = h16*2, w8  = w16*2;

    mapSizes[0][0] = h8;  mapSizes[0][1] = w8;
    mapSizes[1][0] = h16; mapSizes[1][1] = w16;
    mapSizes[2][0] = h32; mapSizes[2][1] = w32;
    
    mapPixels[0] = h8*w8;
    mapPixels[1] = h16*w16;
    mapPixels[2] = h32*w32;
}

// Returns the new length of order and inserts new element into
// while maintain descending order in the order array 
static int yunet_insert_ordered(int *order, int orderLength, fix16_t *scores, int n){
    int i = 0;
    while(i < orderLength) {

        if(scores[n] > scores[order[i]]){

            int i_start = orderLength < YUNET_MAX_PRE_DETECTS-1 ? orderLength : YUNET_MAX_PRE_DETECTS-1;
            for(int i2 = i_start; i2 > i; i2--)   // move down all lower elements
                order[i2] = order[i2-1];
            order[i] = n;

            if(orderLength < YUNET_MAX_PRE_DETECTS) {
                orderLength++;
            } 
            return orderLength;
        }
        i++;
    }

    if(i == orderLength && orderLength < YUNET_MAX_PRE_DETECTS){   // not inserted, room at the end
        order[i] = n;
        orderLength++;
    }
    return orderLength;
}




int post_process_yunet_int8(object_t faces[],int max_faces, int8_t *network_outputs[12],int zero_points[], fix16_t scale_outs[],
                            int image_width, int image_height, fix16_t confidence_threshold, fix16_t nms_threshold) {

    const int mapStrides[3] = {8,16,32};

    int mapSizes[3][2];
    int mapPixels[3];

    // Calculate pixel maps and store in
    yunet_map_geometry(image_width, image_height, mapSizes, mapPixels);

    int8_t** class_maps  = &network_outputs[0];    
    int8_t** location_maps  = &network_outputs[3]; // holds raw bounding box predictions with three buffers for each stride
    int8_t** object_maps  = &network_outputs[6];
    int8_t** landmark_maps = &network_outputs[9];

    int scoresLength = mapPixels[0]+mapPixels[1]+mapPixels[2];
    fix16_t scores[scoresLength];

    int s = 0;  // index to scores
    for (int i = 0; i < 3; i++) {
        int8_t* cls_map = class_maps[i];
        int8_t* obj_map = object_maps[i];

        int pixels = mapPixels[i];
        for (int n = 0; n < pixels; n++) {

            fix16_t cls_fix16 = int8_to_fix16_single(cls_map[n], scale_outs[i], zero_points[i]);
            fix16_t cls = fix16_logistic_activate(cls_fix16); // Sigmoid activation function that was cut from graph

            fix16_t obj_fix16 = int8_to_fix16_single(obj_map[n], scale_outs[6+i], zero_points[6+i]);  // offset of 6 for scale & zero point for object location
            fix16_t obj = fix16_logistic_activate(obj_fix16);

            scores[s++] = fix16_sqrt(fix16_mul(cls,obj));

        }
    }

    // add scores above the cutoff threshold into order. Order is a array
    // of a sorted list of indexes into scores
    int order[YUNET_MAX_PRE_DETECTS];
    int order_length = 0;

    for (int n = 0; n < scoresLength; n++) {
        if (scores[n] > confidence_threshold){
            order_length = yunet_insert_ordered(order, order_length, scores, n);
        }
    }

    int faces_length = 0;
    int map_num = 0;
    int local_index;


    for(int n = 0; n < order_length; n++) {
        int scores_index = order[n];

        faces[faces_length].detect_score = scores[scores_index];

        if (scores_index < mapPixels[0]) {                  // in map 0 (stride 8)
            map_num = 0;
            local_index = scores_index;
        }
        else if (scores_index < mapPixels[0] + mapPixels[1]) {  // in map 1 (stride 16)
            map_num = 1;
            local_index = scores_index - mapPixels[0];
        }
        else {                                               // in map 2 (stride 32)
            map_num = 2;
            local_index = scores_index - mapPixels[0] - mapPixels[1];
        }

        int pixels = mapPixels[map_num];
        int y = local_index / mapSizes[map_num][1];
        int x = local_index - y * mapSizes[map_num][1];
        int stride = mapStrides[map_num];
        fix16_t fix16_stride = fix16_from_int(stride);
        fix16_t priorX = fix16_from_int(x*stride); // pixel map center x point
        fix16_t priorY = fix16_from_int(y*stride);  // pixel map center y point


        // bbox: centre offset in cells, size as log-scale multiples of stride
        // TODO: make this section more clear in the future, currently hard to read
        int8_t* locPtr = &location_maps[map_num][local_index];
        fix16_t r0 = int8_to_fix16_single(*locPtr,scale_outs[3+map_num],zero_points[3+map_num]); locPtr += pixels;
        fix16_t r1 = int8_to_fix16_single(*locPtr,scale_outs[3+map_num],zero_points[3+map_num]); locPtr += pixels;
        fix16_t r2 = int8_to_fix16_single(*locPtr,scale_outs[3+map_num],zero_points[3+map_num]); locPtr += pixels;
        fix16_t r3 = int8_to_fix16_single(*locPtr,scale_outs[3+map_num],zero_points[3+map_num]);

        fix16_t center_x = priorX + fix16_mul(r0, fix16_stride);  // center x coord of bounding box
        fix16_t center_y = priorY + fix16_mul(r1, fix16_stride);
        
        fix16_t w = fix16_mul(fix16_exp(r2), fix16_stride);
        fix16_t h = fix16_mul(fix16_exp(r3), fix16_stride);

        // [left, top, right, bottom]
        faces[faces_length].box[0] = center_x - w/2; //left X coordinates 
        faces[faces_length].box[1] = center_y - h/2; //top Y coord
        faces[faces_length].box[2] = center_x + w/2; //right X coord
        faces[faces_length].box[3] = center_y + h/2; //bottom Y coord

        // NMS
        int passNms = 1;
        for (int f = 0; f < faces_length; f++) {
            if (calcIou_LTRB(faces[f].box, faces[faces_length].box) > nms_threshold) {
                passNms = 0;
                break;
            }
        }
        if(!passNms){
            continue;
        }

         // landmarks: elements are every "pixels" elements
        fix16_t landmark_scale = scale_outs[9 + map_num]; // offset of 9 for scale & zero point for object location
        int landmark_zero = zero_points[9 + map_num];

        for (int p = 0; p < 5; p++) {
            int8_t rawX = landmark_maps[map_num][local_index + (2*p)     * pixels];
            int8_t rawY = landmark_maps[map_num][local_index + (2*p + 1) * pixels];

            fix16_t dx = int8_to_fix16_single(rawX, landmark_scale, landmark_zero);
            fix16_t dy = int8_to_fix16_single(rawY, landmark_scale, landmark_zero);

            faces[faces_length].points[p][0] = priorX + fix16_mul(dx, fix16_stride); 
            faces[faces_length].points[p][1] = priorY + fix16_mul(dy, fix16_stride);
        }



        faces_length++;
        if(faces_length >= max_faces){
            break;
        }
           
    }

    return faces_length;
}
