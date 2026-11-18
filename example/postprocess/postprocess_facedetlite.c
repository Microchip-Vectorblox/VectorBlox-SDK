#include "postprocess.h"
#include <stdio.h>

// Post-processing for the Qualcomm "Lightweight-Face-Detection" (face_det_lite)
// model.  C translation of the python detect() in
// vbx/postprocess/face_det_lite.py (itself a port of
// qai_hub_models.models.face_det_lite.utils).
//
// The model has a single detection stride of 8 and three output maps, each laid
// out channel-major as [channel][y][x] (NCHW with batch 1):
//   network_outputs[0] heatmap  : 1  channel  -- face center confidence (logits)
//   network_outputs[1] bbox     : 4  channels -- left/top/right/bottom offsets
//   network_outputs[2] landmark : 10 channels -- 5 (x,y) facial landmarks
//

#define FDL_STRIDE 8
#define FDL_MAX_PREDETECTS 64

// build a list of pixel indices above threshold, sorted by descending score
static int fdl_collect_candidates(const fix16_t* scores, int w, int h,
                                  fix16_t confidence_threshold, int order[], int max_order){
    int orderLength = 0;
    for(int y=0; y<h; y++){
        for(int x=0; x<w; x++){
            int n = y*w + x;
            if(scores[n] <= confidence_threshold) continue;

            int i=0;
            while(i<orderLength){ // find the insertion index (highest score first)
                if(scores[n] > scores[order[i]]){
                    int i_start = orderLength < max_order-1 ? orderLength : max_order-1;
                    for(int i2=i_start; i2>i; i2--) // move down all lower elements
                        order[i2] = order[i2-1];
                    order[i] = n;
                    if(orderLength < max_order) orderLength++;
                    break;
                }
                i++;
            }
            if(i==orderLength && orderLength<max_order){ // not inserted and room at the end
                order[i] = n;
                orderLength++;
            }
        }
    }
    return orderLength;
}

// decode the box+landmarks for the candidate at pixel (x,y) into faces[idx]
static void fdl_decode(object_t* face, int x, int y,
                       const fix16_t location[4], const fix16_t landmarks[10]){
    int stride = FDL_STRIDE;
    fix16_t anchX = fix16_from_int(x*stride);
    fix16_t anchY = fix16_from_int(y*stride);

    // [left, top, right, bottom] = ([cx,cy,cx,cy] + [-l,-t,r,b]) * stride
    fix16_t left   = anchX - location[0]*stride;
    fix16_t top    = anchY - location[1]*stride;
    fix16_t right  = anchX + location[2]*stride;
    fix16_t bottom = anchY + location[3]*stride;
    // normalize to [left, top, right, bottom]
    face->box[0] = left   < right  ? left   : right;
    face->box[1] = top    < bottom ? top    : bottom;
    face->box[2] = left   < right  ? right  : left;
    face->box[3] = top    < bottom ? bottom : top;

    // landmarks: channels are [x0..x4, y0..y4]; point = (offset + center) * stride
    for(int p=0; p<5; p++){
        face->points[p][0] = anchX + landmarks[p]   * stride;
        face->points[p][1] = anchY + landmarks[p+5] * stride;
    }
}

int post_process_facedetlite_int8(object_t faces[], int max_faces, int8_t *network_outputs[3],
                                  int zero_points[], fix16_t scale_outs[],
                                  int image_width, int image_height,
                                  fix16_t confidence_threshold, fix16_t nms_threshold){
    int w = image_width  / FDL_STRIDE;   // 80 for a 640 wide input
    int h = image_height / FDL_STRIDE;   // 60 for a 480 tall input
    int pixels = w*h;

    int8_t* hmMap   = network_outputs[0]; // 1  channel
    int8_t* locMap  = network_outputs[1]; // 4  channels
    int8_t* landMap = network_outputs[2]; // 10 channels

    fix16_t scores[pixels]; // sigmoid of the heatmap logits
    for(int n=0; n<pixels; n++){
        fix16_t logit = int8_to_fix16_single(hmMap[n], scale_outs[0], zero_points[0]);
        scores[n] = logit;// fix16_logistic_activate(logit);
    }

    int order[FDL_MAX_PREDETECTS];
    int orderLength = fdl_collect_candidates(scores, w, h, confidence_threshold, order, FDL_MAX_PREDETECTS);

    int facesLength = 0;
    for(int n=0; n<orderLength; n++){
        int ind = order[n];
        int y = ind / w;
        int x = ind - y*w;

        fix16_t location[4];
        for(int c=0; c<4; c++)
            location[c] = int8_to_fix16_single(locMap[c*pixels+ind], scale_outs[1], zero_points[1]);
        fix16_t landmarks[10];
        for(int c=0; c<10; c++)
            landmarks[c] = int8_to_fix16_single(landMap[c*pixels+ind], scale_outs[2], zero_points[2]);

        fdl_decode(&faces[facesLength], x, y, location, landmarks);
        faces[facesLength].detect_score = scores[ind];

        // NMS against already-kept (higher scoring) faces
        if(nms_threshold != -fix16_one){
            int passNms = 1;
            for(int f=0; f<facesLength; f++){
                if(calcIou_LTRB(faces[f].box, faces[facesLength].box) > nms_threshold){
                    passNms = 0;
                    break;
                }
            }
            if(!passNms)
                continue;
        }

        facesLength++;
        if(facesLength >= max_faces)
            break;
    }
    return facesLength;
}