#include "postprocess.h"
#include <stdio.h>


int post_process_scrfd_int8(object_t faces[],int max_faces, int8_t *network_outputs[9],int zero_points[], fix16_t scale_outs[], 
                            int image_width, int image_height,
                            fix16_t confidence_threshold, fix16_t nms_threshold, model_t *model){
    const int mapStrides[3] = {8,16,32};
    const int maxPreDetects = 64;
    
    int h32 = image_height/32;  // image height at stride=32
    int w32 = image_width/32;  // image width at stride=32
    int h16 = h32<<1;
    int w16 = w32<<1;
    int h8 = h16<<1;
    int w8 = w16<<1;
    int mapSizes[3][2] = {{h8,w8},{h16,w16},{h32,w32}};
    int mapPixels[3] = {h8*w8, h16*w16, h32*w32};
    // each map shape is [anchor][channel][y][x]
    //   there are 2 anchors per pixel
    //   location map has 4 channels; confidence has 2 channels; landmarks has 10 channels
    //   the range of x and y pixel are given by mapSizes[...]
    int8_t** confMaps = &network_outputs[0];
    int8_t** locMaps = &network_outputs[3];
    int8_t** landMaps = &network_outputs[6];

    int scoresLength = 2*(mapPixels[0]+mapPixels[1]+mapPixels[2]);
    fix16_t scores[scoresLength]; // this could be stored as int16, since the values should be <1
    int s = 0;  // index to scores
    for(int mapNum=0; mapNum<3; mapNum++){
        int8_t* confMap = confMaps[mapNum];
        int pixels = mapPixels[mapNum];
        for(int n=0; n<pixels*2; n++){
            scores[s++] = int8_to_fix16_single((int8_t)confMap[n],scale_outs[mapNum],zero_points[mapNum]); //standardize scores
        }
    }

    // add scores above threshold to a sorted list of indices (indices of highest scores first)
    int order[maxPreDetects];
    int orderLength = 0;
    for(int n=0; n<scoresLength; n++){
        if(scores[n] > confidence_threshold){
            int i=0;
            while(i<orderLength){ // find the insertion index
                if(scores[n] > scores[order[i]]){
		            int i_start = orderLength < maxPreDetects-1 ? orderLength : maxPreDetects-1;
                    for(int i2=i_start; i2>i; i2--) // move down all lower elements
                        order[i2] = order[i2-1];
                    order[i] = n;
		            if (orderLength < maxPreDetects) orderLength++;
                        break;
                }
                i++;
            }
            if(i==orderLength && orderLength<maxPreDetects){   // if not inserted and there's room, then insert at the end
                order[i] = n;
                orderLength++;
            }
        }
    }
    int facesLength = 0;
    for(int n=0; n<orderLength; n++){
        int ind = order[n];
        faces[facesLength].detect_score = scores[ind];

        // get map number from index
        int mapNum = 0;
        if(ind >= 2*mapPixels[0]){
            ind -= 2*mapPixels[0];
            mapNum++;
            if(ind >= 2*mapPixels[1]){
                ind -= 2*mapPixels[1];
                mapNum++;
            }
        }

        // get anchor from index
        int pixels = mapPixels[mapNum];
        int anchNum = 0;
        if (ind >= pixels){
            anchNum = 1;
            ind -= pixels;
        }

        // get pixel indices
        int y = ind / mapSizes[mapNum][1];
        int x = ind - y*mapSizes[mapNum][1];
        // get prior
        int stride = mapStrides[mapNum];
        fix16_t anchX = fix16_from_int(x*stride);
        fix16_t anchY = fix16_from_int(y*stride);
        // python anchor is equal to [anchX, anchY]
        
        // get box location data
        fix16_t location[4];  //int8
        int8_t* locPtr = &locMaps[mapNum][anchNum*4*pixels+ind];
        for(int nLoc=0; nLoc<4; nLoc++){
            location[nLoc] = int8_to_fix16_single(*locPtr,scale_outs[3+mapNum],zero_points[3+mapNum]) * stride;
            locPtr += pixels;
        }
        // [left, top, right, bottom]
        faces[facesLength].box[0] = anchX - location[0];
        faces[facesLength].box[1] = anchY - location[1];
        faces[facesLength].box[2] = anchX + location[2];
        faces[facesLength].box[3] = anchY + location[3];
        
        // NMS
        int passNms = 1;
        for(int f=0; f<facesLength; f++){
            fix16_t iou = calcIou_LTRB(faces[f].box, faces[facesLength].box);
            if(iou > nms_threshold){
                passNms = 0;
                break;
            }
        }
        if(!passNms)
            continue;
        // landmarks
        int8_t* landPtr = &landMaps[mapNum][anchNum*10*pixels+ind];    // elements are every "pixels" elements TODO convert to fix16
        fix16_t stride_16 = fix16_from_int(stride);
        for(int p=0; p<5; p++){
            faces[facesLength].points[p][0] = anchX + fix16_mul(int8_to_fix16_single(*landPtr,scale_outs[6+mapNum],zero_points[6+mapNum]), stride_16);
            landPtr += pixels;
            faces[facesLength].points[p][1] = anchY + fix16_mul(int8_to_fix16_single(*landPtr,scale_outs[6+mapNum],zero_points[6+mapNum]), stride_16);
            landPtr += pixels;
        }

        facesLength++;
        if(facesLength>=max_faces)
            break;
    }
    return facesLength;
}


int post_process_scrfd(object_t faces[],int max_faces, fix16_t *network_outputs[9],
                            int image_width, int image_height,
                            fix16_t confidence_threshold, fix16_t nms_threshold){


    const int mapStrides[3] = {8,16,32};
    const int maxPreDetects = 64;
    
    int h32 = image_height/32;  // image height at stride=32
    int w32 = image_width/32;  // image width at stride=32
    int h16 = h32<<1;
    int w16 = w32<<1;
    int h8 = h16<<1;
    int w8 = w16<<1;
    int mapSizes[3][2] = {{h8,w8},{h16,w16},{h32,w32}};
    int mapPixels[3] = {h8*w8, h16*w16, h32*w32};
    // each map shape is [anchor][channel][y][x]
    //   there are 2 anchors per pixel
    //   location map has 4 channels; confidence has 2 channels; landmarks has 10 channels
    //   the range of x and y pixel are given by mapSizes[...]
    fix16_t** confMaps = &network_outputs[0];
    fix16_t** locMaps = &network_outputs[3];
    fix16_t** landMaps = &network_outputs[6];

    int scoresLength = 2*(mapPixels[0]+mapPixels[1]+mapPixels[2]);
    fix16_t scores[scoresLength]; // this could be stored as int16, since the values should be <1
    int s = 0;  // index to scores
    for(int mapNum=0; mapNum<3; mapNum++){
        fix16_t* confMap = confMaps[mapNum];       
        int pixels = mapPixels[mapNum];
        for(int n=0; n<pixels*2; n++){
            scores[s++] = confMap[n];
        }
    }

    // add scores above threshold to a sorted list of indices (indices of highest scores first)
    int order[maxPreDetects];
    int orderLength = 0;
    for(int n=0; n<scoresLength; n++){
        if(scores[n] > confidence_threshold){
            int i=0;
            while(i<orderLength){ // find the insertion index
                if(scores[n] > scores[order[i]]){
		            int i_start = orderLength < maxPreDetects-1 ? orderLength : maxPreDetects-1;
                    for(int i2=i_start; i2>i; i2--) // move down all lower elements
                        order[i2] = order[i2-1];
                    order[i] = n;
		            if (orderLength < maxPreDetects) orderLength++;
                        break;
                }
                i++;
            }
            if(i==orderLength && orderLength<maxPreDetects){   // if not inserted and there's room, then insert at the end
                order[i] = n;
                orderLength++;
            }
        }
    }
    
    int facesLength = 0;
    for(int n=0; n<orderLength; n++){
        int ind = order[n];
        faces[facesLength].detect_score = scores[ind];

        // get map number from index
        int mapNum = 0;
        if(ind >= 2*mapPixels[0]){
            ind -= 2*mapPixels[0];
            mapNum++;
            if(ind >= 2*mapPixels[1]){
                ind -= 2*mapPixels[1];
                mapNum++;
            }
        }

        // get anchor from index
        int pixels = mapPixels[mapNum];
        int anchNum = 0;
        if (ind >= pixels){
            anchNum = 1;
            ind -= pixels;
        }

        // get pixel indices
        int y = ind / mapSizes[mapNum][1];
        int x = ind - y*mapSizes[mapNum][1];

        // get prior
        int stride = mapStrides[mapNum];
        fix16_t anchX = fix16_from_int(x*stride);
        fix16_t anchY = fix16_from_int(y*stride);
        // python anchor is equal to [anchX, anchY]

        // get box location data
        fix16_t location[4];
        fix16_t* locPtr = &locMaps[mapNum][anchNum*4*pixels+ind];
        for(int nLoc=0; nLoc<4; nLoc++){
            location[nLoc] = *locPtr * stride;
            locPtr += pixels;
        }
        // [left, top, right, bottom]
        faces[facesLength].box[0] = anchX - location[0];
        faces[facesLength].box[1] = anchY - location[1];
        faces[facesLength].box[2] = anchX + location[2];
        faces[facesLength].box[3] = anchY + location[3];
        
        // NMS
        int passNms = 1;
        for(int f=0; f<facesLength; f++){
            fix16_t iou = calcIou_LTRB(faces[f].box, faces[facesLength].box);
            if(iou > nms_threshold){
                passNms = 0;
                break;
            }
        }
        if(!passNms)
            continue;
        
        // landmarks
        fix16_t* landPtr = &landMaps[mapNum][anchNum*10*pixels+ind];    // elements are every "pixels" elements
        fix16_t stride_16 = fix16_from_int(stride);
        for(int p=0; p<5; p++){
            faces[facesLength].points[p][0] = anchX + fix16_mul(*landPtr, stride_16);
            landPtr += pixels;
            faces[facesLength].points[p][1] = anchY + fix16_mul(*landPtr, stride_16);
            landPtr += pixels;
        }

        facesLength++;
        if(facesLength>=max_faces)
            break;
    }
    return facesLength;
}
