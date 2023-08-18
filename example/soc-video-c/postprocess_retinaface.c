#include "postprocess.h"
#include <stdio.h>


static inline fix16_t softmaxRetinaface(fix16_t a,fix16_t b){
    if(b<=a)
        return 0;   // score<=0.5, which will be below the threshold, so quit early
    a = fix16_exp(a-b);
    a = fix16_add(a, fix16_one);
    a = fix16_div(fix16_one, a); //TODO: optimize by skipping this and adjusting threshold value instead
    return a;
}



int post_process_retinaface(object_t faces[],int max_faces, fix16_t *network_outputs[9],
                            int image_width, int image_height,
                            fix16_t confidence_threshold, fix16_t nms_threshold){


    const fix16_t variance0 = F16(0.1);
    const fix16_t variance1 = F16(0.2);

    const int mapStrides[3] = {8,16,32};
    const int minSizes[3][2] = {{16,32},{64,128},{256,512}};
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
    fix16_t** locMaps = &network_outputs[0];
    fix16_t** confMaps = &network_outputs[3];
    fix16_t** landMaps = &network_outputs[6];

    int scoresLength = 2*(mapPixels[0]+mapPixels[1]+mapPixels[2]);
    fix16_t scores[scoresLength]; // this could be stored as int16, since the values should be <1
    int s = 0;  // index to scores
    for(int mapNum=0; mapNum<3; mapNum++){
        fix16_t* confMap = confMaps[mapNum];
        int pixels = mapPixels[mapNum];
        for(int n=0; n<pixels; n++){
            // Optimization: first read the face confidence. If the score is less than zero, don't bother reading the non-face confidence.
            //  This helps speed up execution since memory reads are expensive.
            fix16_t faceConf = confMap[n+pixels];
            if(faceConf>0)
                scores[s++] = softmaxRetinaface(confMap[n], faceConf);
            else
                scores[s++] = 0;
            faceConf = confMap[n+3*pixels];
            if(faceConf>0)
                scores[s++] = softmaxRetinaface(confMap[n+2*pixels], faceConf);
            else
                scores[s++] = 0;
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
        int anchNum = ind%2;
        ind = ind>>1;

        // get pixel indices
        int y = ind / mapSizes[mapNum][1];
        int x = ind - y*mapSizes[mapNum][1];

        // get prior
        fix16_t anchS = fix16_from_int(minSizes[mapNum][anchNum]);  // could replace this by log2 value and use bitshift instead of multiply
        fix16_t anchX = fix16_from_int(x*mapStrides[mapNum] + (mapStrides[mapNum]>>1));
        fix16_t anchY = fix16_from_int(y*mapStrides[mapNum] + (mapStrides[mapNum]>>1));
        // python anchor is equal to [anchX, anchY, anchS, anchS]

        // get box location data
        int pixels = mapPixels[mapNum];
        fix16_t location[4];
        fix16_t* locPtr = &locMaps[mapNum][anchNum*4*pixels+ind];
        for(int nLoc=0; nLoc<4; nLoc++){
            location[nLoc] = *locPtr;
            locPtr += pixels;
        }
        fix16_t boxX = anchX + fix16_mul(location[0], fix16_mul(variance0, anchS));
        fix16_t boxY = anchY + fix16_mul(location[1], fix16_mul(variance0, anchS));
        fix16_t boxW = fix16_mul(anchS, fix16_exp(fix16_mul(location[2], variance1)));
        fix16_t boxH = fix16_mul(anchS, fix16_exp(fix16_mul(location[3], variance1)));
        // convert from [center x, center y, width, height] to [left, top, right, bottom]
        faces[facesLength].box[0] = boxX-(boxW>>1);
        faces[facesLength].box[1] = boxY-(boxH>>1);
        faces[facesLength].box[2] = boxX+(boxW>>1);
        faces[facesLength].box[3] = boxY+(boxH>>1);
        
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
        
        fix16_t* landPtr = &landMaps[mapNum][anchNum*10*pixels+ind];    // elements are every "pixels" elements
        fix16_t var0MulAnchS = fix16_mul(variance0, anchS);
        for(int p=0; p<5; p++){
            faces[facesLength].points[p][0] = anchX + fix16_mul(*landPtr, var0MulAnchS);
            landPtr += pixels;
            faces[facesLength].points[p][1] = anchY + fix16_mul(*landPtr, var0MulAnchS);
            landPtr += pixels;
        }

        facesLength++;
        if(facesLength>=max_faces)
            break;
    }
    return facesLength;
}
