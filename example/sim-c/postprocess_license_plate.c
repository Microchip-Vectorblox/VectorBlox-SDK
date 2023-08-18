#include "postprocess.h"
#include <stdio.h>


const float lpr_anchors[3][2] = {{180.0,52.0}, {60.0,18.0}, {20.0,7.0}};
static fix16_t fix16_half = F16(.5);
static fix16_t fix16_nthous = F16(-1000);

#define maxPreDetects 64

#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

void xywhssToKps(fix16_t* xywhss, fix16_t* kps) {
    fix16_t x = xywhss[0];
    fix16_t y = xywhss[1];
    fix16_t w = xywhss[2];
    fix16_t h = xywhss[3];
    fix16_t sx = xywhss[4];
    fix16_t sy = xywhss[5];

    fix16_t k[8] = {x, y, x, y, x, y, x, y};
    fix16_t a[8] = {-w, -h,  w, -h, -w,  h,  w,  h};
    fix16_t s[8] = {sx, sy, sx,-sy,-sx, sy,-sx,-sy};
    for(int i = 0; i < 8; i++) {
        a[i] = fix16_mul(a[i],fix16_half);
        s[i] = fix16_mul(s[i],fix16_half);
        k[i] = k[i] + a[i] +s[i];
        kps[i] = k[i];
    }
}

void kpsToXywh(fix16_t* kps, fix16_t* location) {
    fix16_t left = MIN(MIN(kps[0], kps[2]), MIN(kps[2], kps[4]));
    fix16_t right = MAX(MAX(kps[0], kps[2]), MAX(kps[2], kps[4]));
    fix16_t top = MIN(MIN(kps[1], kps[3]), MIN(kps[5], kps[7]));
    fix16_t bottom = MAX(MAX(kps[1], kps[3]), MAX(kps[5], kps[7]));
    fix16_t w = right - left;
    fix16_t h = bottom - top;
    fix16_t x = fix16_mul((left+right),fix16_half);
    fix16_t y = fix16_mul((top+bottom),fix16_half);
    location[0] = x;
    location[1] = y;
    location[2] = w;
    location[3] = h;
}


void kpsToLTRB(fix16_t* kps, fix16_t* location) {
    fix16_t left = MIN(MIN(kps[0], kps[2]), MIN(kps[2], kps[4]));
    fix16_t right = MAX(MAX(kps[0], kps[2]), MAX(kps[2], kps[4]));
    fix16_t top = MIN(MIN(kps[1], kps[3]), MIN(kps[5], kps[7]));
    fix16_t bottom = MAX(MAX(kps[1], kps[3]), MAX(kps[5], kps[7]));
    location[0] = left;
    location[1] = top;
    location[2] = right;
    location[3] = bottom;
}

int post_process_lpd(object_t objects[],int max_objects, fix16_t *detectOutputs[9], int image_width, int image_height,
					fix16_t confidence_threshold, fix16_t nms_threshold,int detectNumOutputs) {

    const int mapStrides[3] = {32,16,8};
    int h32 = image_height/32;  // image height at stride=32
    int w32 = image_width/32;  // image width at stride=32
    int h16 = h32<<1;
    int w16 = w32<<1;
    int h8 = h16<<1;
    int w8 = w16<<1;
    int mapSizes[3][2] = {{h32,w32},{h16,w16},{h8,w8}};
    int mapPixels[3] = {h32*w32, h16*w16, h8*w8};
    fix16_t *objMaps[3], *keyMaps[3], *boxMaps[3], *shapeOutput[3];

    // 3 maps for object, keypoints, and boxes
    if(detectNumOutputs == 6) {
    	objMaps[0] = detectOutputs[0];
    	objMaps[1] = detectOutputs[2];
    	objMaps[2] = detectOutputs[4];

    	shapeOutput[0] = detectOutputs[1];
    	shapeOutput[1] = detectOutputs[3];
    	shapeOutput[2] = detectOutputs[5];
    } else {
    	objMaps[0] = detectOutputs[0];
    	objMaps[1] = detectOutputs[3];
    	objMaps[2] = detectOutputs[6];

    	keyMaps[0] = detectOutputs[1];
    	keyMaps[1] = detectOutputs[4];
    	keyMaps[2] = detectOutputs[7];

    	boxMaps[0] = detectOutputs[2];
    	boxMaps[1] = detectOutputs[5];
    	boxMaps[2] = detectOutputs[8];
    }

    int scoresLength = (mapPixels[0] + mapPixels[1] + mapPixels[2]);
    fix16_t scores[scoresLength]; // this could be stored as int16, since the values should be <1
    int s = 0;  // index to scores
    for(int mapNum = 0; mapNum < 3; mapNum++) {
    	fix16_t* objMap = objMaps[mapNum];
        int pixels = mapPixels[mapNum];
        for(int n = 0; n < pixels; n++) {
            scores[s] = objMap[n];
            s++;
        }
    }

    // add scores above threshold to a sorted list of indices (indices of highest scores first)
    int order[maxPreDetects] = {0};
    int orderLength = 0;
    for(int n = 0; n < scoresLength; n++) {
    	if(scores[n] > confidence_threshold) {
    		int i = 0;
            while(i < orderLength) { // find the insertion index
            	if(scores[n] > scores[order[i]]) {
            		int i_start = orderLength < maxPreDetects-1 ? orderLength : maxPreDetects-1;
                    for(int i2 = i_start; i2 > i; i2--) // move down all lower elements
                    	order[i2] = order[i2-1];

                    order[i] = n;
                    if (orderLength < maxPreDetects) orderLength++;
                    break;
                }
                i++;
            }
            if((i == orderLength) && (orderLength < maxPreDetects)) {   // if not inserted and there's room, then insert at the end
            	order[i] = n;
                orderLength++;
            }
        }
    }

    int length = 0;
    for(int n = 0; n < orderLength; n++) {
    	int ind = order[n];
        objects[length].detect_score = scores[ind];

        // get map number from index
        int mapNum = 0;
        if(ind >= mapPixels[0]) {
        	ind -= mapPixels[0];
            mapNum++;
            if(ind >= mapPixels[1]) {
            	ind -= mapPixels[1];
                mapNum++;
            }
        }

        // get anchor from index
        int pixels = mapPixels[mapNum];
        int anchNum = 0;
        if (ind >= pixels) {
        	anchNum = 1;
            ind -= pixels;
        }

        // get pixel indices
        int y = ind / mapSizes[mapNum][1];
        int x = ind - y*mapSizes[mapNum][1];

        // get prior
        //objects[length].stride = mapStrides[mapNum];

        fix16_t raw[6];
        fix16_t* current_output_layer = shapeOutput[mapNum];
        for(int i = 0; i < 6; i++) {
        	raw[i] = current_output_layer[i*mapPixels[mapNum]+ y*mapSizes[mapNum][1]+x];
        }

        fix16_t xywhss[6];
        fix16_t kps[8];
        // get box location data
        fix16_t location[4];
        if (detectNumOutputs == 6) {  //6 output layers
        	for(int i =0; i<4; i++) {
        		location[i] = raw[i];
                location[i] = fix16_logistic_activate(raw[i]);
                location[i] = location[i]<<1;
            }
        }
        else{   //9 output layers
            fix16_t* locPtr = &boxMaps[mapNum][anchNum*4*pixels+ind];
            for(int nLoc = 0; nLoc < 4; nLoc++) {
            	location[nLoc] = *locPtr;// * stride;
                locPtr += pixels;
                location[nLoc] = fix16_logistic_activate(location[nLoc]);
                location[nLoc] = location[nLoc]<<1;
            }
        }

        // code below comes from Python for getting box[center x, center y, width, height]
        fix16_t box_x = location[0] + fix16_from_float(x - 0.5);
        fix16_t box_y = location[1] + fix16_from_float(y - 0.5);
        fix16_t box_w = fix16_mul(location[2], location[2]);  //calculations for box width
        fix16_t box_h = fix16_mul(location[3], location[3]);  //calculations for box height

        location[0] = (box_x)<<(5-mapNum); //fix16_mul(box_x, fix16_from_int(mapStrides[mapNum]));
        location[1] = (box_y)<<(5-mapNum); //fix16_mul(box_y, fix16_from_int(mapStrides[mapNum]));
        location[2] = fix16_mul(box_w, fix16_from_float(lpr_anchors[mapNum][anchNum*2]));
        location[3] = fix16_mul(box_h, fix16_from_float(lpr_anchors[mapNum][anchNum*2+1]));

        //set xywhss
        xywhss[0] = location[0];  //center x
        xywhss[1] = location[1];  //center y
        xywhss[2] = location[2];  //w
        xywhss[3] = location[3];  //h
        xywhss[4] = fix16_mul(raw[4], fix16_from_int(.25*mapStrides[mapNum])); //sx
        xywhss[5] = fix16_mul(raw[5], fix16_from_int(mapStrides[mapNum]));  //sy

        if(detectNumOutputs == 6){
        	xywhssToKps(xywhss, kps); //pass both the xywhss array and the keypoints per specified plate
            //kpsToXywh(kps, location); //update from keypoints in location
            kpsToLTRB(kps, location); //update from keypoints in location
        }
        objects[length].box[0] = location[0];
        objects[length].box[1] = location[1];
        objects[length].box[2] = location[2];
        objects[length].box[3] = location[3];

        // NMS
        int passNms = 1;
        for(int f=0; f<length; f++){
        	//fix16_t iou = calcIou_XYWH(objects[f].box, objects[length].box);
        	fix16_t iou = calcIou_LTRB(objects[f].box, objects[length].box);
            if(iou > nms_threshold){
            	passNms = 0;
                break;
            }
        }

        if(!passNms)
        	continue;

        if(detectNumOutputs == 9) {
        	// keypoints
        	fix16_t stride_val = mapStrides[mapNum];
        	int stride_y = 9;  //default max y for stride 32
            int stride_x = 32; //default max x for stride 32
            if (stride_val == 16){
                stride_y = 18;
                stride_x = 64;
            }
            if (stride_val == 8){
                stride_y = 36;
                stride_x = 128;
            }

            for(int p=0; p<4; p++){
            	// assign keypoints based on stride x and y values per each stride, indexing is done based on stride_y and stride_x
                // y and x values are the actual location in which the keypoint is found
                objects[length].points[p][0] = fix16_mul(keyMaps[mapNum][(2*p+0)*stride_y*stride_x + y*stride_x + x] + fix16_from_int(x) + fix16_from_float(0.5), stride_val);
                objects[length].points[p][1] = fix16_mul(keyMaps[mapNum][(2*p+1)*stride_y*stride_x + y*stride_x + x] + fix16_from_int(y) + fix16_from_float(0.5), stride_val);
            }
        } else {
        	for(int p = 0; p < 4; p++) {
        		objects[length].points[p][0]= kps[2*p+0];
        		objects[length].points[p][1] = kps[2*p+1];
            }
        }

        length++;
        if(length >= max_objects)
        	break;
    }

    return length;
}

static char CHARS[] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'};
#define CHAR_LENGTH 37
//Global specifications
#define PLATE_LENGTH 10
#define RECOGNIZER_COLS 18
#define MAX_PLATE_LETTERS 9
#define MIN_PLATE_LETTERS 4


void PlateDecodeIndicies(int* maxInd, char* label) {
    char test[PLATE_LENGTH] = "";
    int prev = -1;
    for(int n = 0; n < RECOGNIZER_COLS; n++){
        if ((maxInd[n]!= prev) && (maxInd[n] != (CHAR_LENGTH-1))){
            strncat(test,&CHARS[maxInd[n]],1);
        }
        prev = maxInd[n];
    }

    strcpy(label, test);
}

fix16_t PlateDecodeCStyle(fix16_t* raw,  int output_length, char* label) {
    fix16_t temp_outputs[output_length];
    for (int i = 0; i < output_length; i++) {
        temp_outputs[i] = raw[i];
    }

    fix16_t maxVal[RECOGNIZER_COLS];
	int maxInd[RECOGNIZER_COLS] = {0};                 //index of max value for each col
    fix16_t secVal[RECOGNIZER_COLS];
    int secInd[RECOGNIZER_COLS] = {0};                 //index of second highest val for each col
    int testInd[RECOGNIZER_COLS] = {0};
    fix16_t minDiff = 0;
    int minCol;

    //set highest value and second highest val for each col to -1000
    for (int c = 0; c < RECOGNIZER_COLS; c++) {
    	maxVal[c] = fix16_nthous;
    	secVal[c] = fix16_nthous;
    }
    for(int c = 0; c< RECOGNIZER_COLS; c++){           // stores the values and indexes of the highest and 2nd highest values
        for(int r = 0; r < CHAR_LENGTH; r++){
            if(temp_outputs[r*RECOGNIZER_COLS + c] > maxVal[c]){
                secVal[c] = maxVal[c];
                secInd[c] = maxInd[c];
                maxVal[c] = temp_outputs[r*RECOGNIZER_COLS + c];
                maxInd[c] = r;
            }
            else if(temp_outputs[r*RECOGNIZER_COLS + c] > secVal[c]){
                secVal[c] = temp_outputs[r*RECOGNIZER_COLS + c];
                secInd[c] = r;
            }
        }
    }

    PlateDecodeIndicies(maxInd, label);
    if (strlen(label) > MAX_PLATE_LETTERS  || strlen(label) < MIN_PLATE_LETTERS) {
    	return 0;
    }

    fix16_t diffVal[RECOGNIZER_COLS] = {0};
    for(int c = 0; c < RECOGNIZER_COLS; c++){
        diffVal[c] = maxVal[c] - secVal[c];
    }

    while(1) {
    	char testLabel[PLATE_LENGTH] = "";

        for(int i = 0; i < RECOGNIZER_COLS; i++)
            testInd[i] = maxInd[i];

        minDiff = diffVal[0];
        minCol = 0;
        for(int c = 1; c < RECOGNIZER_COLS; c++){
            if (diffVal[c] < minDiff){
                minDiff=diffVal[c];
                minCol = c;
            }
        }
        testInd[minCol] = secInd[minCol];
        PlateDecodeIndicies(testInd, testLabel);
        if (strcmp(testLabel, label)) {
            return minDiff;
        } else{
            temp_outputs[secInd[minCol] * RECOGNIZER_COLS + minCol] = fix16_nthous;  //set to -1000
            secVal[minCol] = fix16_nthous;
            for (int r = 0; r < CHAR_LENGTH; r++){
                if(r != maxInd[minCol]  && (temp_outputs[r*RECOGNIZER_COLS + minCol] > secVal[minCol])){
                    secVal[minCol] = temp_outputs[r*RECOGNIZER_COLS + minCol];
                    secInd[minCol] = r;
                }
            }
            diffVal[minCol] = maxVal[minCol] - secVal[minCol];
        }
    }
}

fix16_t post_process_lpr(fix16_t *output, int output_length, char *label) {
	fix16_t conf =  PlateDecodeCStyle(output, output_length, label);
	return conf;
}
