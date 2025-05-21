#include "postprocess.h"
#include <stdio.h>
#include <stdbool.h>

#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

#ifdef __cplusplus
extern "C" {
#endif

//global params

const int NUM_KEYPOINTS = 17;
const int LOCAL_MAXIMUM_RADIUS = 1;
const int NUM_EDGES = 16;

const int PARENT_CHILD_TUPLES[16][2] = {{0, 1}, {1, 3}, {0, 2}, {2, 4}, {0, 5}, {5, 7}, {7, 9}, {5, 11}, {11, 13}, {13, 15}, {0, 6}, {6, 8}, {8, 10}, {6, 12}, {12, 14}, {14, 16}};


int buildPartWithScoreQueue_int8(int8_t *scores,fix16_t scoreThreshold,queue_element_t *parts, int height, int width,int zero_point,fix16_t scale){
    fix16_t score = 0;
    int queue_count = 0;
    for(int keypoint_id = 0; keypoint_id<NUM_KEYPOINTS; keypoint_id++){
        for(int hmy = 0; hmy<height; hmy++){
            for(int hmx = 0; hmx<width; hmx++){
                score = int8_to_fix16_single(scores[keypoint_id*height*width + hmy*width + hmx],scale,zero_point); 
                if (score >= scoreThreshold){
                    fix16_t yStart = MAX(hmy - LOCAL_MAXIMUM_RADIUS, 0);
                    fix16_t yEnd = MIN(hmy + LOCAL_MAXIMUM_RADIUS + 1, height);
                    fix16_t xStart = MAX(hmx - LOCAL_MAXIMUM_RADIUS, 0);
                    fix16_t xEnd = MIN(hmx + LOCAL_MAXIMUM_RADIUS + 1, width);
                    bool localmax = true;
                    for(int yCurrent = yStart; yCurrent < yEnd; ++yCurrent){
                        for(int xCurrent = xStart; xCurrent < xEnd; ++xCurrent){
                            if( score < int8_to_fix16_single(scores[keypoint_id*height*width + yCurrent*width + xCurrent],scale,zero_point)){
                                localmax = false; 
                            }
                        }
                    }
                    if (localmax){ //insert in order into queue
                        int pos =0;
                        int m;
                        for (int i = 0; i < queue_count; i++){ //find insertion position
                            if (score > parts[i].scores){
                                pos = i;
                                break;
                            }
                            if (score < parts[queue_count-1].scores){ 
                                pos = queue_count;
                                break;
                            }
                        }
                        if (pos != queue_count){ //insert into position
                            m = queue_count - pos + 1 ;
                            for (int i = 0; i <= m; i++){
                                parts[queue_count - i + 2].scores = parts[queue_count - i + 1].scores;
                                parts[queue_count - i + 2].id = parts[queue_count - i + 1].id;
                                parts[queue_count - i + 2].points[0] = parts[queue_count - i + 1].points[0];
                                parts[queue_count - i + 2].points[1] = parts[queue_count - i + 1].points[1];
                            }
                        }

                        //append to end
                        parts[pos].scores = score;
                        parts[pos].id = keypoint_id;
                        parts[pos].points[0]=hmy;
                        parts[pos].points[1]=hmx;

                        queue_count++;           
                    }
                }
            }
        }
    }
    return queue_count;
}

int buildPartWithScoreQueue(fix16_t *scores,fix16_t scoreThreshold,queue_element_t *parts, int height, int width){
    fix16_t score = 0;
    int queue_count = 0;
    for(int keypoint_id = 0; keypoint_id<NUM_KEYPOINTS; keypoint_id++){
        for(int hmy = 0; hmy<height; hmy++){
            for(int hmx = 0; hmx<width; hmx++){
                score = scores[keypoint_id*height*width + hmy*width + hmx]; 
                if (score >= scoreThreshold){
                    fix16_t yStart = MAX(hmy - LOCAL_MAXIMUM_RADIUS, 0);
                    fix16_t yEnd = MIN(hmy + LOCAL_MAXIMUM_RADIUS + 1, height);
                    fix16_t xStart = MAX(hmx - LOCAL_MAXIMUM_RADIUS, 0);
                    fix16_t xEnd = MIN(hmx + LOCAL_MAXIMUM_RADIUS + 1, width);
                    bool localmax = true;
                    for(int yCurrent = yStart; yCurrent < yEnd; ++yCurrent){
                        for(int xCurrent = xStart; xCurrent < xEnd; ++xCurrent){
                            if( score < scores[keypoint_id*height*width + yCurrent*width + xCurrent]){
                                localmax = false; 
                            }
                        }
                    }
                    if (localmax){ //insert in order into queue
                        int pos =0;
                        int m;
                        for (int i = 0; i < queue_count; i++){ //find insertion position
                            if (score > parts[i].scores){
                                pos = i;
                                break;
                            }
                            if (score < parts[queue_count-1].scores){ 
                                pos = queue_count;
                                break;
                            }
                        }
                        if (pos != queue_count){ //insert into position
                            m = queue_count - pos + 1 ;
                            for (int i = 0; i <= m; i++){
                                parts[queue_count - i + 2].scores = parts[queue_count - i + 1].scores;
                                parts[queue_count - i + 2].id = parts[queue_count - i + 1].id;
                                parts[queue_count - i + 2].points[0] = parts[queue_count - i + 1].points[0];
                                parts[queue_count - i + 2].points[1] = parts[queue_count - i + 1].points[1];
                            }
                        }

                        //append to end
                        parts[pos].scores = score;
                        parts[pos].id = keypoint_id;
                        parts[pos].points[0]=hmy;
                        parts[pos].points[1]=hmx;

                        queue_count++;           
                    }
                }
            }
        }
    }
    return queue_count;
}


bool withinNmsRadiusOfCorrespondingPoint(poses_t *poses,int squaredNmsRadius,fix16_t *rootImageCoords,int rootId,int poseCount){
    for (int i =0; i<poseCount; i++){
        if(fix16_sq((poses[i].keypoints[rootId][0])-rootImageCoords[0]) + fix16_sq((poses[i].keypoints[rootId][1])-rootImageCoords[1]) <= fix16_from_int(squaredNmsRadius)){
            return true;
        }
    }                     
    return false;
}





fix16_t traverseToTargetKeypoint(int edgeId, fix16_t source_y, fix16_t source_x,int targetKeypointId,fix16_t *scores,fix16_t *offsets,int outputStride, fix16_t *displacements,fix16_t *imageCoord,int height, int width){


    fix16_t score = 0;
    int sourceKeypointIndices[2];
    sourceKeypointIndices[0] = MIN(MAX(fix16_to_int(fix16_mul(source_y,fix16_from_float(1.0/outputStride))),0),height-1);
    sourceKeypointIndices[1] = MIN(MAX(fix16_to_int(fix16_mul(source_x,fix16_from_float(1.0/outputStride))),0),width-1);

    fix16_t displacedPoint[2];
    displacedPoint[0] = source_y + displacements[edgeId*height*width + sourceKeypointIndices[0]*width + sourceKeypointIndices[1]];
    displacedPoint[1] = source_x + displacements[(edgeId+NUM_EDGES)*height*width + sourceKeypointIndices[0]*width + sourceKeypointIndices[1]];

    int displacedPointIndices[2];
    displacedPointIndices[0] = MIN(MAX(fix16_to_int(fix16_mul(displacedPoint[0],fix16_from_float(1.0/outputStride))),0),height-1);
    displacedPointIndices[1] = MIN(MAX(fix16_to_int(fix16_mul(displacedPoint[1],fix16_from_float(1.0/outputStride))),0),width-1);
    score = scores[targetKeypointId*height*width + displacedPointIndices[0]*width + displacedPointIndices[1]];
    
    imageCoord[0] = fix16_from_int(displacedPointIndices[0]*outputStride) + offsets[targetKeypointId*height*width + displacedPointIndices[0]*width + displacedPointIndices[1]];
    imageCoord[1] = fix16_from_int(displacedPointIndices[1]*outputStride) + offsets[(targetKeypointId+NUM_KEYPOINTS)*height*width + displacedPointIndices[0]*width + displacedPointIndices[1]];
    
    return score; 
}

fix16_t getInstanceScore(poses_t *poses,int squaredNmsRadius,fix16_t *keypointScores,fix16_t keypointCoords[][2], int kpScoreslength ,int poseCount){
    fix16_t notOverlappedKeypointScores  = 0;
    fix16_t rootCoord[2];
    for(int i = 0; i < kpScoreslength; i++){
        rootCoord[0] = keypointCoords[i][0];
        rootCoord[1] = keypointCoords[i][1];
        if(!withinNmsRadiusOfCorrespondingPoint(poses, squaredNmsRadius, rootCoord, i, poseCount)){
            notOverlappedKeypointScores += keypointScores[i];
        }
    }
    
    return fix16_mul(notOverlappedKeypointScores,fix16_from_float(1.0/kpScoreslength));
}


void decodePose(fix16_t rootScore, int rootId, fix16_t *rootImageCoord, fix16_t *scores, fix16_t *offsets, int outputStride,fix16_t *displacementsFwd, fix16_t *displacementsBwd,poses_t *pose,int height, int width){


    fix16_t keypointScores[17] = {0};
    fix16_t keypointCoords[17][2] = {0};

    keypointScores[rootId] = rootScore;
    keypointCoords[rootId][0] = rootImageCoord[0];
    keypointCoords[rootId][1] = rootImageCoord[1];
    fix16_t imageCoord[2];
    int targetKeypointId,sourceKeypointId;  
    for(int bwd = NUM_EDGES-1; bwd>=0; bwd--){
        targetKeypointId = PARENT_CHILD_TUPLES[bwd][0];
        sourceKeypointId = PARENT_CHILD_TUPLES[bwd][1];
        if (keypointScores[sourceKeypointId] > 0.0 && keypointScores[targetKeypointId] == 0.0){
            keypointScores[targetKeypointId] = traverseToTargetKeypoint(bwd, keypointCoords[sourceKeypointId][0],keypointCoords[sourceKeypointId][1], targetKeypointId, scores, offsets, outputStride, displacementsBwd, imageCoord, height, width);
            keypointCoords[targetKeypointId][0] = imageCoord[0];
            keypointCoords[targetKeypointId][1] = imageCoord[1];
        }
    }
    for(int fwd = 0; fwd < NUM_EDGES; fwd++){
        sourceKeypointId = PARENT_CHILD_TUPLES[fwd][0];
        targetKeypointId = PARENT_CHILD_TUPLES[fwd][1];
        if (keypointScores[sourceKeypointId] > 0.0 && keypointScores[targetKeypointId] == 0.0){
            keypointScores[targetKeypointId] = traverseToTargetKeypoint(fwd, keypointCoords[sourceKeypointId][0],keypointCoords[sourceKeypointId][1], targetKeypointId, scores, offsets, outputStride, displacementsFwd, imageCoord, height, width);
            keypointCoords[targetKeypointId][0] = imageCoord[0];
            keypointCoords[targetKeypointId][1] = imageCoord[1];
        }
    }

    for(int i =0; i<NUM_KEYPOINTS; i++){
        pose->keypoints[i][0]=keypointCoords[i][0];
        pose->keypoints[i][1]=keypointCoords[i][1];
        pose->scores[i] = keypointScores[i];

    }
}

int decodeMultiplePoses(poses_t poses[],fix16_t *scores, fix16_t *offsets,fix16_t *displacementsFwd, fix16_t *displacementsBwd, int outputStride,int maxPoseDetections, fix16_t scoreThreshold,int nmsRadius, fix16_t minPoseScore,int height, int width){

    int sqnmsRadius = nmsRadius * nmsRadius;
    int queue_count = 0;
    int pose_count = 0;
    fix16_t poseScore;
    poses_t temp_pose;
    queue_element_t queue[200];
    //initialize values
    for (int i = 0;i < 200;i++)
        queue[i].scores = 0;
    fix16_t rootScore = 0;
    int rootId = 0;
    int rootCoord[2] = {0};
    fix16_t rootImageCoord[2] = {0};
    queue_count = buildPartWithScoreQueue(scores, scoreThreshold, queue, height, width);
    for (int i =0; i < queue_count; i++){
        rootScore = queue[i].scores;
        rootId = queue[i].id;
        rootCoord[0] = queue[i].points[0];
        rootCoord[1] = queue[i].points[1];


        rootImageCoord[0] =  (fix16_from_int(rootCoord[0]*outputStride)) + offsets[rootId * height*width + rootCoord[0]*width + rootCoord[1]];
        rootImageCoord[1] =  (fix16_from_int(rootCoord[1]*outputStride)) + offsets[(rootId+NUM_KEYPOINTS) * height*width + rootCoord[0]*width + rootCoord[1]];

        if (withinNmsRadiusOfCorrespondingPoint(poses, sqnmsRadius, rootImageCoord, queue[i].id, pose_count)){
            continue;
        }
        
        decodePose(rootScore,rootId,rootImageCoord,scores, offsets,outputStride, displacementsFwd, displacementsBwd,&temp_pose, height, width); 

        poseScore = getInstanceScore(poses, sqnmsRadius, temp_pose.scores, temp_pose.keypoints, NUM_KEYPOINTS ,pose_count);
 
        if (poseScore>=minPoseScore){
            poses[pose_count].poseScore = poseScore;
            for(int i = 0; i< NUM_KEYPOINTS; i++ ){
                poses[pose_count].scores[i] = temp_pose.scores[i];
                poses[pose_count].keypoints[i][0] = temp_pose.keypoints[i][0];
                poses[pose_count].keypoints[i][1] = temp_pose.keypoints[i][1];               
            }
            pose_count++;
        }
        if(pose_count>=maxPoseDetections){
            break;
        }

    }
    return pose_count;
}

fix16_t traverseToTargetKeypoint_int8(int edgeId, fix16_t source_y, fix16_t source_x,int targetKeypointId,int8_t *scores,int8_t *offsets,int outputStride, int8_t *displacements,fix16_t *imageCoord,int height, int width, int zero_point[], fix16_t scale[],int disp){


    fix16_t score = 0;
    int sourceKeypointIndices[2];
    sourceKeypointIndices[0] = MIN(MAX(fix16_to_int(fix16_mul(source_y,fix16_from_float(1.0/outputStride))),0),height-1);
    sourceKeypointIndices[1] = MIN(MAX(fix16_to_int(fix16_mul(source_x,fix16_from_float(1.0/outputStride))),0),width-1);

    fix16_t displacedPoint[2];
    displacedPoint[0] = source_y + int8_to_fix16_single(displacements[edgeId*height*width + sourceKeypointIndices[0]*width + sourceKeypointIndices[1]],scale[disp],zero_point[disp]);
    displacedPoint[1] = source_x + int8_to_fix16_single(displacements[(edgeId+NUM_EDGES)*height*width + sourceKeypointIndices[0]*width + sourceKeypointIndices[1]],scale[disp],zero_point[disp]);

    int displacedPointIndices[2];
    displacedPointIndices[0] = MIN(MAX(fix16_to_int(fix16_mul(displacedPoint[0],fix16_from_float(1.0/outputStride))),0),height-1);
    displacedPointIndices[1] = MIN(MAX(fix16_to_int(fix16_mul(displacedPoint[1],fix16_from_float(1.0/outputStride))),0),width-1);
    score = int8_to_fix16_single(scores[targetKeypointId*height*width + displacedPointIndices[0]*width + displacedPointIndices[1]],scale[1],zero_point[1]);
    
    imageCoord[0] = fix16_from_int(displacedPointIndices[0]*outputStride) + int8_to_fix16_single(offsets[targetKeypointId*height*width + displacedPointIndices[0]*width + displacedPointIndices[1]],scale[0],zero_point[0]);
    imageCoord[1] = fix16_from_int(displacedPointIndices[1]*outputStride) + int8_to_fix16_single(offsets[(targetKeypointId+NUM_KEYPOINTS)*height*width + displacedPointIndices[0]*width + displacedPointIndices[1]],scale[0],zero_point[0]);
    
    return score; 
}

void decodePose_int8(fix16_t rootScore, int rootId, fix16_t *rootImageCoord, int8_t *scores, int8_t *offsets, int outputStride,int8_t *displacementsFwd, int8_t *displacementsBwd,poses_t *pose,int height, int width, int zero_points[], fix16_t scale_outs[]){


    fix16_t keypointScores[17] = {0};
    fix16_t keypointCoords[17][2] = {0};

    keypointScores[rootId] = rootScore;
    keypointCoords[rootId][0] = rootImageCoord[0];
    keypointCoords[rootId][1] = rootImageCoord[1];
    fix16_t imageCoord[2];
    int disp;
    int targetKeypointId,sourceKeypointId;  
    for(int bwd = NUM_EDGES-1; bwd>=0; bwd--){
        targetKeypointId = PARENT_CHILD_TUPLES[bwd][0];
        sourceKeypointId = PARENT_CHILD_TUPLES[bwd][1];
        if (keypointScores[sourceKeypointId] > 0.0 && keypointScores[targetKeypointId] == 0.0){
            disp = 3;
            keypointScores[targetKeypointId] = traverseToTargetKeypoint_int8(bwd, keypointCoords[sourceKeypointId][0],keypointCoords[sourceKeypointId][1], targetKeypointId, scores, offsets, outputStride, displacementsBwd, imageCoord, height, width, zero_points, scale_outs, disp);
            keypointCoords[targetKeypointId][0] = imageCoord[0];
            keypointCoords[targetKeypointId][1] = imageCoord[1];
        }
    }
    for(int fwd = 0; fwd < NUM_EDGES; fwd++){
        sourceKeypointId = PARENT_CHILD_TUPLES[fwd][0];
        targetKeypointId = PARENT_CHILD_TUPLES[fwd][1];
        if (keypointScores[sourceKeypointId] > 0.0 && keypointScores[targetKeypointId] == 0.0){
            disp = 2;
            keypointScores[targetKeypointId] = traverseToTargetKeypoint_int8(fwd, keypointCoords[sourceKeypointId][0],keypointCoords[sourceKeypointId][1], targetKeypointId, scores, offsets, outputStride, displacementsFwd, imageCoord, height, width, zero_points, scale_outs, disp);
            keypointCoords[targetKeypointId][0] = imageCoord[0];
            keypointCoords[targetKeypointId][1] = imageCoord[1];
        }
    }

    for(int i =0; i<NUM_KEYPOINTS; i++){
        pose->keypoints[i][0]=keypointCoords[i][0];
        pose->keypoints[i][1]=keypointCoords[i][1];
        pose->scores[i] = keypointScores[i];

    }
}
int decodeMultiplePoses_int8(poses_t poses[],int8_t *scores, int8_t *offsets,int8_t *displacementsFwd, int8_t *displacementsBwd, int outputStride,int maxPoseDetections, fix16_t scoreThreshold,int nmsRadius, fix16_t minPoseScore,int height, int width, int zero_points[], fix16_t scale_outs[]){

    int sqnmsRadius = nmsRadius * nmsRadius;
    int queue_count = 0;
    int pose_count = 0;
    fix16_t poseScore;
    poses_t temp_pose;
    queue_element_t queue[200];
    //initialize values
    for (int i = 0;i < 200;i++)
        queue[i].scores = 0;
    fix16_t rootScore = 0;
    int rootId = 0;
    int rootCoord[2] = {0};
    fix16_t rootImageCoord[2] = {0};
    queue_count = buildPartWithScoreQueue_int8(scores, scoreThreshold, queue, height, width, zero_points[1], scale_outs[1]); 
    for (int i =0; i < queue_count; i++){
        rootScore = queue[i].scores;
        rootId = queue[i].id;
        rootCoord[0] = queue[i].points[0];
        rootCoord[1] = queue[i].points[1];


        rootImageCoord[0] =  (fix16_from_int(rootCoord[0]*outputStride)) + int8_to_fix16_single(offsets[rootId * height*width + rootCoord[0]*width + rootCoord[1]], scale_outs[0],zero_points[0]);
        rootImageCoord[1] =  (fix16_from_int(rootCoord[1]*outputStride)) + int8_to_fix16_single(offsets[(rootId+NUM_KEYPOINTS) * height*width + rootCoord[0]*width + rootCoord[1]], scale_outs[0],zero_points[0]);

        if (withinNmsRadiusOfCorrespondingPoint(poses, sqnmsRadius, rootImageCoord, queue[i].id, pose_count)){          
            continue;
        }
        
        decodePose_int8(rootScore,rootId,rootImageCoord,scores, offsets,outputStride, displacementsFwd, displacementsBwd,&temp_pose, height, width, zero_points, scale_outs); 

        poseScore = getInstanceScore(poses, sqnmsRadius, temp_pose.scores, temp_pose.keypoints, NUM_KEYPOINTS ,pose_count);

        if (poseScore>=minPoseScore){
            poses[pose_count].poseScore = poseScore;
            for(int i = 0; i< NUM_KEYPOINTS; i++ ){
                poses[pose_count].scores[i] = temp_pose.scores[i];
                poses[pose_count].keypoints[i][0] = temp_pose.keypoints[i][0];
                poses[pose_count].keypoints[i][1] = temp_pose.keypoints[i][1];               
            }
            pose_count++;
        }
        if(pose_count>=maxPoseDetections){
            break;
        }

    }
    return pose_count;
}



#ifdef __cplusplus
}
#endif