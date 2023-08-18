#ifndef _TRACKING_H
#define _TRACKING_H

#include "libfixmatrix/fixmatrix.h"
#include "postprocess.h"
#include <string.h>

#define TRACKING
#define NAME_LENGTH 12

typedef struct {
    mf16 x; // state estimate [w,y,width,dx,dy,dwidth]
    mf16 P; // covariance matrix estimate
} kalmanFilter_t;

typedef struct {
    kalmanFilter_t filter;
    int frames;
    int framesSinceRec;
    object_t* object; // bool match;
    int misses;
    char name[NAME_LENGTH];
    fix16_t similarity;
    fix16_t res;
    fix16_t gender;
    fix16_t age;
    int atrMeas;
} track_t;

typedef struct {
    fix16_t residualWeights[4];
    fix16_t maxResidual;
    fix16_t similarityAlphaUp;
    fix16_t similarityAlphaDown;
    fix16_t detectThresh;
    int maxMisses;
    fix16_t ageAlphaUp;
    fix16_t genderAlphaUp;
    fix16_t ageAlphaDown;
    fix16_t genderAlphaDown;
    int tracksLength;
    int recognitionTrackInd;
} Tracker_t;


void boxCoordinates(fix16_t* box, kalmanFilter_t* filter);
void trackerInit(Tracker_t* tracker, int max_tracks, track_t* pTracks[], track_t tracks[],int use_plate);
void updateDetection(object_t objects[], int length, Tracker_t* tracker, int max_tracks, track_t* pTracks[],int use_plates);
void updateRecognition(track_t* pTracks[], int trackInd, fix16_t similarity, char* name, Tracker_t* tracker);
void updateAttribution(track_t* track, fix16_t gender, fix16_t age, Tracker_t* tracker, int is_frontal_face);

void matchTracks(object_t objects[], int length, Tracker_t* tracker, int max_tracks, track_t* pTracks[], int tracks[],int use_plate);
void updateFilters(object_t objects[], int length, Tracker_t* tracker, track_t* pTracks[], int tracks[]);

#endif
