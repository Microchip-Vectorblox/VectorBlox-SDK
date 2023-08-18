#include "tracking.h"



#define PLATE 1
#if PLATE
const static float timeFactor = 30/15; //#30/FPS
#endif

static bool isNameSubset(char* name1, char* name2){
	if( (name1[0]=='\0') || (name2[0]=='\0') ) //check if either name is empty
		return false;
	if(strlen(name1)>=strlen(name2))
		return false;
	return (strstr(name1,name2)!= NULL ); //looking for substring and returns valid pointer if name 2 is in name1
}

static mf16 filterF;
static mf16 filterQ;
static mf16 filterH;
static mf16 filterR;

static void filtersInit(int use_plate){
	filterF.rows = 8;
	filterF.columns = 8;
	mf16_fill_diagonal(&filterF, fix16_one);
	filterF.data[0][4] = fix16_one;
	filterF.data[1][5] = fix16_one;
	filterF.data[2][6] = fix16_one;
	filterF.data[3][7] = fix16_one;

	fix16_t xAcc,yAcc,wAcc,hAcc,xVar,yVar,wVar,hVar;
	filterQ.rows = 8;
	filterQ.columns = 8;
	mf16_fill(&filterQ, 0);
	if(use_plate){
		xAcc = F16(0.05 * timeFactor * 4);
		yAcc = F16(0.01 * timeFactor * 4);
		wAcc = F16(0.025 * timeFactor * 4);
		hAcc = F16(0.005 * timeFactor * 4);
		xVar = F16(2);
		yVar = F16(2);
		wVar = F16(16);
		hVar = F16(4);
	}

	else{
		xAcc = F16(2);
		yAcc = F16(2);
		wAcc = F16(1);
		hAcc = F16(1);
		xVar = F16(50);
		yVar = F16(50);
		wVar = F16(200);
		hVar = F16(200);
	}



	filterQ.data[0][0] = xAcc>>2;
	filterQ.data[1][1] = yAcc>>2;
	filterQ.data[2][2] = wAcc>>2;
	filterQ.data[3][3] = hAcc>>2;
	filterQ.data[0][4] = filterQ.data[4][0] = xAcc>>1;
	filterQ.data[1][5] = filterQ.data[5][1] = yAcc>>1;
	filterQ.data[2][6] = filterQ.data[6][2] = wAcc>>1;
	filterQ.data[3][7] = filterQ.data[7][3] = hAcc>>1;
	filterQ.data[4][4] = xAcc;
	filterQ.data[5][5] = yAcc;
	filterQ.data[6][6] = wAcc;
	filterQ.data[7][7] = hAcc;


	filterH.rows = 4;
	filterH.columns = 8;
	mf16_fill(&filterH, 0);
	filterH.data[0][0] = fix16_one;
	filterH.data[1][1] = fix16_one;
	filterH.data[2][2] = fix16_one;
	filterH.data[3][3] = fix16_one;

	filterR.rows = 4;
	filterR.columns = 4;
	mf16_fill(&filterR, 0);

	filterR.data[0][0] = xVar;
	filterR.data[1][1] = yVar;
	filterR.data[2][2] = wVar;
	filterR.data[3][3] = hVar;

}

void kalmanFilterInit(fix16_t* box, kalmanFilter_t* filter,int use_plate) {
	filter->x.rows = 8;
	filter->x.columns = 1;
	filter->x.data[0][0] = (box[0]+box[2])>>1; // x
	filter->x.data[1][0] = (box[1]+box[3])>>1; // y
	filter->x.data[2][0] = (box[2]-box[0]); // w
	filter->x.data[3][0] = (box[3]-box[1]); // h
	filter->x.data[4][0] = 0;
	filter->x.data[5][0] = 0;
	filter->x.data[6][0] = 0;
	filter->x.data[7][0] = 0;

     fix16_t xVar;
     fix16_t yVar;
     fix16_t wVar;
     fix16_t hVar;
     fix16_t dxVar;
     fix16_t dyVar;
     fix16_t dwVar;
     fix16_t dhVar;
     if(use_plate){
    	 xVar = F16(2);
    	 yVar = F16(2);
    	 wVar = F16(16);
    	 hVar = F16(4);
    	 dxVar = F16(4);
    	 dyVar = F16(4);
    	 dwVar = F16(24);
    	 dhVar = F16(1);
     }
     else{
    	 xVar = F16(50);
    	 yVar = F16(50);
    	 wVar = F16(200);
    	 hVar = F16(200);
    	 dxVar = F16(10);
    	 dyVar = F16(10);
    	 dwVar = F16(5);
    	 dhVar = F16(5);

     }



	filter->P.rows = 8;
	filter->P.columns = 8;
	mf16_fill(&filter->P, 0);
	filter->P.data[0][0] = xVar;
	filter->P.data[1][1] = yVar;
	filter->P.data[2][2] = wVar;
	filter->P.data[3][3] = hVar;
	filter->P.data[4][4] = dxVar;
	filter->P.data[5][5] = dyVar;
	filter->P.data[6][6] = dwVar;
	filter->P.data[7][7] = dhVar;

}

void boxToObservation(fix16_t* box, mf16* obs) {
	obs->rows = 4;
	obs->columns = 1;
	obs->errors = 0;
	obs->data[0][0] = (box[0]+box[2])>>1; // x
	obs->data[1][0] = (box[1]+box[3])>>1; // y
	obs->data[2][0] = (box[2]-box[0]); // w
	obs->data[3][0] = (box[3]-box[1]); // h
}

void boxCoordinates(fix16_t* box, kalmanFilter_t* filter) {
	fix16_t wHalf = filter->x.data[2][0]>>1;
	fix16_t hHalf = filter->x.data[3][0]>>1;
	box[0] = filter->x.data[0][0]-wHalf;
	box[1] = filter->x.data[1][0]-hHalf;
	box[2] = filter->x.data[0][0]+wHalf;
	box[3] = filter->x.data[1][0]+hHalf;
}



// res  =  F  *  A
// (8x1 = 8x8 * 8x1)
void filter_FxA_8x1res(mf16* res, mf16* A){
	res->data[0][0] = fix16_add(A->data[0][0], A->data[4][0]);
	res->data[1][0] = fix16_add(A->data[1][0], A->data[5][0]);
	res->data[2][0] = fix16_add(A->data[2][0], A->data[6][0]);
	res->data[3][0] = fix16_add(A->data[3][0], A->data[7][0]);
	res->data[4][0] = A->data[4][0];
	res->data[5][0] = A->data[5][0];
	res->data[6][0] = A->data[6][0];
	res->data[7][0] = A->data[7][0];
}

// res  =  F  *  A
// (8x8 = 8x8 * 8x8)
void filter_FxA_8x8res(mf16* res, mf16* A){
	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 8; j++){
			res->data[i][j] = fix16_add(A->data[i][j], A->data[i+4][j]);
		}
	}

	for(int i = 4; i < 8; i++){
		for(int j = 0; j < 8; j++){
			res->data[i][j] = A->data[i][j];
		}
	}
}

// res  =  A  * F.T
// (8x8 = 8x8 * 8x8) 
void filter_AxFT_8x8res(mf16* res, mf16* A){
	for(int i = 0; i < 8; i++){
		for(int j = 0; j < 4; j++){
			res->data[i][j] = fix16_add(A->data[i][j], A->data[i][j+4]);
		}

		for(int j = 4; j < 8; j++){
			res->data[i][j] = A->data[i][j];
		}
	}
}

// res  =  H  *  A
// (4x8 = 4x8 * 8x8)
void filter_HxA_4x8res(mf16* res, mf16* A){
	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 8; j++){
			res->data[i][j] = A->data[i][j];
		}
	}
}

// res  =  A  * H.T
// (4x4 = 4x8 * 8x4)
void filter_AxHT_4x4res(mf16* res, mf16* A){
	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 4; j++){
			res->data[i][j] = A->data[i][j];
		}
	}
}

// res  =  A  *  H.T
// (8x4 = 8x8 * 8x4)
void filter_AxHT_8x4res(mf16* res, mf16* A){
	for(int i = 0; i < 8; i++){
		for(int j = 0; j < 4; j++){
			res->data[i][j] = A->data[i][j];
		}
	}
}

// res =  A   *  H
// (8x8 = 8x4 * 4x8)
void filter_AxH_8x8res(mf16* res, mf16* A){
	for(int i = 0; i < 8; i++){
		for(int j = 0; j < 4; j++){
			res->data[i][j] = A->data[i][j];
		}

		for(int j = 4; j < 8; j++){
			res->data[i][j] = 0;
		}
	}
}


void innovationResidual(mf16* boxToObs, mf16* y, kalmanFilter_t* filter) {
	mf16 x_apri;    // a priori state estimate
	x_apri.rows = 8;
	x_apri.columns = 1;
	filter_FxA_8x1res(&x_apri, &filter->x);    // mf16_mul(&x_apri, &filterF, &filter->x);    // x_apri = F*x

	/*
    mf16_mul(&x_apri, &filter->H, &x_apri);
    is equivalent to:
    x_apri.data[0][0] = x_apri.data[0][0];
    x_apri.data[1][0] = x_apri.data[1][0];
    x_apri.data[2][0] = x_apri.data[2][0];
    x_apri.data[3][0] = x_apri.data[3][0];
    so it can be omitted.
	 */

	mf16_sub(y, boxToObs, &x_apri);   // y = z - H*x_apri
}

void filterPredictUpdate(kalmanFilter_t* filter) {
	// Update with no new measurement, so Predict only
	// x = self.F*self.x;  a priori state estimate
	filter_FxA_8x1res(&filter->x, &filter->x);    // mf16_mul(&filter->x, &filterF, &filter->x);
	// P = self.F*self.P*np.transpose(self.F) + self.Q;  a priori covariance estimate
	filter_FxA_8x8res(&filter->P, &filter->P);    // mf16_mul(&filter->P, &filterF, &filter->P);
	filter_AxFT_8x8res(&filter->P, &filter->P);    // mf16_mul_bt(&filter->P, &filter->P, &filterF);
	mf16_add(&filter->P, &filter->P, &filterQ);
}

void filterUpdate(mf16* boxToObs, kalmanFilter_t* filter){
	// Predict
	// x_apri = self.F*self.x;  a priori state estimate
	mf16 x_apri;
	// mf16_mul(&x_apri, &filterF, &filter->x);
	x_apri.rows = 8;
	x_apri.columns = 1;
	filter_FxA_8x1res(&x_apri, &filter->x);
	// P_apri = self.F*self.P*np.transpose(self.F) + self.Q;  a priori covariance estimate
	mf16 P_apri;
	P_apri.rows = 8;
	P_apri.columns = 8;
	filter_FxA_8x8res(&P_apri, &filter->P);    // mf16_mul(&P_apri, &filterF, &filter->P);
	filter_AxFT_8x8res(&P_apri, &P_apri);    // mf16_mul_bt(&P_apri, &P_apri, &filterF);
	mf16_add(&P_apri, &P_apri, &filterQ);
	// Update
	// y = z - self.H*x_apri;  innovation residual
	mf16 y;
	// mf16_mul(&y, &filter->H,&x_apri);
	// mf16_sub(&y, boxToObs, &y);
	mf16_sub(&y, boxToObs, &x_apri);
	// S = self.H*P_apri*np.transpose(self.H) + self.R;  innovation covariance
	mf16 S;
	S.rows = 4;
	S.columns = 8;
	filter_HxA_4x8res(&S, &P_apri);    // mf16_mul(&S, &filterH, &P_apri);
	filter_AxHT_4x4res(&S, &S);    // mf16_mul_bt(&S, &S, &filterH);
	mf16_add(&S,&S, &filterR);
	// K = P_apri*np.transpose(self.H)*np.linalg.pinv(S);  optimal Kalman gain
	mf16 K;
	K.rows = 8;
	K.columns = 4;
	mf16_cholesky(&S, &S);
	mf16_invert_lt(&S, &S); // inverse S
	filter_AxHT_8x4res(&K, &P_apri); // mf16_mul_bt(&K, &P_apri, &filterH);
	mf16_mul(&K,&K,&S);
	// self.x = x_apri + K*y;  a posteriori state estimate
	mf16_mul(&filter->x, &K, &y);
	mf16_add(&filter->x, &filter->x, &x_apri);
	// self.P = (np.eye(6)-K*self.H)*P_apri;  a posteriori covariance estimate
	mf16_fill_diagonal(&filter->P, fix16_one);
	filter_AxH_8x8res(&K, &K);    // mf16_mul(&K, &K, &filterH);
	mf16_sub(&filter->P, &filter->P, &K);
	mf16_mul(&filter->P, &filter->P, &P_apri);
}



void trackerInit(Tracker_t* tracker, int max_tracks, track_t* pTracks[], track_t tracks[],int use_plate) {
	if(use_plate){
		tracker->residualWeights[0] = F16(0.25);
		tracker->residualWeights[1] = F16(1.0);
		tracker->residualWeights[2] = F16(0.5);
		tracker->residualWeights[3] = F16(1.0);
		tracker->maxResidual = F16(75);
		tracker->similarityAlphaUp = F16(0.995);
		tracker->similarityAlphaDown = F16(0.005);
		tracker->detectThresh = F16(2.6);
		tracker->maxMisses = 3;
	}
	else{
		tracker->residualWeights[0] = F16(0.5);
		tracker->residualWeights[1] = F16(0.5);
		tracker->residualWeights[2] = F16(0.5);
		tracker->residualWeights[3] = F16(0.5);
		tracker->maxResidual = F16(50);
		tracker->similarityAlphaUp = F16(0.95);
		tracker->similarityAlphaDown = F16(0.05);
		tracker->detectThresh = F16(0.45);
		tracker->maxMisses = 2;
	}
	tracker->ageAlphaUp = F16(0.2);
	tracker->ageAlphaDown = F16(0.02);
	tracker->genderAlphaUp = F16(0.1);
	tracker->genderAlphaDown = F16(0.01);
	tracker->tracksLength = 0;

for(int t = 0; t < max_tracks; t++)
	pTracks[t] = &tracks[t];

filtersInit(use_plate);
}

void updateDetection(object_t objects[], int length, Tracker_t* tracker, int max_tracks, track_t* pTracks[],int use_plate) {
	// calculate residual between each object and track
	fix16_t res[length][tracker->tracksLength];
	mf16 y;
	mf16 boxToObs;
	for(int f = 0; f < length; f++) {
		for(int t = 0; t < tracker->tracksLength; t++) {
			boxToObservation(objects[f].box, &boxToObs);
			innovationResidual(&boxToObs, &y, &pTracks[t]->filter);
			fix16_t residual = 0;
			for(int n = 0; n < 4; n++){
				residual = fix16_sadd(residual, fix16_ssq(fix16_smul(y.data[n][0], tracker->residualWeights[n])));
			}
			res[f][t] = fix16_sqrt(residual);
		}
	}
	int tracks[length];
	// clear existing connections between objects and tracks (objects have an index to the track, tracks have a pointer to the object)
	for(int f = 0; f < length; f++)
		tracks[f] = -1;
	for(int t = 0; t < tracker->tracksLength; t++) {
		pTracks[t]->object = NULL;
		pTracks[t]->framesSinceRec++;
	}

	// match objects to tracks, selecting closest matches first
	while(1) {
		fix16_t resMin = fix16_maximum;
		int fMin = 0;
		int tMin = 0;
		for(int f = 0; f < length; f++) {
			for(int t = 0; t < tracker->tracksLength; t++) {
				if(res[f][t] < resMin) {
					resMin = res[f][t];
					fMin = f;
					tMin = t;
				}
			}
		}

		if(resMin >= tracker->maxResidual)
			break;

		pTracks[tMin]->res = resMin;
		for(int t = 0; t < tracker->tracksLength; t++) {
			res[fMin][t] = fix16_maximum; // don't use this object again
		}
		for(int f = 0; f < length; f++) {   
			res[f][tMin] = fix16_maximum; // don't use this track again
		}

		tracks[fMin] = tMin;
		pTracks[tMin]->object = &objects[fMin];
	}

	for(int f = 0; f < length; f++) {
		int t = tracks[f];
		if(t == -1) {   // start new track
			if(tracker->tracksLength == max_tracks)   // don't start new track if we have the max number of tracks already
				continue;
			t = tracker->tracksLength;
			tracker->tracksLength++;
			kalmanFilterInit(objects[f].box, &pTracks[t]->filter,use_plate);
			pTracks[t]->frames = 0;
			pTracks[t]->misses = 0;
			pTracks[t]->res = 0;
			pTracks[t]->similarity = 0;
			strcpy(pTracks[t]->name, "");
			pTracks[t]->framesSinceRec = 100;
			tracks[f] = t;
			pTracks[t]->object = &objects[f];
		} else {   // update existing track
			boxToObservation(objects[f].box, &boxToObs);
			filterUpdate(&boxToObs, &pTracks[t]->filter);
			pTracks[t]->frames++;
		}
	}

	for(int t = 0; t < tracker->tracksLength; t++) {
		if(pTracks[t]->object != NULL) {
			pTracks[t]->misses = 0;
		} else {
			pTracks[t]->misses++; // count number of consecutive frames for which track is not matched
			if(pTracks[t]->misses <= tracker->maxMisses)
				filterPredictUpdate(&pTracks[t]->filter);
		}
	}

	// remove tracks with too many misses
	for(int t = 0; t < tracker->tracksLength; t++) {
		if(pTracks[t]->misses > tracker->maxMisses) {
			track_t* removedTrack = pTracks[t];   // swap removed track with the last track
			pTracks[t] = pTracks[tracker->tracksLength-1];
			pTracks[tracker->tracksLength-1] = removedTrack;
			tracker->tracksLength--;
		}
	}

	// next object to recognize
	int maxFrames = -1;
	tracker->recognitionTrackInd = -1;
	for(int t = 0; t < tracker->tracksLength; t++) {
		if(pTracks[t]->object) {
			if(pTracks[t]->framesSinceRec > maxFrames) {
				maxFrames = pTracks[t]->framesSinceRec;
				tracker->recognitionTrackInd = t;
			}
		}
	}
}

void matchTracks(object_t objects[], int length, Tracker_t* tracker, int max_tracks, track_t* pTracks[], int tracks[],int use_plate) {
	// calculate residual between each object and track
	fix16_t res[length][tracker->tracksLength];
	mf16 y;
	mf16 boxToObs;
	for(int f = 0; f < length; f++) {
		for(int t = 0; t < tracker->tracksLength; t++) {
			boxToObservation(objects[f].box, &boxToObs);
			innovationResidual(&boxToObs, &y, &pTracks[t]->filter);
			fix16_t residual = 0;
			for(int n = 0; n < 4; n++){
				residual = fix16_sadd(residual, fix16_ssq(fix16_smul(y.data[n][0], tracker->residualWeights[n])));
			}
			res[f][t] = fix16_sqrt(residual);
		}
	}

	// clear existing connections between objects and tracks (objects have an index to the track, tracks have a pointer to the object)
	for(int f = 0; f < length; f++)
		tracks[f] = -1;
	for(int t = 0; t < tracker->tracksLength; t++) {
		pTracks[t]->object = NULL;
		pTracks[t]->framesSinceRec++;
	}

	// match objects to tracks, selecting closest matches first
	while(1) {
		fix16_t resMin = fix16_maximum;
		int fMin = 0;
		int tMin = 0;
		for(int f = 0; f < length; f++) {
			for(int t = 0; t < tracker->tracksLength; t++) {
				if(res[f][t] < resMin) {
					resMin = res[f][t];
					fMin = f;
					tMin = t;
				}
			}
		}

		if(resMin >= tracker->maxResidual)
			break;

		pTracks[tMin]->res = resMin;
		for(int t = 0; t < tracker->tracksLength; t++) {
			res[fMin][t] = fix16_maximum;   // don't use this object again
		}
		for(int f = 0; f < length; f++)  {  // don't use this track again
			res[f][tMin] = fix16_maximum;
		}

		tracks[fMin] = tMin;
		pTracks[tMin]->object = &objects[fMin];
	}

	for(int f = 0; f < length; f++) {
		int t = tracks[f];
		if(t == -1) {   // start new track
			if(tracker->tracksLength == max_tracks)   // don't start new track if we have the max number of tracks already
				continue;
			t = tracker->tracksLength;
			tracker->tracksLength++;
			kalmanFilterInit(objects[f].box, &pTracks[t]->filter,use_plate);
			pTracks[t]->frames = 0;
			pTracks[t]->misses = 0;
			pTracks[t]->res = 0;
			pTracks[t]->similarity = 0;
			strcpy(pTracks[t]->name, "");
			pTracks[t]->framesSinceRec = 100;
			tracks[f] = t;
			pTracks[t]->object = &objects[f];
		}
	}

	for(int t = 0; t < tracker->tracksLength; t++) {
		if(pTracks[t]->object != NULL) {
			pTracks[t]->misses = 0;
		} else {
			pTracks[t]->misses++; // count number of consecutive frames for which track is not matched

			// remove tracks with too many misses
			if(pTracks[t]->misses > tracker->maxMisses){
				track_t* removedTrack = pTracks[t];   // swap removed track with the last track
				pTracks[t] = pTracks[tracker->tracksLength-1];
				pTracks[tracker->tracksLength-1] = removedTrack;
				tracker->tracksLength--;
			}
		}
	}

	// next object to recognize
	int maxFrames = -1;
	tracker->recognitionTrackInd = -1;
	for(int t = 0; t < tracker->tracksLength; t++) {
		if(pTracks[t]->object != NULL) {
			if(pTracks[t]->framesSinceRec > maxFrames) {
				maxFrames = pTracks[t]->framesSinceRec;
				tracker->recognitionTrackInd = t;
			}
		}
	}
}

void updateFilters(object_t objects[], int length, Tracker_t* tracker, track_t* pTracks[], int tracks[]) {

	mf16 boxToObs;
	for(int f = 0; f < length; f++){
		int t = tracks[f];
		if(t != -1) {   // update existing track
			boxToObservation(objects[f].box, &boxToObs);
			filterUpdate(&boxToObs, &pTracks[t]->filter);
			pTracks[t]->frames++;
		}
	}

	for(int t = 0; t < tracker->tracksLength; t++) {
		if(pTracks[t]->object == NULL) {
			if(pTracks[t]->misses <= tracker->maxMisses)
				filterPredictUpdate(&pTracks[t]->filter);
		}
	}
}

void updateRecognition(track_t* tracks[], int trackInd, fix16_t similarity, char* name, Tracker_t* tracker) {
	fix16_t alpha;  // exponential smoothing factor
	track_t* track = tracks[trackInd];
	if(strcmp(name, track->name)) {

		if((similarity > tracker->detectThresh) && (similarity > track->similarity || (isNameSubset(track->name, name))) && !isNameSubset(name, track->name)){
			alpha = tracker->similarityAlphaUp;
			track->similarity = 0;
			strcpy(track->name, name);
		}
		else{
			alpha = tracker->similarityAlphaDown;
			similarity = 0;
		}
	}
	else{
		if(similarity>track->similarity)
			alpha = tracker->similarityAlphaUp;
		else
			alpha = tracker->similarityAlphaDown;
	}
	track->similarity = fix16_mul(alpha,similarity) + fix16_mul(fix16_one-alpha, track->similarity);    // exponential smoothing
	track->framesSinceRec = 0;
}

void updateAttribution(track_t* track, fix16_t gender, fix16_t age, Tracker_t* tracker, int is_frontal_face){
	fix16_t age_alpha = tracker->ageAlphaUp;
	fix16_t gender_alpha = tracker->genderAlphaUp;

	if(is_frontal_face == 0){ // not a frontal view of the face, lower the alphas
		age_alpha = tracker->ageAlphaDown;
		gender_alpha = tracker->genderAlphaDown;
	}

	if(track->atrMeas == 0){
		track->gender = gender;
		track->age = age;
	}
	else if(track->atrMeas < 30){
		// average attributes
		track->gender = fix16_div((gender + (track->atrMeas*track->gender)) , fix16_from_int(track->atrMeas+1));
		track->age = fix16_div((age + (track->atrMeas*track->age)) , fix16_from_int(track->atrMeas+1));
	}
	else{
		track->gender = fix16_mul(gender_alpha, gender) + fix16_mul(fix16_one-gender_alpha, track->gender);
		track->age = fix16_mul(age_alpha, age) + fix16_mul(fix16_one-age_alpha, track->age);
	}
	track->atrMeas++;
}

