// This is the C translation of sdk/example/python/space.py
// 
#include "postprocess.h"
#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// This is set for
#define SPACE_IMAGE_WIDTH  1920
#define SPACE_IMAGE_HEIGHT 1080



// convert raw rotation coordinates into cosine matrix
void rot_6d_to_matrix(float *dcm, float* raw)
{
    float r1[3];
    r1[0] = raw[0]; r1[1] = raw[1]; r1[2] = raw[2];
    float r1_norm = sqrt(r1[0]*r1[0] + r1[1]*r1[1] + r1[2]*r1[2]);
    r1[0] = r1[0] / r1_norm;
    r1[1] = r1[1] / r1_norm;
    r1[2] = r1[2] / r1_norm;

    float r2[3];
    r2[0] = raw[3]; r2[1] = raw[4]; r2[2] = raw[5];
    float dot = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2];
    r2[0] = r2[0] - dot * r1[0];
    r2[1] = r2[1] - dot * r1[1];
    r2[2] = r2[2] - dot * r1[2];

    float r2_norm = sqrt(r2[0]*r2[0] + r2[1]*r2[1] + r2[2]*r2[2]);
    r2[0] = r2[0] / r2_norm;
    r2[1] = r2[1] / r2_norm;
    r2[2] = r2[2] / r2_norm;

    float r3[3];
    r3[0] = (r1[1] * r2[2]) - (r1[2] * r2[1]); //a2b3-a3b2
    r3[1] = (r1[2] * r2[0]) - (r1[0] * r2[2]); //a3b1-a1b3
    r3[2] = (r1[0] * r2[1]) - (r1[1] * r2[0]); //a1b2-a2b1

    dcm[0] = r1[0];
    dcm[1] = r2[0];
    dcm[2] = r3[0];

    dcm[3] = r1[1];
    dcm[4] = r2[1];
    dcm[5] = r3[1];

    dcm[6] = r1[2];
    dcm[7] = r2[2];
    dcm[8] = r3[2];
}


// convert raw translation values into final translation values
void delta_xy_tz_to_translation(float *t, float *translation_raw, float *cameraMatrix, float* anchors, float strides, int image_height, int image_width, int input_height, int input_width)
{
    t[2] = translation_raw[2];
    float cx = translation_raw[0] * strides + (anchors[0] + anchors[2])/2;
    float cy = translation_raw[1] * strides + (anchors[1] + anchors[3])/2;

    cx = cx / input_width * image_width;
    cy = cy / input_height * image_height;

    t[0] = (cx - cameraMatrix[2]) * t[2] / cameraMatrix[0];
    t[1] = (cy - cameraMatrix[3+2]) * t[2] / cameraMatrix[3+1];
}


// fix16 only holds +/-32767; clamp (and zero NaN) so off-screen points don't overflow
static fix16_t safe_fix16_from_float(float v)
{
	if (v != v) return 0;
	if (v > 32000.f) v = 32000.f;
	if (v < -32000.f) v = -32000.f;
	return fix16_from_float(v);
}

// convert final 6D position into a 2D projection
void project_keypoints(fix16_t* points2D, float *dcm, float *r, float *K, float *dist, float *kpts)
{
	float xyz[36];
	float x[12];
	float y[12];

	for (int i=0; i < 3; i++) {
		for (int j=0; j < 12; j++) {
			xyz[i*12+j] = kpts[j]*dcm[i] + kpts[12+j]*dcm[3+i] + kpts[24+j]*dcm[6+i] + kpts[36+j]*r[i];
		}
	}
	float x0,y0,r2,cdist;
	for (int i=0; i < 12; i++) {
		x0 = xyz[i] / xyz[24+i];
		y0 = xyz[12+i] / xyz[24+i];
		r2 = x0*x0 + y0*y0;
		cdist = 1 + dist[0]*r2 + dist[1]*r2*r2 + dist[4]*r2*r2*r2;
		x[i] = x0*cdist + dist[2]*2*x0*y0 + dist[3]*(r2 + 2*x0*x0);
		y[i] = y0*cdist + dist[2]*(r2+2*y0*y0) + dist[3]*(2*x0*y0);
		points2D[i] = safe_fix16_from_float(K[0]*x[i] + K[2]);
		points2D[12+i] = safe_fix16_from_float(K[3+1]*y[i] + K[3+2]);
	}
}


// get stride for output set
int get_strides(int set)
{
	int pyramid_levels[5] = {7,6,5,4,3};
	return 1<<pyramid_levels[set];
}

// get xmin,ymin,xmax,ymax anchors for specific output
void get_anchors(float *anchors, int set, int c, int h, int w)
{

	int stride = get_strides(set);
	float scales[3]  = {1., 1.2599210498948732, 1.5874010519681994};
	float ratio_x[3]  = {1., 1.4, 0.7};
	float ratio_y[3]  = {1., 0.7, 1.4};
	int anchor_scale = 4;
	float base_anchor_size = anchor_scale * stride * scales[c/3];

	float anchor_size_x_2 = base_anchor_size * ratio_x[c%3] / 2;
	float anchor_size_y_2 = base_anchor_size * ratio_y[c%3] / 2;

	int x = stride*w + stride/2;
	int y = stride*h + stride/2;

	float xmin = x - anchor_size_x_2;
	float ymin = y - anchor_size_y_2;
	float xmax = x + anchor_size_x_2;
	float ymax = y + anchor_size_y_2;

	anchors[0] = xmin;
	anchors[1] = ymin;
	anchors[2] = xmax;
	anchors[3] = ymax;
}

// convert raw coordinates into final bounding box, adjusting with anchors
void raw_output_to_bbox(float *output, float *bbox_raw, float *anchor, int image_height, int image_width)
{
    float anchor_width    = anchor[2] - anchor[0];
    float anchor_height   = anchor[3] - anchor[1];
    float anchor_center_x = anchor[0] + anchor_width/2;
    float anchor_center_y = anchor[1] + anchor_height/2;

    float x = anchor_center_x + anchor_width * bbox_raw[0];
    float y = anchor_center_y + anchor_height * bbox_raw[1];
    float w = anchor_width * exp(bbox_raw[2]);
    float h = anchor_height * exp(bbox_raw[3]);

    output[0] = x - w/2;
    if (output[0] < 0) output[0] = 0;
    if (output[0] > image_width-1) output[0] = image_width-1;

    output[1] = y - h/2;
    if (output[1] < 0) output[1] = 0;
    if (output[1] > image_height-1) output[1] = image_height-1;

    output[2] = x + w/2;
    if (output[2] < 0) output[2] = 0;
    if (output[2] > image_width-1) output[2] = image_width-1;

    output[3] = y + h/2;
    if (output[3] < 0) output[3] = 0;
    if (output[3] > image_height-1) output[3] = image_height-1;
}

// return score and 2D projection of highest scoring output
fix16_t post_process_space_int8(fix16_t *points2D, int8_t **output_buffers_int8, model_t *model, int transpose_first)
{

	float anchors[4];
	float strides;
	float raw_bbox[4];/// = {0., -0.08082398, 0.09237026, 0.4849439};
	float translation_raw[3];// = {-0.20935534,-0.6699371,2.6797483};
	float rotation_raw[6];// = {-4.2851324, 3.4519122, -0.7141887, 1.7854718, 4.166101, 1.9045032};
	float bbox[4];
	float t[3];
	float dcm[9];

	// coeffs taken from SPNV2 repo
	float cameraMatrix[9] = {2988.5795163815555,0,960, 0,2988.3401159176124,600, 0.,0.,1.};
	float distCoeffs[6] = {-0.22383016606510672, 0.51409797089106379, -0.00066499611998340662, -0.00021404771667484594, -0.13124227429077406};
	float tangoKeypoints[48] = {0.,-0.37, -0.37, 0.37, 0.37, -0.37, -0.37, 0.37, 0.37, -0.5427, 0.5427, 0.305,
		0.,-0.385, 0.385, 0.385, -0.385, -0.264, 0.304, 0.304, -0.264, 0.4877, 0.4877, -0.579,
		0.,0.3215, 0.3215, 0.3215, 0.3215, 0., 0., 0., 0., 0.2535, 0.2591, 0.2515,
		1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.};

	int num_outputs = (int)model_get_num_outputs(model);
	fix16_t max_val = F16(-999);


	if (num_outputs == 15) {
		// yolo-spn with split outputs (see space.py / yolo-spn_prep_onnx.py)
		// per stride set g (largest stride first): [box(64), det(1), pose_xy(2), pose_z(1), pose_rot(6)]
		const int expected_c[5] = {64, 1, 2, 1, 6};
		for (int o = 0; o < num_outputs; o++) {
			int *od = model_get_output_shape(model, o);
			if (model_get_output_dims(model, o) != 4 || od[1] != expected_c[o % 5]) {
				printf("post_process_space: unexpected yolo-spn output %d (channels %d, expected %d)\n",
						o, od[1], expected_c[o % 5]);
				return 0;
			}
		}

		// principal point is the image center, as in space.py (original size is 1920x1200; if smaller, assume cropped)
		cameraMatrix[2] = SPACE_IMAGE_WIDTH/2.0f;
		cameraMatrix[5] = SPACE_IMAGE_HEIGHT/2.0f;

		int input_dims = model_get_input_dims(model, 0);
		int *in_shape = model_get_input_shape(model, 0);
		int input_height = in_shape[input_dims-2];
		int input_width = in_shape[input_dims-1];

		// det_max = [d.max() for d in det]; stride_ind = argmax(det_max); max_ind = det[stride_ind].argmax()
		int max_set = 0, max_x = 0, max_y = 0;
		for (int g = 0; g < 3; g++) {
			int d = g*5 + 1;
			int *dd = model_get_output_shape(model, d);
			int size = dd[2] * dd[3];
			int8_t *det8 = output_buffers_int8[d];
			int local_idx = 0;
			int local_max = det8[0];
			for (int s = 1; s < size; s++) {
				if (det8[s] > local_max) {
					local_max = det8[s];
					local_idx = s;
				}
			}
			fix16_t val = int8_to_fix16_single((int8_t)local_max,
					model_get_output_scale_fix16_value(model, d), model_get_output_zeropoint(model, d));
			if (val > max_val) {
				max_val = val;
				max_set = g;
				max_y = local_idx / dd[3];
				max_x = local_idx % dd[3];
			}
		}

		// space.py default threshold; below it no spacecraft is reported
		if (max_val <= F16(0.90)) {
			return 0;
		}

		int box_o = max_set*5 + 0;
		int xy_o  = max_set*5 + 2;
		int z_o   = max_set*5 + 3;
		int rot_o = max_set*5 + 4;
		int *sd = model_get_output_shape(model, xy_o);
		int H = sd[2], W = sd[3];
		strides = (float)(input_width / W);

		// anchor_x, anchor_y = max_x*stride + stride//2, max_y*stride + stride//2
		anchors[0] = max_x*strides + (float)((int)strides/2);
		anchors[1] = max_y*strides + (float)((int)strides/2);
		anchors[2] = anchors[0];
		anchors[3] = anchors[1];

		// raw_pos = concat(pose_xy[:,max_y,max_x], pose_z[:,max_y,max_x])
		int8_t *xy8 = output_buffers_int8[xy_o];
		fix16_t xy_scale = model_get_output_scale_fix16_value(model, xy_o);
		int32_t xy_zp = model_get_output_zeropoint(model, xy_o);
		for (int i = 0; i < 2; i++) {
			translation_raw[i] = fix16_to_float(int8_to_fix16_single(xy8[i*H*W + max_y*W + max_x], xy_scale, xy_zp));
		}
		int8_t *z8 = output_buffers_int8[z_o];
		translation_raw[2] = fix16_to_float(int8_to_fix16_single(z8[max_y*W + max_x],
					model_get_output_scale_fix16_value(model, z_o), model_get_output_zeropoint(model, z_o)));

		// raw_rot = pose_rot[:,max_y,max_x]
		int8_t *rot8 = output_buffers_int8[rot_o];
		fix16_t rot_scale = model_get_output_scale_fix16_value(model, rot_o);
		int32_t rot_zp = model_get_output_zeropoint(model, rot_o);
		for (int i = 0; i < 6; i++) {
			rotation_raw[i] = fix16_to_float(int8_to_fix16_single(rot8[i*H*W + max_y*W + max_x], rot_scale, rot_zp));
		}

		// bounding box (DFL decode of the winning cell), scaled to the original image like space.py
		fix16_t xywh[4];
		int *bd = model_get_output_shape(model, box_o);
		ultralytics_process_box_int8(xywh, output_buffers_int8[box_o], fix16_minimum, max_y, max_x, bd[2], bd[3],
				model_get_output_zeropoint(model, box_o), model_get_output_scale_fix16_value(model, box_o));
		float x = fix16_to_float(xywh[0]), y = fix16_to_float(xywh[1]);
		float w = fix16_to_float(xywh[2]), h = fix16_to_float(xywh[3]);
		bbox[0] = (x - w/2) * SPACE_IMAGE_WIDTH;
		bbox[1] = (y - h/2) * SPACE_IMAGE_HEIGHT;
		bbox[2] = bbox[0] + w * SPACE_IMAGE_WIDTH;
		bbox[3] = bbox[1] + h * SPACE_IMAGE_HEIGHT;

		// scale z to original model size (space.py: tz *= (1920/768)/(image_size[0]/input_size[0]))
		float translation_scaled[3] = {translation_raw[0], translation_raw[1],
			translation_raw[2] * (1920.f/768.f) / ((float)SPACE_IMAGE_WIDTH / (float)input_width)};

		delta_xy_tz_to_translation(t, translation_scaled, cameraMatrix, anchors, strides,
				SPACE_IMAGE_HEIGHT, SPACE_IMAGE_WIDTH, input_height, input_width);
		rot_6d_to_matrix(dcm, rotation_raw);
		project_keypoints(points2D, dcm, t, cameraMatrix, distCoeffs, tangoKeypoints);

		return max_val;
	}

	// sort outputs, smallest to largest
	int cls[5], tr0[5], tr1[5], box[5], rot[5];
	int is_yolo = num_outputs == 9;

	if (is_yolo) {
		for(int o=0; o<3; o++) { // 3 output sizes 16,32,64
			for(int i=0; i< num_outputs; i++){
				int *out_dims = model_get_output_shape(model,i);
				if (out_dims[2] == 16*1<<o) {
					if (out_dims[1] == 9) {
						rot[o] = i;
					} else if (out_dims[1] == 64) {
						box[o] = i;
					} else if (out_dims[1] == 1) {
						cls[o] = i;
					} else {
						printf("shouldn't be here\n");
					}
				}
			}
		}
	} else {
		for(int o=6; o>1; o--){ // 5 output sizes 4,8,16,32,64
			int output_seen = 0;
			for(int i=0; i< num_outputs; i++){
				int *out_dims = model_get_output_shape(model,i);
				if (out_dims[2] == 1<<o) {
					if (out_dims[1] == 9) {
						if ((transpose_first && output_seen) || (!transpose_first && !output_seen)) {
							cls[o-2] = i;
						} else {
							tr0[o-2] = i;
						}
						output_seen = 1;
					} else if (out_dims[1] == 18) {
						tr1[o-2] = i;
					} else if (out_dims[1] == 36) {
						box[o-2] = i;
					} else {
						rot[o-2] = i;
					}
				}
			}
		}
	}

	//get max class
	int max_set = 0;
	int max_idx = 0;
	int max_c = 0;
	int max_h = 0;
	int max_w = 0;
	int anchor_h = 0;
	int anchor_w = 0;
	int osizes = is_yolo ? 2 : 4;

	for(int o=osizes; o>=0; o--) { // output sizes from largest to smallest: 64,32,16,8,4
		int i = cls[o];
		int *out_dims = model_get_output_shape(model,i);

		int odims = model_get_output_dims(model,i);
		int out_size = 1;
		for (int d = 0; d < odims; d++) {
			out_size *= out_dims[d];
		}

		// find local max. could be an argmax
		int local_max_idx = 0;
		int local_max_val = -255;
		int8_t* out8 = output_buffers_int8[i];
		for (int s = 0; s < out_size; s++) {
			if (out8[s] > local_max_val) {
				local_max_idx = s;
				local_max_val = out8[s];
			}
		}

		// check if local max greater than other outputs
		int32_t zero_point = model_get_output_zeropoint(model,i);
		fix16_t f16_scale = model_get_output_scale_fix16_value(model,i);
		fix16_t val = fix16_mul(fix16_from_int((int32_t)(local_max_val) - zero_point),f16_scale);

		if (val > max_val) {
			max_val = val;
			max_set = o;
			max_idx = local_max_idx;
			max_c = local_max_idx / (out_dims[odims-1]*out_dims[odims-2]);
			max_w = local_max_idx % out_dims[odims-1];
			max_h = (local_max_idx - max_c * (out_dims[odims-1]*out_dims[odims-2])) / out_dims[odims-1];
			anchor_w = local_max_idx % out_dims[odims-2];
			anchor_h = (local_max_idx - max_c * (out_dims[odims-2]*out_dims[odims-1])) / out_dims[odims-2];
		}
	}

	// get bbox
	if (is_yolo) {
		int32_t zero_point;
		fix16_t f16_scale;

		fix16_t xywh[4];
		int b = box[max_set];
		int *box_dims = model_get_output_shape(model,b);
		int8_t* box8 = output_buffers_int8[b];
		zero_point = model_get_output_zeropoint(model,b);
		f16_scale = model_get_output_scale_fix16_value(model,b);

		fix16_t angle = fix16_minimum;
		ultralytics_process_box_int8(xywh, box8, angle, max_h, max_w, box_dims[2], box_dims[3], zero_point, f16_scale);

		fix16_t bx0 = fix16_mul(fix16_sub(xywh[0], fix16_mul(xywh[2],F16(0.5))), F16(768));
		fix16_t by0 = fix16_mul(fix16_sub(xywh[1], fix16_mul(xywh[3],F16(0.5))), F16(512));
		bbox[0] = fix16_to_float(bx0);
		bbox[1] = fix16_to_float(by0);
		bbox[2] = fix16_to_float(fix16_add(fix16_mul(xywh[2], F16(768)), bx0));
		bbox[3] = fix16_to_float(fix16_add(fix16_mul(xywh[3], F16(512)), by0));

		if (max_set == 0) strides = 32;
		else if (max_set == 1) strides = 16;
		else strides = 8;

		anchors[0] = max_w*strides + strides/2;
		anchors[1] = max_h*strides + strides/2;
		anchors[2] = anchors[0];
		anchors[3] = anchors[1];

		int r = rot[max_set];
		int* rshape = model_get_output_shape(model,r);
		int8_t* rot8 = output_buffers_int8[r];
		zero_point = model_get_output_zeropoint(model,r);
		f16_scale = model_get_output_scale_fix16_value(model,r);

		for (int i=0; i<3; i++) {
			int c = (i+max_c*9); // first 3 values
			int idx = c*rshape[2]*rshape[3]+max_h*rshape[3]+max_w;
			translation_raw[i] = fix16_to_float(fix16_mul(fix16_from_int((int32_t)(rot8[idx]) - zero_point),f16_scale));
		}
		for (int i=0; i<6; i++) {
			int c = (i+3+max_c*9); // remaining 6 values;
			int idx = c*rshape[2]*rshape[3]+max_h*rshape[3]+max_w;
			rotation_raw[i] = fix16_to_float(fix16_mul(fix16_from_int((int32_t)(rot8[idx]) - zero_point),f16_scale));
		}

	} else {
		get_anchors(anchors, max_set, max_c, anchor_h, anchor_w);
		strides = 1.0 * get_strides(max_set);

		int idx, c;
		int32_t zero_point;
		fix16_t f16_scale;


		int b = box[max_set];
		int* bshape = model_get_output_shape(model,b);
		int8_t* box8 = output_buffers_int8[b];
		zero_point = model_get_output_zeropoint(model,b);
		f16_scale = model_get_output_scale_fix16_value(model,b);
		for (int i=0; i<4; i++) {
			c = (i+max_c*4); // bshape / 9
			idx = c*bshape[2]*bshape[3]+max_h*bshape[3]+max_w;
			raw_bbox[i] = fix16_to_float(fix16_mul(fix16_from_int((int32_t)(box8[idx]) - zero_point),f16_scale));
		}

		int t0 = tr0[max_set];
		int t1 = tr1[max_set];
		int* t0shape = model_get_output_shape(model,t0);
		int* t1shape = model_get_output_shape(model,t1);
		int8_t* tr08 = output_buffers_int8[t0];
		int8_t* tr18 = output_buffers_int8[t1];

		for (int i=0; i<3; i++) {
			c = (i+max_c*3); // bshape / 9
			if (c >= 18) {
				c -= 18;
				idx = c*t0shape[2]*t0shape[3]+max_h*t0shape[3]+max_w;
				zero_point = model_get_output_zeropoint(model,t0);
				f16_scale = model_get_output_scale_fix16_value(model,t0);
				translation_raw[i] = fix16_to_float(fix16_mul(fix16_from_int((int32_t)(tr08[idx]) - zero_point),f16_scale));
			} else {
				idx = c*t1shape[2]*t1shape[3]+max_h*t1shape[3]+max_w;
				zero_point = model_get_output_zeropoint(model,t1);
				f16_scale = model_get_output_scale_fix16_value(model,t1);
				translation_raw[i] = fix16_to_float(fix16_mul(fix16_from_int((int32_t)(tr18[idx]) - zero_point),f16_scale));
			}
		}

		int r = rot[max_set];
		int* rshape = model_get_output_shape(model,r);
		int8_t* rot8 = output_buffers_int8[r];
		zero_point = model_get_output_zeropoint(model,r);
		f16_scale = model_get_output_scale_fix16_value(model,r);
		for (int i=0; i<6; i++) {
			int c = (i+max_c*6); // rshape / 9
			int idx = c*rshape[2]*rshape[3]+max_h*rshape[3]+max_w;
			rotation_raw[i] = fix16_to_float(fix16_mul(fix16_from_int((int32_t)(rot8[idx]) - zero_point),f16_scale));
		}

		raw_output_to_bbox(bbox, raw_bbox, anchors, 512, 768);
	}


	// calulate translation, rotation and  project
	delta_xy_tz_to_translation(t, translation_raw, cameraMatrix, anchors, strides, 1080, 1920, 512, 768);
	rot_6d_to_matrix(dcm, rotation_raw);
	project_keypoints(points2D, dcm, t, cameraMatrix, distCoeffs, tangoKeypoints);


	return max_val;
}


#ifdef __cplusplus
}
#endif
