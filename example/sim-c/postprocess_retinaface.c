#include "postprocess.h"
#include <stdio.h>

typedef struct {
    int shape[4];
    fix16_t height[6];
    fix16_t width[6];
    fix16_t offset;
    fix16_t variance[4];
} prior_t;

// this macro is necessary instead of fix16_from_float() because this way is
// done at compiletime
#define fix16_conv(a) (fix16_t)((a) * (1 << 16))

static prior_t priors[] = {
    {
        .shape = {1, 8, 40, 40},
        .height = {fix16_conv(16), fix16_conv(32)},
        .width = {fix16_conv(16), fix16_conv(32)},
        .offset = fix16_conv(0.5),
        .variance = {fix16_conv(0.1), fix16_conv(0.1), fix16_conv(0.2),
                     fix16_conv(0.2)},
    },
    {
        .shape = {1, 8, 20, 20},
        .height = {fix16_conv(64), fix16_conv(128)},
        .width = {fix16_conv(64), fix16_conv(128)},
        .offset = fix16_conv(0.5),
        .variance = {fix16_conv(0.1), fix16_conv(0.1), fix16_conv(0.2),
                     fix16_conv(0.2)},
    },
    {
        .shape = {1, 8, 10, 10},
        .height = {fix16_conv(256), fix16_conv(512)},
        .width = {fix16_conv(256), fix16_conv(512)},
        .offset = fix16_conv(0.5),
        .variance = {fix16_conv(0.1), fix16_conv(0.1), fix16_conv(0.2),
                     fix16_conv(0.2)},
    },
};

typedef struct{
    fix16_t cx,cy,w,h;
} prior_box;

static prior_box gen_prior_box(prior_t *prior, int x, int y, int r, int image_width, int image_height) {
    fix16_t fix_x = fix16_from_int(x);
    fix16_t fix_y = fix16_from_int(y);
    fix16_t fix_r = fix16_from_int(r);
    fix16_t h = fix16_from_int(prior->shape[2] * image_height/320);
    fix16_t w = fix16_from_int(prior->shape[3] * image_width/320);
    int img_h = fix16_from_int(image_height);
    int img_w = fix16_from_int(image_width);

    fix16_t step_w = fix16_div(img_w, w);
    fix16_t step_h = fix16_div(img_h, h);
    fix16_t offset = prior->offset;
    fix16_t center_x = fix16_mul(fix_x + offset, step_w);
    fix16_t center_y = fix16_mul(fix_y + offset, step_h);
    fix16_t box_h = prior->height[r];
    fix16_t box_w = prior->width[r];

    prior_box ret_box;

    ret_box.cx = center_x;
    ret_box.cy = center_y;
    ret_box.w  = box_w;
    ret_box.h  = box_h;

    return ret_box;
}

static inline fix16_t clamp(fix16_t val,fix16_t low,fix16_t high) {
    if (val < low)
        return low;
    if (val > high)
        return high;
    return val;
}

static face_t get_face(fix16_t *box_output,fix16_t *landmark_output,int x,int y,int r,prior_t* prior, int image_width, int image_height){

    int grid_h = prior->shape[2] * image_height/320;
    int grid_w = prior->shape[3] * image_width/320;
    prior_box pbox = gen_prior_box(prior, x, y, r, image_width, image_height);
    fix16_t pw = pbox.w;
    fix16_t ph = pbox.h;

    fix16_t pcx = pbox.cx;
    fix16_t pcy = pbox.cy;
    //fix16_box base_box;
    int num_landmarks =5;
    face_t face;
    for(int l=0;l<num_landmarks;++l){
        fix16_t lm_x = landmark_output[r*grid_h*grid_w * num_landmarks*2 + grid_h * grid_w * (2*l) + grid_h * y + x];
        fix16_t lm_y = landmark_output[r*grid_h*grid_w * num_landmarks*2 + grid_h * grid_w * (2*l+1) + grid_h * y + x];
        lm_x = fix16_mul(fix16_mul(prior->variance[0], lm_x), pw) + pcx;
        lm_y = fix16_mul(fix16_mul(prior->variance[1], lm_y), ph) + pcy;
        face.points[l][0] = clamp(lm_x,0,fix16_from_int(image_width));
        face.points[l][1] = clamp(lm_y,0,fix16_from_int(image_height));

    }

    fix16_t xmin =
        box_output[r * grid_h * grid_w * 4 + grid_h * grid_w * 0 + grid_h * y + x];
    fix16_t ymin =
        box_output[r * grid_h * grid_w * 4 + grid_h * grid_w * 1 + grid_h * y + x];
    fix16_t xmax =
        box_output[r * grid_h * grid_w * 4 + grid_h * grid_w * 2 + grid_h * y + x];
    fix16_t ymax =
        box_output[r * grid_h * grid_w * 4 + grid_h * grid_w * 3 + grid_h * y + x];

    fix16_t cx =
        fix16_mul(fix16_mul(prior->variance[0],xmin), pw) + pcx;
    fix16_t cy =
        fix16_mul(fix16_mul(prior->variance[1],ymin), ph) + pcy;
    fix16_t w =
        fix16_mul(fix16_exp(fix16_mul(prior->variance[2],xmax)), pw);
    fix16_t h =
        fix16_mul(fix16_exp(fix16_mul(prior->variance[3],ymax)), ph);

    face.box[0] = clamp(cx - w / 2,0,fix16_from_int(image_width));
    face.box[1] = clamp(cy - h / 2,0,fix16_from_int(image_height));
    face.box[2] = clamp(cx + w / 2,0,fix16_from_int(image_width));
    face.box[3] = clamp(cy + h / 2,0,fix16_from_int(image_height));

    return face;
}

static int fix16_face_iou(face_t box_1, face_t box_2, fix16_t thresh)
{
    //return true if the IOU score of box_1 and box2 > threshold

    int width_of_overlap_area = ((box_1.box[2] < box_2.box[2]) ? box_1.box[2] : box_2.box[2]) - ((box_1.box[0] > box_2.box[0]) ? box_1.box[0] : box_2.box[0]);
    int height_of_overlap_area = ((box_1.box[3] < box_2.box[3]) ? box_1.box[3] : box_2.box[3]) - ((box_1.box[1] > box_2.box[1]) ? box_1.box[1] : box_2.box[1]);

    int area_of_overlap;
    if (width_of_overlap_area < 0 || height_of_overlap_area < 0) {
        return 0;
    } else {
        area_of_overlap = fix16_mul(width_of_overlap_area , height_of_overlap_area);
    }

    int box_1_area = fix16_mul(box_1.box[3] - box_1.box[1],box_1.box[2] - box_1.box[0]);
    int box_2_area = fix16_mul(box_2.box[3] - box_2.box[1],box_2.box[2] - box_2.box[0]);
    int area_of_union = box_1_area + box_2_area - area_of_overlap;
    if (area_of_union == 0) return 0;
    float iou= fix16_to_float(area_of_overlap)/fix16_to_float(area_of_union);
    return area_of_overlap > fix16_mul(thresh, area_of_union);
}

#define swap(a, b)                              \
    do {                                        \
        typeof(a) tmp = a;                      \
        a = b;                                \
        b = tmp;                                  \
    } while (0)

static void *insert_into_sorted_array(face_t *faces, int *face_count,
                                          int max_faces, face_t insert) {

    for (int b = 0; b < *face_count; b++) {
        if (faces[b].detectScore < insert.detectScore) {
            swap(faces[b], insert);
        }
    }
    if (*face_count < max_faces) {
        faces[*face_count] = insert;
        *face_count += 1;
    }
}
static inline fix16_t fix16_logistic(fix16_t x) {
    return fix16_div(fix16_one, fix16_add(fix16_one, fix16_exp(-x)));
} // 1 div, 1 exp
static inline fix16_t fix16_logistic_inverse(fix16_t x) {
    return fix16_log(fix16_div(x, fix16_one - x));
}
static inline fix16_t fix16_softmax_retinaface(fix16_t a,fix16_t b){
    //only do the actual softmax if b>a
    if(b<=a){
        return 0;
    }
    fix16_t exp_a = fix16_exp(a);
    fix16_t exp_b = fix16_exp(b);
    return fix16_div(exp_b,fix16_add(exp_a,exp_b));
}
static int get_faces_above_confidence(face_t *faces, int max_faces,
                                      int current_box_count,
                                      fix16_t confidence_threshold,
                                      fix16_t *class_output,
                                      fix16_t *box_output,
                                      fix16_t *landmark_output,
                                      //fix16_t *landmark_output,
                                      int num_classes,
                                      prior_t *prior,
				      int image_width,
				      int image_height) {
    int grid_h = prior->shape[2] * image_height/320;
    int grid_w = prior->shape[3] * image_width/320;
    int repeats = prior->shape[1] / 4;
    int face_count = current_box_count;
    for (int y = 0; y < grid_h; ++y) {
        for (int x = 0; x < grid_w; ++x) {
            for (int r = 0; r < repeats; r++) {
                int idx_none =
                    r * num_classes * grid_h * grid_w + 0 * grid_h * grid_w + y * grid_h + x;
                int idx_face =
                    r * num_classes * grid_h * grid_w + 1 * grid_h * grid_w + y * grid_h + x;
                fix16_t class_confidence = fix16_softmax_retinaface(class_output[idx_none],
                                                                    class_output[idx_face]);

                if (class_confidence >= confidence_threshold ) {

                    face_t face = get_face(box_output,landmark_output,x,y,r,prior, image_width, image_height);
                    face.detectScore = class_confidence;
                    insert_into_sorted_array(faces, &face_count, max_faces,face);
                }
            }
        }
    }
    return face_count;
}
static fix16_box face_to_box(face_t face){
    fix16_box  box;
    box.xmin = face.box[0];
    box.ymin = face.box[1];
    box.xmax = face.box[2];
    box.ymax = face.box[3];
    return box;
}

int post_process_retinaface(face_t faces[],int max_faces, fix16_t *network_outputs[9],
		int image_width, int image_height,
		fix16_t confidence_threshold, fix16_t nms_threshold){

    fix16_t logit_conf = (confidence_threshold);
    int num_classes=2;
    int face_count = 0;
    for (int o = 0; o < 3; ++o) {
        fix16_t *box_outputs = network_outputs[o];
        fix16_t *class_outputs = network_outputs[o+3];
        fix16_t *ldmk_outputs = network_outputs[o+6];
        face_count = get_faces_above_confidence(
            faces, max_faces, face_count, logit_conf, class_outputs, box_outputs,ldmk_outputs,
            num_classes, priors + o, image_width, image_height);
    }
    // faces are at this point sorted by confidence, do nms
    for (int b0 = 0; b0 < face_count; b0++) {
        for (int b1 = 0; b1 < b0; b1++) {
            if (faces[b0].detectScore == 0) {
                continue;
            }

            if (fix16_face_iou(faces[b0], faces[b1],nms_threshold)) {
                // discard b0 by setting confidence class_id to -1;
                faces[b0].detectScore= 0;
                break;
            }
        }
    }
    int discard_count = 0;
    for (int b = 0; b < face_count; b++) {
        if (faces[b].detectScore == 0) { // discarded
            discard_count++;
            continue;
        }
        faces[b - discard_count] = faces[b];
    }
    face_count -= discard_count;
    return face_count;
}
