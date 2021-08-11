#include "postprocess.h"
#include <stdio.h>
char* coco91_classes[] = {
    "unlabeled",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "street sign",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "hat",
    "backpack",
    "umbrella",
    "shoe",
    "eye glasses",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "plate",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "mirror",
    "dining table",
    "window",
    "desk",
    "toilet",
    "door",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "blender",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
    "hair brush"};


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
        .shape = {1, 12, 19, 19},
        .height = {fix16_conv(30.0), fix16_conv(42.42640750334631),
                   fix16_conv(84.85281500669262)},
        .width = {fix16_conv(30.0), fix16_conv(84.85281500669265),
                  fix16_conv(42.426407503346326)},
        .offset = fix16_conv(0.5),
        .variance = {fix16_conv(0.1), fix16_conv(0.1), fix16_conv(0.2),
                     fix16_conv(0.2)},
    },
    {
        .shape = {1, 24, 10, 10},
        .height = {fix16_conv(104.99999999994), fix16_conv(74.24621202454506),
                   fix16_conv(148.49242404909012),
                   fix16_conv(60.62177826487607),
                   fix16_conv(181.87443025249192),
                   fix16_conv(125.49900360603824)},
        .width = {fix16_conv(104.99999999994), fix16_conv(148.49242404909015),
                  fix16_conv(74.24621202454507), fix16_conv(181.8653347946282),
                  fix16_conv(60.61874659720808),
                  fix16_conv(125.49900360603824)},
        .offset = fix16_conv(0.5),
        .variance = {fix16_conv(0.1), fix16_conv(0.1), fix16_conv(0.2),
                     fix16_conv(0.2)},
    },
    {
        .shape = {1, 24, 5, 5},
        .height = {fix16_conv(149.99999910588), fix16_conv(106.06601654574379),
                   fix16_conv(212.13203309148759),
                   fix16_conv(86.60253986222344), fix16_conv(259.8206130978267),
                   fix16_conv(171.02631247097506)},
        .width = {fix16_conv(149.99999910588), fix16_conv(212.1320330914876),
                  fix16_conv(106.0660165457438), fix16_conv(259.8076195866703),
                  fix16_conv(86.59820890843783),
                  fix16_conv(171.02631247097506)},
        .offset = fix16_conv(0.5),
        .variance = {fix16_conv(0.1), fix16_conv(0.1), fix16_conv(0.2),
                     fix16_conv(0.2)},
    },
    {
        .shape = {1, 24, 3, 3},
        .height = {fix16_conv(194.99999821182), fix16_conv(137.88582106694255),
                   fix16_conv(275.7716421338851),
                   fix16_conv(112.58330145957083),
                   fix16_conv(337.7667959431616),
                   fix16_conv(216.3330743270663)},
        .width = {fix16_conv(194.99999821182), fix16_conv(275.77164213388517),
                  fix16_conv(137.88582106694258), fix16_conv(337.7499043787124),
                  fix16_conv(112.57767121966761),
                  fix16_conv(216.3330743270663)},
        .offset = fix16_conv(0.5),
        .variance = {fix16_conv(0.1), fix16_conv(0.1), fix16_conv(0.2),
                     fix16_conv(0.2)},
    },
    {
        .shape = {1, 24, 2, 2},
        .height = {fix16_conv(239.99999731775998),
                   fix16_conv(169.7056255881413), fix16_conv(339.4112511762826),
                   fix16_conv(138.5640630569182),
                   fix16_conv(415.71297878849646),
                   fix16_conv(261.5339335100698)},
        .width =
        {fix16_conv(239.99999731775998), fix16_conv(339.41125117628263),
         fix16_conv(169.70562558814132), fix16_conv(415.69218917075455),
         fix16_conv(138.55713353089737), fix16_conv(261.5339335100698)},
        .offset = fix16_conv(0.5),
        .variance = {fix16_conv(0.1), fix16_conv(0.1), fix16_conv(0.2),
                     fix16_conv(0.2)},
    },
    {
        .shape = {1, 24, 1, 1},
        .height = {fix16_conv(284.9999964237), fix16_conv(403.05086021868016),
                   fix16_conv(201.52543010934008),
                   fix16_conv(493.6344739627967),
                   fix16_conv(164.53659584212716),
                   fix16_conv(292.40382850966574)},
        .width = {fix16_conv(284.9999964237), fix16_conv(201.52543010934002),
                  fix16_conv(403.05086021868004), fix16_conv(164.5448246542656),
                  fix16_conv(493.6591616338313),
                  fix16_conv(292.40382850966574)},
        .offset = fix16_conv(0.5),
        .variance = {fix16_conv(0.1), fix16_conv(0.1), fix16_conv(0.2),
                     fix16_conv(0.2)},
    }};

static const int IMG_SIZE = 300;
typedef struct{
    fix16_t cx,cy,w,h;
} prior_box;
static prior_box gen_prior_box(prior_t *prior, int x, int y, int r) {
    fix16_t fix_x = fix16_from_int(x);
    fix16_t fix_y = fix16_from_int(y);
    fix16_t fix_r = fix16_from_int(r);
    fix16_t h = fix16_from_int(prior->shape[2]);
    fix16_t w = fix16_from_int(prior->shape[3]);
    int img_h = fix16_from_int(IMG_SIZE);
    int img_w = fix16_from_int(IMG_SIZE);

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
static fix16_box get_box(fix16_t *box_output, int x, int y, int r,
                         prior_t *prior) {
    int grid = prior->shape[2];
    int repeats = prior->shape[1] / 4;
    prior_box pbox = gen_prior_box(prior, x, y, r);

    fix16_box base_box;
    base_box.xmin =
        box_output[r * grid * grid * 4 + grid * grid * 0 + grid * y + x];
    base_box.ymin =
        box_output[r * grid * grid * 4 + grid * grid * 1 + grid * y + x];
    base_box.xmax =
        box_output[r * grid * grid * 4 + grid * grid * 2 + grid * y + x];
    base_box.ymax =
        box_output[r * grid * grid * 4 + grid * grid * 3 + grid * y + x];
    fix16_t pw = pbox.w;
    fix16_t ph = pbox.h;

    fix16_t pcx = pbox.cx;
    fix16_t pcy = pbox.cy;
    fix16_t cx =
        fix16_mul(fix16_mul(prior->variance[0], base_box.xmin), pw) + pcx;
    fix16_t cy =
        fix16_mul(fix16_mul(prior->variance[1], base_box.ymin), ph) + pcy;
    fix16_t w =
        fix16_mul(fix16_exp(fix16_mul(prior->variance[2], base_box.xmax)), pw);
    fix16_t h =
        fix16_mul(fix16_exp(fix16_mul(prior->variance[3], base_box.ymax)), ph);


    base_box.xmin = clamp(fix16_to_int( (cx - w / 2)),0,IMG_SIZE);
    base_box.ymin = clamp(fix16_to_int( (cy - h / 2)),0,IMG_SIZE);
    base_box.xmax = clamp(fix16_to_int( (cx + w / 2)),0,IMG_SIZE);
    base_box.ymax = clamp(fix16_to_int( (cy + h / 2)),0,IMG_SIZE);


    return base_box;
}
extern int fix16_box_iou(fix16_box box_1, fix16_box box_2, fix16_t thresh);
#define swap(a, b)                              \
    do {                                        \
        typeof(a) tmp = a;                      \
        a = b;                                \
        b = tmp;                                  \
    } while (0)

static fix16_box *insert_into_sorted_array(fix16_box *boxes, int *box_count,
                                          int max_boxes, fix16_box insert) {

    for (int b = 0; b < *box_count; b++) {
        if (boxes[b].confidence < insert.confidence) {
            swap(boxes[b], insert);
        }
    }
    if (*box_count < max_boxes) {
        boxes[*box_count] = insert;
        *box_count += 1;
    }
}
static int get_boxes_above_confidence(fix16_box *boxes, int max_boxes,
                                      int current_box_count,
                                      fix16_t confidence_threshold,
                                      fix16_t *class_output,
                                      fix16_t *box_output, int num_classes,
                                      prior_t *prior) {
    int grid = prior->shape[2];
    int repeats = prior->shape[1] / 4;
    int box_count = current_box_count;
    for (int r = 0; r < repeats; r++) {
        for (int c = 1; c < num_classes; ++c) {
            for (int y = 0; y < grid; ++y) {
                for (int x = 0; x < grid; ++x) {
                    int idx =
                        r * num_classes * grid * grid + c * grid * grid + y * grid + x;
                    fix16_t class_confidence = class_output[idx];
                    if (class_confidence > confidence_threshold) {
                        fix16_box box = get_box(box_output, x, y, r, prior);
                        box.class_id = c;
                        box.confidence = class_confidence;
                        #if 1
                        insert_into_sorted_array(boxes, &box_count, max_boxes, box);
                        #else
                        boxes[box_count++]=box;
                        #endif
                    }
                }
            }
        }
    }
    return box_count;
}

static inline fix16_t fix16_logistic(fix16_t x) {
    return fix16_div(fix16_one, fix16_add(fix16_one, fix16_exp(-x)));
} // 1 div, 1 exp
static inline fix16_t fix16_logistic_inverse(fix16_t x) {
    return fix16_log(fix16_div(x, fix16_one - x));
}
int post_process_ssdv2(fix16_box *boxes, int max_boxes,
                       fix16_t *network_outputs[12], int num_classes,
                       fix16_t confidence_threshold, fix16_t nms_threshold) {
    fix16_t logit_conf = fix16_logistic_inverse(confidence_threshold);
    int box_count = 0;
    for (int o = 0; o < 6; ++o) {
        fix16_t *class_outputs = network_outputs[2 * o + 1];
        fix16_t *box_outputs = network_outputs[2 * o];
        box_count = get_boxes_above_confidence(
            boxes, max_boxes, box_count, logit_conf, class_outputs, box_outputs,
            num_classes, priors + o);
    }
    // boxes are at this point sorted by confidence, do nms
    for (int b0 = 0; b0 < box_count; b0++) {
        for (int b1 = 0; b1 < b0; b1++) {
            if (boxes[b0].class_id != boxes[b1].class_id) {
                continue;
            }
            if (fix16_box_iou(boxes[b0], boxes[b1],nms_threshold)) {
                // discard b0 by setting confidence class_id to -1;
                boxes[b0].class_id= -1;
                break;
            }
        }
    }
    int discard_count = 0;
    for (int b = 0; b < box_count; b++) {
        if (boxes[b].class_id == -1) { // discarded
            discard_count++;
            continue;
        }
        boxes[b].confidence = fix16_logistic(boxes[b].confidence);
        boxes[b - discard_count] = boxes[b];
    }

    return box_count - discard_count;
}
