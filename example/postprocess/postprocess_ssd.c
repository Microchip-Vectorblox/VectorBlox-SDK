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

char* vehicle_classes[] = {
	"unlabeled",
	"car",
	"plate"};


typedef struct {
    int shape[4];
    fix16_t height[6];
    fix16_t width[6];
    fix16_t offset;
    fix16_t variance[4];
    int img_size;
} prior_t;

static prior_t priors[] = {
    {
        .shape = {1, 12, 19, 19},
        .height = {F16(30.0), F16(42.42640750334631),
                   F16(84.85281500669262)},
        .width = {F16(30.0), F16(84.85281500669265),
                  F16(42.426407503346326)},
        .offset = F16(0.5),
        .variance = {F16(0.1), F16(0.1), F16(0.2),
                     F16(0.2)},
	.img_size = 300,
    },
    {
        .shape = {1, 24, 10, 10},
        .height = {F16(104.99999999994), F16(74.24621202454506),
                   F16(148.49242404909012),
                   F16(60.62177826487607),
                   F16(181.87443025249192),
                   F16(125.49900360603824)},
        .width = {F16(104.99999999994), F16(148.49242404909015),
                  F16(74.24621202454507), F16(181.8653347946282),
                  F16(60.61874659720808),
                  F16(125.49900360603824)},
        .offset = F16(0.5),
        .variance = {F16(0.1), F16(0.1), F16(0.2),
                     F16(0.2)},
	.img_size = 300,
    },
    {
        .shape = {1, 24, 5, 5},
        .height = {F16(149.99999910588), F16(106.06601654574379),
                   F16(212.13203309148759),
                   F16(86.60253986222344), F16(259.8206130978267),
                   F16(171.02631247097506)},
        .width = {F16(149.99999910588), F16(212.1320330914876),
                  F16(106.0660165457438), F16(259.8076195866703),
                  F16(86.59820890843783),
                  F16(171.02631247097506)},
        .offset = F16(0.5),
        .variance = {F16(0.1), F16(0.1), F16(0.2),
                     F16(0.2)},
	.img_size = 300,
    },
    {
        .shape = {1, 24, 3, 3},
        .height = {F16(194.99999821182), F16(137.88582106694255),
                   F16(275.7716421338851),
                   F16(112.58330145957083),
                   F16(337.7667959431616),
                   F16(216.3330743270663)},
        .width = {F16(194.99999821182), F16(275.77164213388517),
                  F16(137.88582106694258), F16(337.7499043787124),
                  F16(112.57767121966761),
                  F16(216.3330743270663)},
        .offset = F16(0.5),
        .variance = {F16(0.1), F16(0.1), F16(0.2),
                     F16(0.2)},
	.img_size = 300,
    },
    {
        .shape = {1, 24, 2, 2},
        .height = {F16(239.99999731775998),
                   F16(169.7056255881413), F16(339.4112511762826),
                   F16(138.5640630569182),
                   F16(415.71297878849646),
                   F16(261.5339335100698)},
        .width =
        {F16(239.99999731775998), F16(339.41125117628263),
         F16(169.70562558814132), F16(415.69218917075455),
         F16(138.55713353089737), F16(261.5339335100698)},
        .offset = F16(0.5),
        .variance = {F16(0.1), F16(0.1), F16(0.2),
                     F16(0.2)},
	.img_size = 300,
    },
    {
        .shape = {1, 24, 1, 1},
        .height = {F16(284.9999964237), F16(403.05086021868016),
                   F16(201.52543010934008),
                   F16(493.6344739627967),
                   F16(164.53659584212716),
                   F16(292.40382850966574)},
        .width = {F16(284.9999964237), F16(201.52543010934002),
                  F16(403.05086021868004), F16(164.5448246542656),
                  F16(493.6591616338313),
                  F16(292.40382850966574)},
        .offset = F16(0.5),
        .variance = {F16(0.1), F16(0.1), F16(0.2),
                     F16(0.2)},
	.img_size = 300,
    }};

static prior_t torch_priors[] = {
    {
        .shape = {1, 24, 20, 20},
        .height = {F16(64.0), F16(84.66403), F16(45.25482), F16(90.50964), F16(36.95041), F16(110.85123)},
        .width = {F16(64.0), F16(84.66403), F16(90.50964), F16(45.25482), F16(110.85123), F16(36.95041)},
        .offset = F16(0.5),
        .variance = {F16(0.1), F16(0.1), F16(0.2), F16(0.2)},
	.img_size = 320,
    },
    {
        .shape = {1, 24, 10, 10},
        .height = {F16(112.0), F16(133.86562), F16(79.195984), F16(158.3919), F16(64.66324), F16(193.98969)},
        .width = {F16(112.0), F16(133.86562), F16(158.3919), F16(79.195984), F16(193.98969), F16(64.66324)},
        .offset = F16(0.5),
        .variance = {F16(0.1), F16(0.1), F16(0.2), F16(0.2)},
	.img_size = 320,
    },
    {
        .shape = {1, 24, 5, 5},

        .height = {F16(160.0), F16(182.42805), F16(113.13707), F16(226.27419), F16(92.37601), F16(277.1281)},
        .width = {F16(160.0), F16(182.42805), F16(226.27419), F16(113.13707), F16(277.1281), F16(92.37601)},
        .offset = F16(0.5),
        .variance = {F16(0.1), F16(0.1), F16(0.2), F16(0.2)},
	.img_size = 320,
    },
    {
        .shape = {1, 24, 3, 3},
        .height = {F16(208.0), F16(230.7553), F16(147.07819), F16(294.15643), F16(120.08885), F16(319.99997)},
        .width = {F16(208.0), F16(230.7553), F16(294.15643), F16(147.07819), F16(319.99997), F16(120.08885)},
        .offset = F16(0.5),
        .variance = {F16(0.1), F16(0.1), F16(0.2), F16(0.2)},
	.img_size = 320,
    },
    {
        .shape = {1, 24, 2, 2},
        .height = {F16(256.0), F16(278.96957), F16(181.01935), F16(320.0), F16(147.80165), F16(320.0)},
        .width = {F16(256.0), F16(278.96957), F16(320.0), F16(181.01935), F16(320.0), F16(147.80165)},
        .offset = F16(0.5),
        .variance = {F16(0.1), F16(0.1), F16(0.2), F16(0.2)},
	.img_size = 320,
    },
    {
        .shape = {1, 24, 1, 1},
        .height = {F16(304.0), F16(311.89743), F16(214.96045), F16(320.0), F16(175.5145), F16(320.0)},
        .width = {F16(304.0), F16(311.89743), F16(320.0), F16(214.96045), F16(320.0), F16(175.5145)},
        .offset = F16(0.5),
        .variance = {F16(0.1), F16(0.1), F16(0.2), F16(0.2)},
	.img_size = 320,
    }};

static prior_t vehicle_priors[] = {
	{
		.shape = {1, 8, 16, 16},
		.height = {F16(17.408000946044922), F16(13.312000274658203)},
		.width = {F16(7.680000305175781), F16(24.832000732421875)},
        .offset = F16(0.5),
        .variance = {F16(0.1), F16(0.1), F16(0.2),
                     F16(0.2)},
	.img_size = 256,
	},
	{
		.shape = {1, 12, 8, 8},
		.height = {F16(46.08000183105469), F16(28.15999984741211), F16(110.08000183105469)},
		.width = {F16(22.27199935913086), F16(84.4800033569336), F16(25.599998474121094)},
        .offset = F16(0.5),
        .variance = {F16(0.1), F16(0.1), F16(0.2),
                     F16(0.2)},
	.img_size = 256,
	},
	{
		.shape = {1, 12, 4, 4},
		.height = {F16(66.55999755859375), F16(87.03999328613281), F16(51.19999694824219)},
		.width = {F16(69.1199951171875), F16(102.39999389648438), F16(140.8000030517578)},
        .offset = F16(0.5),
        .variance = {F16(0.1), F16(0.1), F16(0.2),
                     F16(0.2)},
	.img_size = 256,
	},
	{
		.shape = {1, 4, 2, 2},
		.height = {F16(94.72000122070312)},
		.width = {F16(133.1199951171875)},
        .offset = F16(0.5),
        .variance = {F16(0.1), F16(0.1), F16(0.2),
                     F16(0.2)},
	.img_size = 256,
	},
	{
		.shape = {1, 4, 1, 1},
		.height = {F16(122.8800048828125)},
		.width = {F16(115.20001220703125)},
        .offset = F16(0.5),
        .variance = {F16(0.1), F16(0.1), F16(0.2),
                     F16(0.2)},
	.img_size = 256,
	},
	{
		.shape = {1, 12, 1, 1},
		.height = {F16(161.27999877929688), F16(197.1199951171875), F16(243.20001220703125)},
		.width = {F16(163.83999633789062), F16(197.1199951171875), F16(243.20001220703125)},
        .offset = F16(0.5),
        .variance = {F16(0.1), F16(0.1), F16(0.2),
                     F16(0.2)},
	.img_size = 256,
	}};


typedef struct{
    fix16_t cx,cy,w,h;
} prior_box;


static prior_box gen_prior_box(prior_t *prior, int x, int y, int r) {
    fix16_t fix_x = fix16_from_int(x);
    fix16_t fix_y = fix16_from_int(y);
    fix16_t h = fix16_from_int(prior->shape[2]);
    fix16_t w = fix16_from_int(prior->shape[3]);
    int img_h = fix16_from_int(prior->img_size);
    int img_w = fix16_from_int(prior->img_size);

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

static fix16_box get_box_int8(int8_t *box_output, fix16_t scale, int32_t zero, int x, int y, int r, prior_t *prior) {
    int grid = prior->shape[2];
    prior_box pbox = gen_prior_box(prior, x, y, r);

    fix16_box base_box;
    base_box.xmin =
	int8_to_fix16_single(box_output[r * grid * grid * 4 + grid * grid * 0 + grid * y + x],
		scale, zero);
    base_box.ymin =
        int8_to_fix16_single(box_output[r * grid * grid * 4 + grid * grid * 1 + grid * y + x],
		scale, zero);
    base_box.xmax =
        int8_to_fix16_single(box_output[r * grid * grid * 4 + grid * grid * 2 + grid * y + x],
		scale, zero);
    base_box.ymax =
        int8_to_fix16_single(box_output[r * grid * grid * 4 + grid * grid * 3 + grid * y + x],
		scale, zero);
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


    base_box.xmin = clamp(fix16_to_int( (cx - w / 2)), 0, prior->img_size);
    base_box.ymin = clamp(fix16_to_int( (cy - h / 2)), 0, prior->img_size);
    base_box.xmax = clamp(fix16_to_int( (cx + w / 2)), 0, prior->img_size);
    base_box.ymax = clamp(fix16_to_int( (cy + h / 2)), 0, prior->img_size);


    return base_box;
}

static fix16_box get_box(fix16_t *box_output, int x, int y, int r, prior_t *prior) {
    int grid = prior->shape[2];
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


    base_box.xmin = clamp(fix16_to_int( (cx - w / 2)), 0, prior->img_size);
    base_box.ymin = clamp(fix16_to_int( (cy - h / 2)), 0, prior->img_size);
    base_box.xmax = clamp(fix16_to_int( (cx + w / 2)), 0, prior->img_size);
    base_box.ymax = clamp(fix16_to_int( (cy + h / 2)), 0, prior->img_size);


    return base_box;
}

extern int fix16_box_iou(fix16_box box_1, fix16_box box_2, fix16_t thresh);
#define swap(a, b)                              \
    do {                                        \
        typeof(a) tmp = a;                      \
        a = b;                                \
        b = tmp;                                  \
    } while (0)

void insert_into_sorted_array(fix16_box *boxes, int *box_count,
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

static int get_boxes_above_confidence_torch_int8(fix16_box *boxes, int max_boxes,
                                      int current_box_count,
                                      fix16_t confidence_threshold,
                                      int8_t *class_output, fix16_t class_scale, int32_t class_zero,
                                      int8_t *box_output, fix16_t box_scale, int32_t box_zero,
				      int repeats, int num_classes,
                                      prior_t *prior) {
    int grid = prior->shape[2];
    int box_count = current_box_count;
    int repeated = prior->shape[1] / repeats;
    fix16_t class_scores[num_classes];
    for (int y = 0; y < grid; ++y) {
	    for (int x = 0; x < grid; ++x) {
		    for (int r = 0; r < repeated; r++) {
			    int class_offset = r * num_classes * grid * grid + y * grid + x;
			    int max_score = class_output[class_offset];
			    int max_class = 0;
			    for (int c = 1; c < num_classes; c++) {
				    int8_t score = class_output[class_offset+c*grid*grid];
				    if(max_score < score) {
					    max_score = score;
					    max_class = c;
				    }
			    }
			    if (max_class > 0) {
				    for (int c = 0; c < num_classes; c++) {
					    class_scores[c] = int8_to_fix16_single(class_output[class_offset+c*grid*grid], class_scale, class_zero);
				    }
				    fix16_softmax(class_scores, num_classes, class_scores);
				    fix16_t class_confidence = class_scores[max_class];
				    if (class_confidence > confidence_threshold) {
					    fix16_box box = get_box_int8(box_output, box_scale, box_zero, x, y, r, prior);
					    box.class_id = max_class;
					    box.confidence = class_confidence;
#if 0
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

static int get_boxes_above_confidence_torch(fix16_box *boxes, int max_boxes,
                                      int current_box_count,
                                      fix16_t confidence_threshold,
                                      fix16_t *class_output,
                                      fix16_t *box_output, int repeats, int num_classes,
                                      prior_t *prior) {
    int grid = prior->shape[2];
    int box_count = current_box_count;
    int repeated = prior->shape[1] / repeats;
    fix16_t class_scores[num_classes];
    for (int y = 0; y < grid; ++y) {
	    for (int x = 0; x < grid; ++x) {
		    for (int r = 0; r < repeated; r++) {
			    int class_offset = r * num_classes * grid * grid + y * grid + x;
			    int max_score = class_output[class_offset];
			    int max_class = 0;
			    class_scores[0] =  max_score;
			    for (int c = 1; c < num_classes; c++) {
				    class_scores[c] = class_output[class_offset+c*grid*grid];
				    if(max_score < class_scores[c]) {
					    max_score = class_scores[c];
					    max_class = c;
				    }
			    }
			    if (max_class > 0) {
				    fix16_softmax(class_scores, num_classes, class_scores);
				    fix16_t class_confidence = class_scores[max_class];
				    if (class_confidence > confidence_threshold) {
					    fix16_box box = get_box(box_output, x, y, r, prior);
					    box.class_id = max_class;
					    box.confidence = class_confidence;
#if 0
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

static int get_boxes_above_confidence(fix16_box *boxes, int max_boxes,
                                      int current_box_count,
                                      fix16_t confidence_threshold,
                                      fix16_t *class_output,
                                      fix16_t *box_output, int repeats, int num_classes,
                                      prior_t *prior) {
    int grid = prior->shape[2];
    int box_count = current_box_count;
    for (int r = 0; r < prior->shape[1] / repeats; r++) {
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
#if 0
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

int post_process_ssd(fix16_box *boxes, const int max_boxes,
                       fix16_t **network_outputs, const int num_outputs, const int repeats, const int num_classes,
                       const fix16_t confidence_threshold, const fix16_t nms_threshold, prior_t *priors) {
    fix16_t logit_conf = fix16_logistic_inverse(confidence_threshold);
    int box_count = 0;
    for (int o = 0; o < num_outputs/2; ++o) {
        fix16_t *class_outputs = network_outputs[2 * o + 1];
        fix16_t *box_outputs = network_outputs[2 * o];
        box_count = get_boxes_above_confidence(
            boxes, max_boxes, box_count, logit_conf, class_outputs, box_outputs,
            repeats, num_classes, priors + o);
    }

    // boxes are at this point sorted by confidence, do nms
    for (int b0 = 0; b0 < box_count; b0++) {
        for (int b1 = 0; b1 < b0; b1++) {
            if (boxes[b0].class_id != boxes[b1].class_id) {
                continue;
            }
            if (fix16_box_iou(boxes[b0], boxes[b1], nms_threshold)) {
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

int post_process_ssdv2(fix16_box *boxes, int max_boxes, 
                       fix16_t *network_outputs[12], int num_classes,
                       fix16_t confidence_threshold, fix16_t nms_threshold) 
{
	return post_process_ssd(boxes, max_boxes, network_outputs, 12, 4, num_classes,
			confidence_threshold, nms_threshold, priors);
}

int post_process_ssd_torch(fix16_box *boxes, int max_boxes, 
                       fix16_t *network_outputs[12], int num_classes,
                       fix16_t confidence_threshold, fix16_t nms_threshold) 
{
    int box_count = 0;
    for (int o = 0; o < 6; ++o) {
	    fix16_t *class_outputs = network_outputs[2 * o + 1];
	    fix16_t *box_outputs = network_outputs[2 * o];

	    box_count = get_boxes_above_confidence_torch(
			    boxes, max_boxes, box_count, confidence_threshold, class_outputs, box_outputs,
			    4, num_classes, torch_priors + o);
    }

    fix16_sort_boxes(boxes, NULL, box_count);
    fix16_do_nms(boxes, box_count, nms_threshold);
    int clean_box_count = fix16_clean_boxes(boxes, NULL, box_count, 320, 320);

    return clean_box_count;
}

int post_process_ssd_torch_int8(fix16_box *boxes, int max_boxes, 
                       int8_t *network_outputs[12],
                       fix16_t network_scales[12],
                       int32_t network_zeros[12],
		       int num_classes,
                       fix16_t confidence_threshold, fix16_t nms_threshold)
{
    int box_count = 0;
    for (int o = 0; o < 6; ++o) {
	    int8_t *class_outputs = network_outputs[2 * o + 1];
	    int8_t *box_outputs = network_outputs[2 * o];
	    fix16_t class_scale = network_scales[2 * o + 1];
	    fix16_t box_scale = network_scales[2 * o];
	    int32_t class_zero = network_zeros[2 * o + 1];
	    int32_t box_zero = network_zeros[2 * o];

	    box_count = get_boxes_above_confidence_torch_int8(
			    boxes, max_boxes, box_count, confidence_threshold,
			    class_outputs, class_scale, class_zero,
			    box_outputs, box_scale, box_zero,
			    4, num_classes, torch_priors + o);
    }

    fix16_sort_boxes(boxes, NULL, box_count);
    fix16_do_nms(boxes, box_count, nms_threshold);
    int clean_box_count = fix16_clean_boxes(boxes, NULL, box_count, 320, 320);

    return clean_box_count;
}

int post_process_vehicles(fix16_box *boxes, int max_boxes,
                       fix16_t *network_outputs[12], int num_classes,
                       fix16_t confidence_threshold, fix16_t nms_threshold) 
{
	return post_process_ssd(boxes, max_boxes, network_outputs, 12, 4, num_classes,
			confidence_threshold, nms_threshold, vehicle_priors);
}

