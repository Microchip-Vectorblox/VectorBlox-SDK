#include "warp.h"
#include "../imageScaler/scaler.h"
#include "fixmatrix.h"
#include <stdio.h>

void warp_affine_image(volatile uint32_t* hls_warp_base_addr,
                       uint8_t* image_in, uint8_t* image_out,int out_w,int out_h,
                       const uint32_t* transform_matrix_2x3) {
        volatile warp_registers_t* control = (volatile warp_registers_t*)hls_warp_base_addr;
        control->in_addr = (uint32_t)(uintptr_t) image_in;
        control->out_addr = (uint32_t)(uintptr_t) image_out;
        control->out_width = (uint32_t)(uintptr_t) out_w;
        control->out_height = (uint32_t)(uintptr_t) out_h;
        for(int i = 0; i < 2*3; ++i) {
                control->m[i] = transform_matrix_2x3[i];
        }
        control->start = 1;
        while(control->start == 0);
}


/* src and dst are fix16_t arrays of 3 x,y points [x0, y0, x1, y1, x2, y2] */
/* M is 2x3 mf16 matrix */
/* taken from https://github.com/opencv/opencv/blob/68d15fc62edad980f1ffa15ee478438335f39cc3/modules/imgproc/src/imgwarp.cpp#
L3325 */
static void getAffineTransform(mf16* M, const fix16_t *src, const fix16_t* dst) {
    mf16 A = {6, 6, 0};
    mf16 B = {6, 1, 0};
    mf16 X = {6, 1, 0};

    // set 1s
    A.data[0][2] = fix16_from_int(1);
    A.data[1][5] = fix16_from_int(1);
    A.data[2][2] = fix16_from_int(1);
    A.data[3][5] = fix16_from_int(1);
    A.data[4][2] = fix16_from_int(1);
    A.data[5][5] = fix16_from_int(1);

    // set src.x
    A.data[0][0] = src[0*2];
    A.data[1][3] = src[0*2];
    A.data[2][0] = src[1*2];
    A.data[3][3] = src[1*2];
    A.data[4][0] = src[2*2];
    A.data[5][3] = src[2*2];

    // set src.y
    A.data[0][1] = src[0*2+1];
    A.data[1][4] = src[0*2+1];
    A.data[2][1] = src[1*2+1];
    A.data[3][4] = src[1*2+1];
    A.data[4][1] = src[2*2+1];
    A.data[5][4] = src[2*2+1];

    // set dst
    B.data[0][0] = dst[0*2];
    B.data[1][0] = dst[0*2+1];
    B.data[2][0] = dst[1*2];
    B.data[3][0] = dst[1*2+1];
    B.data[4][0] = dst[2*2];
    B.data[5][0] = dst[2*2+1];

    mf16 Q, R;
    mf16_qr_decomposition(&Q, &R, &A, 1);
    mf16_solve(&X, &Q, &R, &B);

    M->data[0][0] = X.data[0][0];
    M->data[0][1] = X.data[1][0];
    M->data[0][2] = X.data[2][0];
    M->data[1][0] = X.data[3][0];
    M->data[1][1] = X.data[4][0];
    M->data[1][2] = X.data[5][0];

    return;

}

static void calculate_input_point(const mf16* M,const fix16_t out_x,const fix16_t out_y,fix16_t *in_x,fix16_t *in_y) {
        *in_x = fix16_mul(M->data[0][0],out_x) + fix16_mul(M->data[0][1],out_y) + M->data[0][2];
        *in_y = fix16_mul(M->data[1][0],out_x) + fix16_mul(M->data[1][1],out_y) + M->data[1][2];
}

void warp_image_with_points(volatile uint32_t* scale_base_addr,
                            volatile uint32_t* warp_base_addr,
                            uint32_t* image_in, uint8_t* image_out,uint8_t* temp_buffer,
                            fix16_t src_points[6],
                            fix16_t dest_points[6],
                            unsigned src_width,unsigned src_height,unsigned src_stride,
                            unsigned dst_width,unsigned dst_height
                            ) {

        const int PRESCALE_OUT_HEIGHT = 224;
        const int PRESCALE_OUT_WIDTH = 224;
        /* const int PRESCALE_OUT_HEIGHT_FIX =fix16_from_int(224); */
        /* const int PRESCALE_OUT_WIDTH_FIX = fix16_from_int(224); */
        //Get the Affine Transform matrix
        mf16 MM;
        getAffineTransform(&MM,dest_points,src_points);

        //Use the Affine Transform matrix on the four corners of the image to calculate
        //the bounding box of the crop that is done beforethe warp
        fix16_t x1,y1,x2,y2;
        fix16_t points_to_check[4][2] = { {0,0},
                                    {0,fix16_from_int(dst_height)},
                                    {fix16_from_int(dst_width),0},
                                    {fix16_from_int(dst_width),fix16_from_int(dst_height)}};
        for(int p = 0; p < 4; p++){
                fix16_t in_x,in_y;
                calculate_input_point(&MM,points_to_check[p][0],points_to_check[p][1],&in_x,&in_y);
                if((x1 > in_x) || (p == 0)) x1 = in_x;
                if((y1 > in_y) || (p == 0)) y1 = in_y;
                if((x2 < in_x) || (p == 0)) x2 = in_x;
                if((y2 < in_y) || (p == 0)) y2 = in_y;
        }
    //force min point >= 0
    if(x1<0)x1=0;
    if(y1<0)y1=0;

        int scale_in_width= fix16_to_int(x2-x1);
        int scale_in_height=fix16_to_int(y2-y1);
        if((scale_in_width == 0) || (scale_in_height == 0)) {
                printf("Something odd happened. The input to the image scaler was zero sized %dx%d\n",scale_in_width,scale_in_height);
                printf("input points = %x,%x  %x,%x  %x,%x\n",
                                src_points[0],src_points[1],src_points[2],src_points[3],src_points[4],src_points[5]);
                return;
        }
		int min =0;
		//Scaling up of detected objects that are smaller than prescale warp dimensions.
		if ((scale_in_width>scale_in_height) && ((scale_in_width <= PRESCALE_OUT_WIDTH) && (scale_in_height <= PRESCALE_OUT_HEIGHT))) {
        		 if(PRESCALE_OUT_WIDTH - scale_in_width>0)
					 min = PRESCALE_OUT_WIDTH - scale_in_width;
        		 scale_in_width = scale_in_width + min;
        		 scale_in_height = scale_in_height + min;
		}
        //Remove the translation from the Affine Transformation
        //TODO: Should be able to do that by matrix multiplication of a translation matrix
        //But just rerunning getAffineTransform() because it is easier.

        fix16_t new_src_points[6];
        for(int i = 0; i < 3; ++i) {
                int new_x = fix16_to_int(src_points[2*i] - x1 - min)*PRESCALE_OUT_WIDTH/scale_in_width;
                int new_y = fix16_to_int(src_points[2*i+1] - y1 - min)*PRESCALE_OUT_HEIGHT/scale_in_height;
                new_src_points[2*i] =   fix16_from_int(new_x);
                new_src_points[2*i+1] = fix16_from_int(new_y);
        }

        getAffineTransform(&MM,dest_points,new_src_points);

	resize_image_hls_wait(scale_base_addr);
        resize_image_hls(scale_base_addr,
                                        image_in,
                                        scale_in_width,scale_in_height,src_stride,fix16_to_int(x1),fix16_to_int(y1),
                                        temp_buffer,PRESCALE_OUT_WIDTH,PRESCALE_OUT_HEIGHT);

        for(int c = 0; c < 3; c++) {
                const uint32_t transform_2x3[6] =
                        {MM.data[0][0], MM.data[0][1], MM.data[0][2],
                        MM.data[1][0], MM.data[1][1], MM.data[1][2]};

                warp_affine_image(warp_base_addr,
                      temp_buffer+c*PRESCALE_OUT_HEIGHT*PRESCALE_OUT_WIDTH,
                      image_out+c*dst_width*dst_height,
                      dst_width,dst_height,transform_2x3);
        }
 }
