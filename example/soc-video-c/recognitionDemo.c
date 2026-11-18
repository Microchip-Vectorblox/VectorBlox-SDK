#include "recognitionDemo.h"
#include "imageScaler/scaler.h"
#include "warpAffine/warp.h"
#include "frameDrawing/draw_assist.h"
#include "frameDrawing/draw.h"
#include <string.h>
#include <sys/time.h>

extern volatile uint32_t* PROCESSING_FRAME_ADDRESS;
extern volatile uint32_t* PROCESSING_NEXT_FRAME_ADDRESS;
extern volatile uint32_t* PROCESSING_NEXT2_FRAME_ADDRESS;
extern volatile uint32_t* SCALER_BASE_ADDRESS;
extern volatile uint32_t* WARP_BASE_ADDRESS;
extern volatile uint32_t* RED_DDR_FRAME_START_ADDR;
extern volatile uint32_t* GREEN_DDR_FRAME_START_ADDR;
extern volatile uint32_t* BLUE_DDR_FRAME_START_ADDR;

static inline void* virt_to_phys(vbx_cnn_t* vbx_cnn,void* virt){
  return (char*)(virt) + vbx_cnn->dma_phys_trans_offset;
}

extern int fps;
extern uint32_t* overlay_draw_frame;

// Globals Specification
#if VBX_SOC_DRIVER
	#include "pdma/pdma_helpers.h"
	#define MAX_TRACKS 48
	#define DB_LENGTH 32
	extern int delete_embedding_mode;
	extern int add_embedding_mode;
	extern int capture_embedding;
#if PDMA
	extern int8_t* pdma_mmap_t;
	extern uint64_t pdma_out;
	extern int32_t pdma_channel;
#endif	
	
#else
	#define MAX_TRACKS 20
	#define DB_LENGTH 4
	int delete_embedding_mode=0;
	int add_embedding_mode = 0;
	int capture_embedding = 0;
#endif

uint8_t* warp_temp_buffer = NULL;

// Database Embeddings
#define PRESET_DB_LENGTH 4
#define PIPELINED 1
int id_check = 0;
int db_end_idx = 4;
int face_count = 0;
char *db_nameStr[DB_LENGTH] = {
	"Bob",
	"John",
	"Nancy",
	"Tina",
};

#define EMBEDDING_LENGTH 128

int16_t db_embeddings[DB_LENGTH][EMBEDDING_LENGTH] = {
	{-2109,-9091,8291,-655,2691,7491,-1673,9455,-9091,9309,-1236,-6036,-7636,-7127,9455,-73,-6909,-3273,-9091,-4655,6909,-4727,3636,-5236,4509,5236,-4800,-9091,-5745,-727,-3491,-218,-9091,-5891,-2909,218,1964,9455,3709,-1818,3636,-7200,9018,3273,9455,873,9455,2545,-3273,-1673,-1527,1745,3418,-4436,-9091,-9091,6400,436,6618,-6400,-218,509,-7564,7200,-9091,-800,-5382,1600,5018,-5673,1164,-1018,73,1745,1673,2400,-364,9455,1527,-218,-73,5236,-8073,9455,5673,-2473,-4436,-9091,1600,-9091,2764,-436,-1818,7564,-9091,-9091,9455,-9091,-7200,-2255,-2473,-8073,3564,-1091,-2473,-7200,1527,-4509,3345,3055,-3200,-1309,1673,-8727,9455,-4582,-509,-9091,-5527,8945,3491,6909,9455,-2400,-7709,3927,-5745,-9091,},
	{-5853,-225,8855,4202,8405,9756,-9380,6754,-1426,7279,-1801,-1651,-5553,1426,1126,375,9305,3002,-9380,-8705,-4127,-9380,9756,-9305,8255,-6229,4428,3977,-2251,7429,375,4428,-1501,-7129,3527,-7354,1201,9756,-4052,6304,-4653,-4878,-1126,-1201,2476,-9380,9756,-5103,-7729,901,-9380,5853,-7204,9756,-5103,-5628,-6604,901,150,-8255,-9380,3677,-4878,6003,2476,5253,600,-9305,1576,300,5328,-525,1876,1276,6604,1951,-4503,-1576,-8330,4428,-5853,-375,-5853,6304,300,9756,-901,-4277,-9380,1651,-3227,5928,-9380,2476,5253,-3902,4428,2326,-901,2251,5928,-6754,-9380,300,8480,3677,-9380,-9380,-4803,3977,-1726,9305,150,1051,3452,-1501,-1876,-976,2326,3827,9756,-3077,-9380,-9380,1951,7204,-1426,-4878,},
	{-6988,-5696,1291,-1899,9570,-608,-1367,6000,-9494,3646,-6076,-4405,-8962,8127,-9494,0,6380,5089,-532,-4557,4785,-7443,-2355,-1215,9874,911,4405,-5165,-2203,-2658,-4937,6988,-9494,4329,-2582,9342,-7747,7595,6684,1519,6684,2051,9874,-3418,4481,1823,5241,6456,5165,608,-4253,9874,-1139,835,-9494,5469,1899,-3646,-1899,5317,-2051,-4785,-6380,-2430,2127,987,6608,-5089,9266,-7823,6532,-9494,-1595,4253,9266,-9494,2051,-4937,-3798,2734,1367,-2506,3798,5620,-4785,9874,1215,-8962,-3874,6684,9874,-4253,6836,911,-9494,5772,1899,-2279,-9494,7747,-9494,1595,-9494,4861,4025,-7443,-1671,5620,8810,-5848,-835,-9494,-835,6988,-9494,911,-9494,-1139,-3798,-5620,-1823,-2810,-7140,2886,911,3950,-5317,-1443,},
	{-5027,10714,-7747,165,10714,-8736,-989,330,-3544,2308,-8489,-494,3461,-330,1731,-3791,7088,-1896,-4533,-5192,-2472,4450,-3544,-4780,5275,-3049,-742,-659,-10302,2720,577,8324,-10302,-7500,-10302,-5439,-4533,-1978,-5769,-3956,577,2472,-7994,2390,8241,5769,-6840,7170,-8818,-4615,4862,10714,-8571,3132,-3461,3049,-330,-659,742,-2555,3873,2555,-5275,2308,6923,7994,5357,-10302,-2555,-4780,-2143,7829,-4286,7417,3791,907,-4286,5851,824,-1401,494,-2472,-3626,-3626,-7252,-5192,1566,330,-8241,7088,7170,-8736,4038,9066,-10302,-10302,-6099,9313,1154,5687,6511,-10302,-4780,2143,-1154,-10302,-2555,2472,-5192,-6923,5275,-8489,-7417,5027,3132,3214,-2555,-1813,-10302,-8654,5110,-2967,6346,-2060,-10302,-2967,-7335,-4780,},
};

//#define EMBEDDING_LENGTH 512

//int16_t db_embeddings[DB_LENGTH][EMBEDDING_LENGTH] = {
//	{-768,-81,1778,5495,2101,-1455,202,3435,1051,-4808,-1778,-1818,2142,-1172,606,2909,3031,4283,889,323,323,-1374,-3839,3192,-2626,-1697,-2222,687,-162,1778,-4081,-2909,-2667,-3354,2546,1657,1293,-1131,-283,-1616,-1737,2586,1859,-3637,727,-4364,-283,162,-525,1131,364,3677,-4768,-3839,-3313,-4808,-2303,283,283,2869,525,3596,2020,-1535,1535,-4808,1010,2667,404,-849,5495,4243,-4808,-1414,1293,1172,929,4566,-1495,-687,3313,2950,1616,-4808,2586,-1131,3071,2626,-1697,3596,-3717,5495,-3919,-3515,-1980,-1818,-4808,4202,2707,2182,970,1091,2222,2020,-4808,-2061,3556,808,-2303,-202,5495,2505,-1010,4526,-485,-121,-3071,-3273,3233,-4808,4687,5293,647,-4808,4202,2303,-2546,2344,687,-364,3919,-3435,-2061,2424,2788,2424,4202,-3192,889,-2748,2505,5051,1333,-2667,-3313,-4808,2586,2909,1253,1374,-2626,2020,4121,-3758,3515,-1778,0,5495,3273,-2020,3515,5495,3192,1091,2344,3758,3798,3233,2020,-4485,-3273,-1495,2020,2626,4889,-1616,-404,-3839,3111,-3960,3273,-2748,1616,4526,-970,-4808,3192,-162,-1778,-4808,3677,1899,-849,5253,-4485,-3233,-4324,525,-2788,-1657,3758,1212,1131,2748,-1051,3475,5495,2101,4808,-81,444,-4808,1940,-2303,-929,4970,4647,2424,-3313,4445,-1455,-4041,-1212,-1293,3354,-3313,1495,-3111,364,4566,-1172,-4808,-202,-444,40,-4808,-3556,-1980,-40,2626,-1899,-4162,-2344,4647,4162,3233,1131,566,2828,-2505,-1616,1737,-849,5132,2626,-4808,4970,444,2142,-2142,3556,5495,1980,1091,3556,5212,-162,-1737,-4808,-202,889,-3192,2263,-606,-1576,-889,2950,-4162,4041,364,-1778,5495,5495,-3435,162,-3354,4808,-202,-1172,-4526,3273,2707,-4162,5495,2142,-647,606,5495,-1859,-1253,485,-1535,606,-727,566,1172,5334,-4808,3394,-485,1576,-1940,-1131,-3717,-4808,-2101,-283,-485,849,-2707,2586,-1737,2303,-1616,-4808,-1010,929,-849,-121,-2465,-1616,-1576,3071,3596,-2667,3233,-1414,2142,-566,3394,-364,-606,-1899,1859,-4808,1455,-1657,-4081,-4808,2950,-4566,-121,-2061,-768,-889,-4647,-1535,-1778,2909,-3879,2546,525,606,5495,2586,-3758,4081,566,5495,-1414,2101,4121,3031,-4000,3475,3111,-2101,4485,404,1576,1172,1899,5172,-2546,5495,485,-3273,-3394,162,1212,1414,-162,-485,4647,-4808,1051,-1778,121,5495,364,0,4404,4081,-4364,-3515,4041,5495,-444,5495,2142,849,5374,-162,-1616,606,4121,-2424,-4808,849,-283,4526,202,-3435,-4808,1657,-283,768,-4808,-1818,1657,-323,2263,1737,3556,2101,-364,-4808,-4808,-2142,-929,-4364,1657,1374,-2061,808,2424,1091,-1899,1455,-404,-2546,-2707,1131,121,-1010,-2303,-2505,-3313,1859,-929,-2909,-4081,4243,1535,121,121,3717,5495,-2586,-889,647,687,5374,-1940,1535,1172,-2869,-1333,404,-2182,-2384,-525,81,849,606,-4808,-202,-687,-1212,-4324,-525,606,3879,1455,1091,-4808,-4808,-1535,-1616,-283,-2505,-2182,2748,-849,-2101,-2344,444,-2020,3233,4162,-4000,444},
//	{44,3051,-392,2920,2005,-218,1744,3749,-828,-872,3792,-1482,-3269,-4664,-1961,-218,2136,2833,2441,-1395,-1351,3443,1220,392,-1787,1569,-2310,-567,5928,5143,392,-567,-4925,-436,2833,4795,1264,-2877,-1874,-4664,-5187,-2354,828,-3008,-131,1177,-3269,1438,-697,-2354,-3051,5231,3008,0,0,-1220,1744,305,-3836,0,2223,654,785,3051,2397,-5187,2441,-349,2615,436,654,1438,44,5928,-1831,-3574,-4359,5928,-349,-872,4184,1395,-3008,-5187,131,-87,1613,3661,-44,-5187,1613,872,1831,741,-1787,828,2354,1613,4620,-2702,4315,-1003,1395,-3661,-1700,3661,1133,1264,-2833,-1438,-1569,-567,3661,828,-828,-1003,4010,1918,-1090,1090,2397,2049,-828,-436,-523,-567,-2790,697,-4184,1264,1787,218,-4533,1351,-1700,-436,-4010,5579,-1787,1787,1046,-5187,3531,4664,-3836,-4228,5841,741,-3966,-1308,-1220,3487,-349,-5187,-4707,4315,3182,5318,-2179,-872,-5187,5884,-610,3356,-44,-1700,3879,-2092,2833,2267,2267,-1308,-1046,305,3661,1961,-2397,5928,-131,-2833,-5187,-1831,1438,479,-2049,-2528,5928,-2833,-3661,-2964,-1656,-1482,4097,2049,5274,-2397,-2746,-1177,-1569,-5187,2092,915,-5187,3356,-4446,2310,-174,5056,-218,-567,5928,-1613,1395,2354,-3008,5143,3705,4010,-392,305,-4315,1003,-1831,-3269,-1133,-1613,697,-3574,-5187,-5100,4446,2049,-785,3225,-2267,-3749,4795,3138,349,-1613,-4315,305,-828,1569,0,-2659,-2267,2484,5405,262,-654,436,-3879,2659,2964,-174,4402,-523,-2659,2267,5231,-1700,-3225,1177,1308,5928,-3443,2397,-4925,-1438,1700,-262,-1656,174,3618,-2659,2484,-436,-3705,2179,2615,-2310,5797,3269,-1744,-5187,3138,-2267,1918,741,3879,1220,-2659,5623,-2049,610,-5187,1831,1438,-2659,1700,523,4882,-741,-610,-2790,2920,610,-3531,5928,-2005,-5187,-1700,2136,-2136,-305,-3182,-2005,-4097,-1177,-4359,-2615,-785,87,-5100,-262,-5187,-1090,828,523,1308,-785,1526,-3138,741,-741,4402,2179,-2920,-872,1613,-3182,-2615,-4272,-3836,-2179,44,1090,44,4925,-1526,-1438,-2572,-1308,3182,3008,-2092,2484,-3356,915,523,-1787,1438,2964,1177,-1264,-828,2659,-174,2528,-5187,3618,-131,-5187,-2092,436,2223,2615,-4969,-5187,-2223,-5187,5928,-3008,959,-1700,392,-3966,5928,-5187,2223,-1438,2920,-959,-3531,-5187,-2964,610,-2223,3138,-4707,5928,1569,1961,-131,1177,-436,5928,131,1351,610,-2702,-349,-5187,-5187,-1656,2179,-5187,131,3269,3792,5492,523,1526,-915,-3836,4315,218,-1177,523,-2702,1220,741,3356,3008,4097,-3008,-3356,-1090,-4577,-4097,2092,-2049,-2484,1569,-5187,1961,-2136,-654,-785,4882,-392,2659,5928,-1438,349,3313,-1918,44,1874,-1046,-5187,2223,-262,2005,131,2659,1961,5928,2702,1526,741,2790,4141,87,2441,-262,-2136,-1046,1831,-1308,-1046,-2092,1177,-2964,-1351,-2223,-872,1526,3095,4664,-3400,-2005,2441,-174,-3836,-2528,-1744,-828,2354,4097,1351,3618,-785,1874,-5187,-3487,-1700,-5187,-349,610,5928,},
//	{-3477,-1500,-2379,-4208,-1061,1757,-2415,1537,-3477,-293,4062,476,-2598,586,3843,3696,2562,3952,-4135,329,1354,4977,4977,3550,-183,-2562,-4355,4977,73,1061,2635,2232,769,-622,3001,-1683,-256,-293,4977,1757,-3989,4977,1500,-1793,3550,-769,-4355,915,1903,-3074,220,-2086,4977,439,3513,-4355,2306,-1537,2159,-2086,2013,-4355,4794,-2818,3403,-2928,329,-769,4977,-3184,-4355,-183,-2562,4684,-4355,3769,-3916,-110,4977,4245,-1610,-951,-1757,-110,-146,-2635,3367,-3111,-1171,-2452,-1464,-293,3623,3184,-1537,-1903,1317,3769,4977,915,-37,-4026,4172,4062,-2196,4977,3001,1940,2745,2086,-2159,4355,2232,1720,2781,-1940,-3440,-3477,256,1683,1610,-2671,-1757,-2269,1244,1427,-2671,-2342,-4355,403,1281,-4355,4977,-1976,-659,2928,549,2196,-769,1061,4977,329,512,1061,4977,-4245,110,3147,-2196,3403,2452,-3294,4245,3477,-1793,-622,-439,-4355,-3220,-2928,659,2049,4977,-4355,2671,-1464,-4355,4501,2781,-1830,-2196,549,-110,-2415,4977,2891,-4355,3696,1244,-4355,2562,4245,-4355,-915,1354,-1354,2708,-1391,-1134,1208,-73,-1903,4501,-1866,4977,1647,-4355,4977,-805,-2159,-3586,512,1354,1610,4867,4977,1903,4977,-4355,3806,2598,110,-1500,1610,4318,-1427,-2086,-2159,-1683,-1134,-4208,-4355,-1061,-2452,-4355,403,1940,-3843,4977,1866,2086,3001,3001,4977,-2708,1391,4538,1610,-4355,-1025,-3879,-1903,-2854,-4355,1098,-915,3513,-4355,-4355,4099,1317,-3806,-2013,988,2415,-2232,110,1244,4355,-1281,1464,732,329,2159,1757,2086,1866,4977,4574,-3660,-2562,4465,4977,659,-4318,1134,2196,146,2086,4977,2525,-4355,476,3550,-3147,4977,3916,-2086,-951,439,878,3440,1391,-1281,3220,1391,-2086,2049,2854,-4099,-3696,1208,-2342,842,4208,-220,4135,2562,110,1574,1208,4538,-1061,1647,-1574,-1683,-4355,-3806,4208,4684,-2562,-4355,-915,3477,-4355,-3952,2379,2891,3696,4977,805,-1903,1647,622,1830,-4355,-2562,110,4282,4208,4977,-2086,-3843,1281,-1427,3184,-769,4977,4684,-4355,-2525,-329,1208,2818,-2342,-1537,1830,-4355,366,3220,-1171,1647,732,-329,-4355,-2854,732,2452,951,2379,988,2525,-1244,4977,-4355,403,1903,-695,-549,1903,732,-915,-4355,1391,-2159,3623,-2488,-4355,4977,-1025,-329,256,-183,-4172,2964,-2598,4428,2342,-769,1903,-2781,-842,732,-1025,73,-1976,2818,1720,-1244,-4355,-3586,-3147,878,-4355,1208,-4355,293,2452,4977,3074,4977,-1610,-3147,-73,4977,-2598,4831,2232,732,-3733,-4355,183,-3403,-2525,4977,3074,-1061,-2854,1720,2708,1720,-4245,1866,4684,-878,2745,1940,-2379,-1025,4465,-3952,-1391,-220,-329,-549,4977,-4355,-4355,549,-2013,-2854,2013,-3733,-3074,-2635,-73,-3879,878,-4355,-4355,1061,695,-512,1683,-1134,-3952,-293,1574,-2818,-1940,439,-2379,4977,-4355,1537,-329,4391,476,-73,-4355,-4355,-2854,1537,3330,-366,37,-146,220,-4355,3586,-695,2269,439,4977,-2123,329,-1354,3440,2049,-3330,1866,3550,},
//	{-3842,1572,-1877,2707,4584,1135,1615,-2357,-2401,-1397,3449,1528,-2838,-1834,-917,960,-4671,1310,437,-175,-2619,218,5937,3143,-873,1266,1703,349,218,-1135,-1528,3973,3231,-4933,611,4147,-2008,-4016,-1310,-3885,-2401,2707,-524,-4933,3929,-655,1353,480,-5195,-1528,1877,1484,5937,-2925,-5064,-3405,-1484,1921,175,-568,5937,-4933,5937,-5195,-218,-1310,437,3012,5020,218,-2357,-1790,-349,-218,87,-3798,-611,-2139,-3187,3536,1484,-3798,-1659,-4104,-1572,-917,2270,-1790,3449,-3143,-960,2270,218,1222,-3885,-3973,1572,4715,5413,1484,2576,44,1091,1048,786,-1004,-3929,-4671,1353,2576,1790,218,-3361,1397,-2052,-829,-1834,5937,1965,-3143,568,1572,3361,1528,2314,-5195,131,-1746,3667,-5195,5937,4671,1659,3492,-5195,-2139,5282,829,-2270,-2357,-3973,-3623,4758,3623,1484,5588,1834,-1091,-3798,-437,1746,44,1965,5937,4453,-131,-480,4104,1091,-2838,-742,1179,2750,1877,-786,-1310,44,-917,-306,1484,480,4846,-655,873,1921,-2401,-393,-2445,1135,-2401,-1877,2881,-2881,5937,3405,-742,4147,-1222,-1091,-4191,87,2357,-611,-3449,-1397,1572,3842,-3143,1746,-3798,-2357,-2750,-1310,5937,1397,-393,3842,-1048,-4366,2052,-4933,2532,1746,-960,3885,960,-3449,2750,742,-480,2663,-5195,3143,-2095,-349,2576,5588,1004,5937,5064,218,-524,3405,87,2750,-4453,-1659,-960,1703,-4060,-3143,917,-4235,-2619,3885,-5195,5937,-3667,4278,-5195,3885,1659,4977,4540,1048,-2576,1441,1659,3798,-437,87,-1790,-2532,5937,1353,3929,786,-1048,742,-437,2357,-437,-3361,5937,1834,-742,-4540,-2183,-3711,3100,5937,1746,-44,1179,-1266,1091,-2270,-960,829,3536,44,1965,1746,-4366,1965,2794,1397,-524,1877,-3798,568,-87,-218,2052,-306,-3536,-1528,-1484,-3012,2052,5937,-524,-1266,742,5675,-3405,1179,-873,-1266,1397,-3492,1484,5937,829,-1659,2750,3929,-3231,1135,-2139,-786,-524,-2270,1091,218,-1790,-5195,1266,698,-4584,-2488,-4104,87,4933,-349,2183,5937,2095,3842,-1135,2750,-2052,-5195,-5195,2357,-5020,3143,-917,-4060,-437,611,-1746,-611,-1615,-2663,-1877,742,1353,-917,2925,5937,3842,-611,-1397,-1222,2532,2532,-2881,1703,-1703,524,-2008,-5195,-4409,-2838,-218,1222,-3973,-2969,960,-2445,-5195,-5195,1266,2052,1397,1266,2576,524,-524,262,1921,-4104,1615,-5195,2183,1659,-3667,-4191,-2314,-1484,-1397,-349,44,-175,-2707,-5195,1746,5937,1048,0,5108,4977,218,5937,1135,437,2314,44,-306,1266,-3492,-3274,-44,-3623,-262,829,2052,-1965,-4802,568,-2226,-5195,-306,-1266,3187,-2619,698,-4889,5937,-2183,524,873,-2532,175,-87,-4016,-2576,1965,-873,1528,-3274,1310,-3012,2139,-2008,611,3187,-5195,1484,-4627,-2139,-3361,4715,1441,-1135,4060,-1572,1484,-4497,-1484,2226,-2794,4191,4366,4235,-655,-87,-786,-1572,-5195,-655,873,-5064,-3623,-4409,4366,-3885,349,1397,2576,87,-1484,4366,-1397,-2838,-349,3449,2139,-2838,-2925,-4758,},
//};


bool not_duplicate(char* id){
	for(int i = 0; i < db_end_idx;i++){
		if(strcmp(db_nameStr[i],id)==0){
			return false;
		}
	}
	return true;

}

void append_name(char* name_entered){
	if(id_check ==1){
		if(strlen(name_entered)>0){
			strcpy(db_nameStr[db_end_idx-1],name_entered);
		}
		else{
			face_count++;
		}
		printf("Embedding '%s' added to database\n",db_nameStr[db_end_idx-1]);
	}


	id_check = 0;
	add_embedding_mode=0;

}


void print_list(){
	for(int i=0; i <db_end_idx; i++){
		printf("%d: %s\n",i+1,db_nameStr[i]); // index starts at 1
	}
	printf("\n");

}
#if PDMA
static int32_t pdma_ch_transfer(uint64_t output_data_phys, void* source_buffer,int offset,int size,vbx_cnn_t *vbx_cnn,int32_t channel){
	uint64_t srcbuf=0x3000000000 + (uint64_t)(uintptr_t)virt_to_phys(vbx_cnn, source_buffer);
	return pdma_ch_cpy(output_data_phys + offset, srcbuf, size, channel);
}
#endif
void embedding_calc(fix16_t* embedding, struct model_descr_t* recognition_model){
	fix16_t sum = 0;
	fix16_t temp[EMBEDDING_LENGTH];
	int8_t* output_buffer_int8 = (int8_t*)(uintptr_t)recognition_model->model_output_buffer[0];
	int32_t zero_point = model_get_output_zeropoint(recognition_model->model,0);
	fix16_t scale = model_get_output_scale_fix16_value(recognition_model->model,0);
	for(int n = 0; n < recognition_model->model_output_length[0]; n++){
		temp[n] = int8_to_fix16_single(output_buffer_int8[n], scale,  zero_point);
		sum += fix16_sq(temp[n]);
	}
	fix16_t norm = fix16_div(fix16_one, fix16_sqrt(sum));
	for(int n = 0; n < recognition_model->model_output_length[0]; n++){	
		embedding[n] = fix16_mul(temp[n], norm);
	}	
}

void delete_embedding(char* input_buf,struct model_descr_t* models,uint8_t modelIdx){

	int rem_ind = -1;
	size_t len = strlen(input_buf);
	memmove(input_buf, input_buf, len);
	rem_ind = atoi(input_buf) - 1; // index starts at 1 (needed as atoi returns 0 if not an integer)
	if((rem_ind <db_end_idx) && (rem_ind >= 0) && len>1){
		printf("\nEmbedding '%s' removed\n", db_nameStr[rem_ind]);		
		for(int re = rem_ind; re<db_end_idx;re++)
		{	
			for(int i = 0; i< EMBEDDING_LENGTH;i++)
				db_embeddings[re][i] = db_embeddings[re+1][i];	
			db_nameStr[re] = db_nameStr[re+1];
		}
		trackClean(models,modelIdx);
		db_end_idx--;
	} else if (len > 1) {
		printf("Embedding index invalid\n");
	}
	delete_embedding_mode = 0;
	printf("Exiting +/- embedding mode \n");

}


void tracksInit(struct model_descr_t* models){
	int use_plate = 0;
	struct model_descr_t *recognition_model = models;
	if(!strcmp(recognition_model->post_process_type, "LPR"))
		use_plate =1;
	track_t *tracks = (track_t*)calloc(MAX_TRACKS,sizeof(track_t));
	Tracker_t *pTracker = (Tracker_t *)calloc(1,sizeof(Tracker_t));
	track_t** pTracks = (track_t**)calloc(MAX_TRACKS,sizeof(track_t*));
	recognition_model->tracks = tracks;
	recognition_model->pTracker = pTracker;
	recognition_model->pTracks = pTracks;
	trackerInit(pTracker, MAX_TRACKS, pTracks, tracks,use_plate);

}


void trackClean(struct model_descr_t* models, uint8_t modelIdx){

	struct model_descr_t *recognition_model = models + modelIdx;
	free(recognition_model->tracks);
	free(recognition_model->pTracker);
	free(recognition_model->pTracks);
	recognition_model->tracks = NULL;
	recognition_model->pTracker = NULL;
	recognition_model->pTracks = NULL;

}
int facing_front(object_t* object){
	
	int bbox_h = fix16_to_int(object->box[3] - object->box[1]);
	int center_y = fix16_to_int(object->box[3] + object->box[1]) / 2;
	int bbox_h_frontal = bbox_h * 0.5;
	int frontal_bbox_top =      center_y - (bbox_h_frontal/2);
	int frontal_bbox_bottom =   center_y + (bbox_h_frontal/2);
	
	int mouth_left = fix16_to_int(object->points[3][0]);
	int mouth_right = fix16_to_int(object->points[4][0]);
	int nose_x = fix16_to_int(object->points[2][0]);
	int nose_y = fix16_to_int(object->points[2][1]);

	if(nose_x < mouth_left || nose_x > mouth_right || nose_y < frontal_bbox_top || nose_y > frontal_bbox_bottom){
		return 0;
	}
	else{
		return 1;
	}	
}


short recognitionDemoInit(vbx_cnn_t* the_vbx_cnn, struct model_descr_t* models, uint8_t modelIdx, int has_attribute_model, int screen_height, int screen_width, int screen_y_offset, int screen_x_offset) {
	struct model_descr_t *detect_model = models + modelIdx;
	// Allocate memory for Models
    // Allocate Memory needed for the Detect Model buffers
	detect_model->model_io_buffers  = vbx_allocate_dma_buffer(the_vbx_cnn, (1+model_get_num_outputs(detect_model->model))*sizeof(detect_model->model_io_buffers[0]), 0);
	if(!detect_model->model_io_buffers){
		printf("Memory allocation issue for model io buffers.\n");
		return -1;
	}
	// Allocate the buffers for output for Detect Model
    for (unsigned o = 0; o < model_get_num_outputs(detect_model->model); ++o) {
		detect_model->model_output_length[o] = model_get_output_length(detect_model->model, o);
		detect_model->pipelined_output_buffers[0][o] = vbx_allocate_dma_buffer(the_vbx_cnn, model_get_output_length(detect_model->model, o)*sizeof(fix16_t), 0);
		detect_model->pipelined_output_buffers[1][o] = vbx_allocate_dma_buffer(the_vbx_cnn, model_get_output_length(detect_model->model, o)*sizeof(fix16_t), 0);
		detect_model->model_io_buffers[o+1] = (uintptr_t)detect_model->pipelined_output_buffers[0][o];
		if(!detect_model->pipelined_output_buffers[0][o] ||!detect_model->pipelined_output_buffers[1][o] ){
			printf("Memory allocation issue for model output buffers.\n");
			return -1;	
		}		
		
	}
	// Allocate the buffer for input for Detect Model
	detect_model->model_input_buffer = vbx_allocate_dma_buffer(the_vbx_cnn, model_get_input_length(detect_model->model, 0)*sizeof(uint8_t), 0);
	detect_model->model_io_buffers[0] = (uintptr_t)detect_model->model_input_buffer;
	if(!detect_model->model_input_buffer) {
		printf("Memory allocation issue for model input buffers.\n");
		return -1;	
	}
	detect_model->buf_idx = 0;
	detect_model->is_running = 0;

	// Allocate memory for Recognition Model I/Os
	// Specify the input size for Recognition Model
	struct model_descr_t *recognition_model = models + modelIdx + 1;
	recognition_model->coord4 = (fix16_t*)malloc(8*sizeof(fix16_t));

	if(!strcmp(recognition_model->post_process_type, "ARCFACE")){
		recognition_model->coord4[0] = F16(38.2946);
		recognition_model->coord4[1] = F16(51.6963);
		recognition_model->coord4[2] = F16(73.5318);
		recognition_model->coord4[3] = F16(51.5014);
		recognition_model->coord4[4] = F16(56.0252);
		recognition_model->coord4[5] = F16(71.7366);
		recognition_model->coord4[6] = F16(56.1396);
		recognition_model->coord4[7] = F16(92.284805);
	} else if(!strcmp(recognition_model->post_process_type, "SPHEREFACE")){
		recognition_model->coord4[0] = F16(30.2946);
		recognition_model->coord4[1] = F16(51.6963);
		recognition_model->coord4[2] = F16(65.5318);
		recognition_model->coord4[3] = F16(51.5014);
		recognition_model->coord4[4] = F16(48.0252);
		recognition_model->coord4[5] = F16(71.7366);
		recognition_model->coord4[6] = F16(48.1396);
		recognition_model->coord4[7] = F16(92.2848);
	} else if(!strcmp(recognition_model->post_process_type, "LPR")){
		recognition_model->coord4[0] = fix16_from_int(1);  //LT
		recognition_model->coord4[1] = fix16_from_int(1);
		recognition_model->coord4[2] = fix16_from_int(146-1);  //RT
		recognition_model->coord4[3] = fix16_from_int(1);
		recognition_model->coord4[6] = fix16_from_int(1);  //LB
		recognition_model->coord4[7] = fix16_from_int(34-1);
	}
	else {
		printf("Recognition Model does not have an expected input length\n");
		return -1;
	}
	// Allocate the buffer for input for Recognition Model
	recognition_model->model_input_buffer = vbx_allocate_dma_buffer(the_vbx_cnn, model_get_input_length(recognition_model->model, 0)*sizeof(uint8_t), 0);
	if(!recognition_model->model_input_buffer){
		printf("Memory allocation issue with recognition input buffer.\n");
		return -1;
	}
	// Specify the output size for Recognition Model
	recognition_model->model_output_length[0] = model_get_output_length(recognition_model->model, 0);
	// Allocate the buffer for output for Recognition Model
	recognition_model->model_output_buffer[0] = vbx_allocate_dma_buffer(the_vbx_cnn, recognition_model->model_output_length[0]*sizeof(fix16_t), 0);
	if(!recognition_model->model_output_buffer[0]){
		printf("Memory allocation issue with recognition output buffer.\n");
		return -1;
	}
	// Allocate Memory needed for the Recognition Model buffers
	recognition_model->model_io_buffers  = vbx_allocate_dma_buffer(the_vbx_cnn, (1+model_get_num_outputs(recognition_model->model))*sizeof(recognition_model->model_io_buffers[0]), 0);
	if(!recognition_model->model_io_buffers){
		printf("Memory allocation issue with recognition io buffers.\n");
		return -1;
	}
	recognition_model->model_io_buffers[0] = (uintptr_t)recognition_model->model_input_buffer;
	recognition_model->model_io_buffers[1] = (uintptr_t)recognition_model->model_output_buffer[0];

	// Allocate Memory needed for Warp Affine Tranformation
	if (warp_temp_buffer == NULL) warp_temp_buffer = vbx_allocate_dma_buffer(the_vbx_cnn, 224*224*3, 0);


	// Allocate memory for Attribute Model I/Os
	if (has_attribute_model) {
		// Specify the input size for Attribute Model
		struct model_descr_t *attribute_model = models + modelIdx + 2;
		// Allocate the buffer for input for Attribute Model
		attribute_model->model_input_buffer = vbx_allocate_dma_buffer(the_vbx_cnn, model_get_input_length(attribute_model->model, 0)*sizeof(uint8_t), 0);
		if(!attribute_model->model_input_buffer){
			printf("Memory allocation issue with attribute input buffer.\n");
			return -1;
		}
		// Specify the output size for Attribute Model
		attribute_model->model_output_length[0] = model_get_output_length(attribute_model->model, 0); // age output, expecting length 1
		attribute_model->model_output_length[1] = model_get_output_length(attribute_model->model, 1); // gender output, expecting length 2
		// Allocate the buffer for output for Attribute Model
		attribute_model->model_output_buffer[0] = vbx_allocate_dma_buffer(the_vbx_cnn, attribute_model->model_output_length[0]*sizeof(fix16_t), 0);
		attribute_model->model_output_buffer[1] = vbx_allocate_dma_buffer(the_vbx_cnn, attribute_model->model_output_length[1]*sizeof(fix16_t), 0);
		if(!attribute_model->model_output_buffer[0] ||!attribute_model->model_output_buffer[1]){
			printf("Memory allocation issue with attribute output buffers.\n");
			return -1;
		}
		// Allocate Memory needed for the Attribute Model buffers
		attribute_model->model_io_buffers  = vbx_allocate_dma_buffer(the_vbx_cnn, (1+model_get_num_outputs(attribute_model->model))*sizeof(attribute_model->model_io_buffers[0]), 0);
		if(!attribute_model->model_io_buffers ){
			printf("Memory allocation issue with attribute io buffers.\n");
			return -1;
		}
		// I/Os of the attribute model
		attribute_model->model_io_buffers[0] = (uintptr_t)attribute_model->model_input_buffer;
		attribute_model->model_io_buffers[1] = (uintptr_t)attribute_model->model_output_buffer[0];
		attribute_model->model_io_buffers[2] = (uintptr_t)attribute_model->model_output_buffer[1];
	}
	
	return 1;
}


int runRecognitionDemo(struct model_descr_t* models, vbx_cnn_t* the_vbx_cnn, uint8_t modelIdx, int use_attribute_model, int screen_height, int screen_width, int screen_y_offset, int screen_x_offset) {
	int err;
	int screen_stride = 0x2000;
	struct model_descr_t *detect_model = models+modelIdx;
	struct model_descr_t *recognition_model = models+modelIdx+1;
	struct model_descr_t *attribute_model = models+modelIdx+2;
	int colour;
	int use_plate = 0;
	char label[256];
	char gender_char;
	int status;
	uint32_t offset;
	int detectInputH = model_get_input_shape(detect_model->model, 0)[2];
	int detectInputW = model_get_input_shape(detect_model->model, 0)[3];
	//Tracks are initialized if current model has no previous tracks
	if(recognition_model->pTracker == NULL || recognition_model->pTracks == NULL){
		tracksInit(recognition_model);
	}
	// Start processing the network if not already running - 1st pass (frame 0)
	if(!detect_model->is_running) {	
#ifdef HLS_RESIZE
		resize_image_hls(SCALER_BASE_ADDRESS,(uint32_t*)(intptr_t)(*PROCESSING_FRAME_ADDRESS),
				screen_width, screen_height, screen_stride, screen_x_offset, screen_y_offset,
				(uint8_t*)virt_to_phys(the_vbx_cnn, (void*)detect_model->model_input_buffer),
				detectInputW, detectInputH);
#else
		*BLUE_DDR_FRAME_START_ADDR  =  SCALER_FRAME_ADDRESS + (2*detectInputW*detectInputH);
		*GREEN_DDR_FRAME_START_ADDR =  SCALER_FRAME_ADDRESS + (1*detectInputW*detectInputH);
		*RED_DDR_FRAME_START_ADDR   =  SCALER_FRAME_ADDRESS + (0*detectInputW*detectInputH);
		offset = (*PROCESSING_FRAME_ADDRESS) - 0x70000000;
		detect_model->model_input_buffer = (uint8_t*)(uintptr_t)(SCALER_FRAME_ADDRESS + offset);
		detect_model->model_io_buffers[0] = (uintptr_t)detect_model->model_input_buffer - the_vbx_cnn->dma_phys_trans_offset;	
#endif		
		//detect_model->model_io_buffers[0] = (uintptr_t)model_get_test_input(detect_model->model,0);
		// Start Detection model
		err = vbx_cnn_model_start(the_vbx_cnn, detect_model->model, detect_model->model_io_buffers);		
		if(err != 0) return err;
		detect_model->is_running = 1;
	}
	
	status = vbx_cnn_model_poll(the_vbx_cnn); //vbx_cnn_model_wfi(the_vbx_cnn); // Check if model done
	while(status > 0) {
		for(int i =0;i<1000;i++);
		status = vbx_cnn_model_poll(the_vbx_cnn);
	}
	
	if(status < 0) {
		return status;
	} else if (status == 0) { // When  model is completed

		int length=0;
//pdma copy buffers
#if PDMA
	vbx_cnn_io_ptr_t pdma_buffer[model_get_num_outputs(detect_model->model)];
	int output_offset=0;
	for(int o =0; o<(int)model_get_num_outputs(detect_model->model);o++){
		int output_length = model_get_output_length(detect_model->model, o);
		pdma_ch_transfer(pdma_out,(void*)detect_model->pipelined_output_buffers[detect_model->buf_idx][o],output_offset,output_length,the_vbx_cnn,pdma_channel);
		pdma_buffer[o] = (vbx_cnn_io_ptr_t)(pdma_mmap_t + output_offset);
		output_offset+= output_length;
	}
#endif
// Swap pipeline IO
		for (int o = 0; o <  model_get_num_outputs(detect_model->model); o++) {
			detect_model->model_io_buffers[o+1] = (uintptr_t)detect_model->pipelined_output_buffers[!detect_model->buf_idx][o];	
		}	
//
		object_t objects[MAX_TRACKS];
		snprintf(label,sizeof(label),"Face Recognition Demo %dx%d  %d fps",detectInputW,detectInputH,fps);
		if(use_attribute_model)
			snprintf(label,sizeof(label),"Face Recognition + Attribute Demo %dx%d  %d fps",detectInputW,detectInputH,fps);
		if(!strcmp(detect_model->post_process_type, "BLAZEFACE")) {
			// Post Processing BlazeFace output
			int anchor_shift = 1;
			if (detectInputH == 256 && detectInputW == 256) anchor_shift = 0;

			length = post_process_blazeface(objects, detect_model->pipelined_output_buffers[detect_model->buf_idx][0], detect_model->pipelined_output_buffers[detect_model->buf_idx][1],
					detect_model->model_output_length[0], MAX_TRACKS, fix16_from_int(1)>>anchor_shift);

		}
		else if (!strcmp(detect_model->post_process_type, "RETINAFACE")) {
			// Post Processing RetinaFace output
			fix16_t confidence_threshold=F16(0.76);
			fix16_t nms_threshold=F16(0.34);
			// ( 0 1 2 3 4 5 6 7 8) -> (5 4 3 8 7 6 2 1 0)
			fix16_t *outputs[9];

			fix16_t** output_buffers = detect_model->pipelined_output_buffers[detect_model->buf_idx];

			outputs[0]=(fix16_t*)(uintptr_t)output_buffers[5];
			outputs[1]=(fix16_t*)(uintptr_t)output_buffers[4];
			outputs[2]=(fix16_t*)(uintptr_t)output_buffers[3];
			outputs[3]=(fix16_t*)(uintptr_t)output_buffers[8];
			outputs[4]=(fix16_t*)(uintptr_t)output_buffers[7];
			outputs[5]=(fix16_t*)(uintptr_t)output_buffers[6];
			outputs[6]=(fix16_t*)(uintptr_t)output_buffers[2];
			outputs[7]=(fix16_t*)(uintptr_t)output_buffers[1];
			outputs[8]=(fix16_t*)(uintptr_t)output_buffers[0];

			length = post_process_retinaface(objects, MAX_TRACKS, outputs, detectInputW, detectInputH,
					confidence_threshold, nms_threshold);
		}
		else if (!strcmp(detect_model->post_process_type, "SCRFD")) {
			// Post Processing SCRFD output
			fix16_t confidence_threshold=F16(0.8);
			fix16_t nms_threshold=F16(0.34);
			//( 0 1 2 3 4 5 6 7 8)->(2 5 8 1 4 7 0 3 6)
#if PDMA
			fix16_t** output_buffers = (fix16_t**)pdma_buffer;
#else			
			fix16_t** output_buffers = detect_model->pipelined_output_buffers[detect_model->buf_idx];
#endif

			//fix16_t* fix16_buffers[9];
			int8_t* output_buffer_int8[9];
			int zero_points[9];
			fix16_t scale_outs[9];
			for(int o=0; o<model_get_num_outputs(detect_model->model); o++){
				int *output_shape = model_get_output_shape(detect_model->model,o);
				int ind = (output_shape[1]/8)*3 + (2-(output_shape[2]/18)); //first dim should be {2,8,20} second dim should be {9,18,36}
				output_buffer_int8[ind]= (int8_t*)(uintptr_t)output_buffers[o];
				zero_points[ind]=model_get_output_zeropoint(detect_model->model,o);
				scale_outs[ind]=model_get_output_scale_fix16_value(detect_model->model,o);
			}		
			length = post_process_scrfd_int8(objects, MAX_TRACKS, output_buffer_int8, zero_points, scale_outs, detectInputW, detectInputH,
				confidence_threshold,nms_threshold,detect_model->model);

		}
		else if (!strcmp(detect_model->post_process_type, "YUNET")) {
			
			fix16_t confidence_threshold=F16(0.72);
			fix16_t nms_threshold=F16(0.45);

#if PDMA
			fix16_t** output_buffers = (fix16_t**)pdma_buffer;
#else
			fix16_t** output_buffers = detect_model->pipelined_output_buffers[detect_model->buf_idx];
#endif

			int8_t* output_buffer_int8[12];
			int zero_points[12];
			fix16_t scale_outs[12];
			int clsSeen[3] = {0,0,0};

			for(int o = 0; o < model_get_num_outputs(detect_model->model); o++) {

				int *output_shape = model_get_output_shape(detect_model->model,o);
				int channels = output_shape[1];
				int height = output_shape[2];

				int map_num = 0;								 	// stride=8
				if(height == detectInputH/16){
					map_num = 1;									// stride=16
				}else if(height == detectInputH/32) {
					map_num = 2; 									// stride=32
				} 

				int ind = 6 + map_num; 								// objectness score
				if(channels == 4) {
					ind = 3 + map_num; 								// box location
				} 
				else if(channels == 10) {
					ind = 9 + map_num; 								// landmarks
				}
				else if(!clsSeen[map_num]) { 
					ind = map_num; 									// class score
					clsSeen[map_num] = 1;							// Note: cannot distiguish with channels bc object channel also = 1
				 }
			
				output_buffer_int8[ind]= (int8_t*)(uintptr_t)output_buffers[o];
				zero_points[ind] = model_get_output_zeropoint(detect_model->model,o);
				scale_outs[ind] = model_get_output_scale_fix16_value(detect_model->model,o);
			}

			length = post_process_yunet_int8(objects, MAX_TRACKS, output_buffer_int8, zero_points, scale_outs, detectInputW, detectInputH,
				confidence_threshold, nms_threshold);

		}
		else if (!strcmp(detect_model->post_process_type, "LPD")) {
			// Post Processing LPD output
			use_plate = 1;
			fix16_t confidence_threshold=F16(0.55);
			fix16_t nms_threshold=F16(0.2);
			int num_outputs = model_get_num_outputs(detect_model->model);
			int8_t* output_buffer_int8[9];
			int zero_points[9];
			fix16_t scale_outs[9];
			fix16_t** output_buffers = detect_model->pipelined_output_buffers[detect_model->buf_idx];
			for (int o = 0; o < num_outputs; o++) {				
				int *output_shape = model_get_output_shape(detect_model->model,o);
				int ind = 2*(output_shape[2]/18) + (output_shape[1]/6); 
				output_buffer_int8[ind]= (int8_t*)(uintptr_t)output_buffers[o];
				zero_points[ind]=model_get_output_zeropoint(detect_model->model,o);
				scale_outs[ind]=model_get_output_scale_fix16_value(detect_model->model,o);
			}

			length = post_process_lpd_int8(objects, MAX_TRACKS, output_buffer_int8, detectInputW, detectInputH,
					confidence_threshold,nms_threshold, num_outputs, zero_points, scale_outs);
							
			snprintf(label,sizeof(label),"Plate Recognition Demo %dx%d  %d fps",detectInputW,detectInputH,fps);
		}
		
		draw_label(label,20,2,overlay_draw_frame,2048,1080,WHITE);

		int tracks[length];
		// If objects are detected			
		if (length > 0) {
			fix16_t confidence;
			char* name="";
			int is_frontal_view ;

			fix16_t x_ratio = fix16_div(screen_width, detectInputW);
			fix16_t y_ratio = fix16_div(screen_height, detectInputH);

			for(int f = 0; f < length; f++) {
				objects[f].box[0] = fix16_mul(objects[f].box[0], x_ratio);
				objects[f].box[1] = fix16_mul(objects[f].box[1], y_ratio);
				objects[f].box[2] = fix16_mul(objects[f].box[2], x_ratio);
				objects[f].box[3] = fix16_mul(objects[f].box[3], y_ratio);
				for(int p = 0; p < 5; p++) {
					objects[f].points[p][0] =fix16_mul(objects[f].points[p][0],x_ratio);
					objects[f].points[p][1] =fix16_mul(objects[f].points[p][1],y_ratio);
				}
			}
			if (add_embedding_mode) {
				object_t* object = &(objects[0]);
				for(int i=0;i<length;i++){
					int new_w = fix16_to_int(objects[i].box[2]) - fix16_to_int(objects[i].box[0]);
					int new_h = fix16_to_int(objects[i].box[3]) - fix16_to_int(objects[i].box[1]);
					if(new_h > (fix16_to_int(object->box[3]) - fix16_to_int(object->box[1])) && (new_w>fix16_to_int(object->box[2]) - fix16_to_int(object->box[0])))
						object = &(objects[i]);				
				}
				recognizeObject(the_vbx_cnn, recognition_model, object, detect_model->post_process_type,
						screen_height, screen_width, screen_stride, screen_y_offset, screen_x_offset);

				// Start Recognition model
				err = vbx_cnn_model_start(the_vbx_cnn, recognition_model->model, recognition_model->model_io_buffers);
				if(err != 0) return err;

				err = vbx_cnn_model_poll(the_vbx_cnn); // Wait for the Recognition model
				while(err > 0) {
					for(int i =0;i<1000;i++);
					err = vbx_cnn_model_poll(the_vbx_cnn);
				}
				if(err < 0) return err;
				fix16_t embedding[EMBEDDING_LENGTH] = {0};
				embedding_calc(embedding,recognition_model);

				int box_thickness=5;
				// Compute the offsets
				int x = fix16_to_int(object->box[0]) + screen_x_offset;
				int y = fix16_to_int(object->box[1]) + screen_y_offset;
				int w = fix16_to_int(object->box[2]) - fix16_to_int(object->box[0]);
				int h = fix16_to_int(object->box[3]) - fix16_to_int(object->box[1]);
				colour = GET_COLOUR(0, 0, 255, 255); //R,G,B,A
				snprintf(label,sizeof(label),"Strong Embedding");
				int frontal_view=facing_front(object);
				if(!frontal_view || (h<112)){
					colour = GET_COLOUR(255,255,0,255);
					snprintf(label,sizeof(label),"Weak Embedding");
				}
				if( x > 0 &&  y > 0 && w > 0 && h > 0) {	
					draw_label(label,x,y+h+screen_y_offset+box_thickness,overlay_draw_frame,2048,1080,WHITE);
					draw_box(x,y,w,h,box_thickness,colour,overlay_draw_frame,2048,1080);
					// Draw the points of detected objects
					for(int p = 0; p < 5; p++) {
						draw_rectangle(fix16_to_int(object->points[p][0])+screen_x_offset,
								fix16_to_int(object->points[p][1])+screen_y_offset, 4, 4, colour,
								overlay_draw_frame, 2048,1080);
					}
				}

				if (capture_embedding) {
					matchEmbedding(embedding,&confidence,&name);
					if(db_end_idx < DB_LENGTH){
						for (int e = 0; e < EMBEDDING_LENGTH; e++) {
							db_embeddings[db_end_idx][e] = (int16_t)embedding[e];
						}
						db_nameStr[db_end_idx] = (char*)malloc(TRACK_NAME_LENGTH);
						sprintf(db_nameStr[db_end_idx], "FACE_%03d", face_count);

						if(fix16_to_int(100*confidence)>40) printf("Warning: Similar embedding already exists (%s)\n\n",name);
						printf("Enter the id of the captured face (default: %s)\n", db_nameStr[db_end_idx]);
						id_check = 1;
						db_end_idx++;
					}
				}
				
			} else {
				// Match detected objects to tracks
				matchTracks(objects, length, recognition_model->pTracker, MAX_TRACKS, recognition_model->pTracks, tracks,use_plate);
				// Run Recognition if there is a tracked object
				if(recognition_model->pTracker->recognitionTrackInd < 0) {
					printf("Obj not tracked\n");
					detect_model->is_running = 0;
					detect_model->buf_idx = !detect_model->buf_idx;
					return 0;
				}
				
				object_t* object = recognition_model->pTracks[recognition_model->pTracker->recognitionTrackInd]->object;
				// Warp the tracked objects							
				recognizeObject(the_vbx_cnn, recognition_model, object, detect_model->post_process_type,
					screen_height, screen_width, screen_stride, screen_y_offset, screen_x_offset);
				if(use_attribute_model){
					// GENDER+AGE ATTRIBUTE 
					// get region within object bbox and see if nose keypoint within region for determining a "frontal view"
					// if frontal view, then perform genderage prediction and adjust track
					is_frontal_view = facing_front(object);
					int bbox_w = fix16_to_int(object->box[2] - object->box[0]);
					int bbox_h = fix16_to_int(object->box[3] - object->box[1]);

					// Resize detected objects to genderage input size
					resize_image_hls(SCALER_BASE_ADDRESS,
						(uint32_t*)(intptr_t)(*PROCESSING_FRAME_ADDRESS), bbox_w, bbox_h, screen_stride, fix16_to_int(object->box[0]), fix16_to_int(object->box[1]),
						(uint8_t*)virt_to_phys(the_vbx_cnn, (void*)attribute_model->model_input_buffer),
						model_get_input_shape(detect_model->model, 0)[3], model_get_input_shape(detect_model->model, 0)[2]);
				}

				// Start Recognition model
				err = vbx_cnn_model_start(the_vbx_cnn, recognition_model->model, recognition_model->model_io_buffers);
				if(err != 0) return err;

				// Update kalman filters
				updateFilters(objects, length, recognition_model->pTracker, recognition_model->pTracks, tracks);

				err = vbx_cnn_model_poll(the_vbx_cnn); // Wait for the Recognition model	
				while(err > 0) {
					for(int i =0;i<1000;i++);
					err = vbx_cnn_model_poll(the_vbx_cnn);
				}
				if(err < 0) return err;


				if(use_attribute_model) {
					// Start attribute model
					vbx_cnn_model_start(the_vbx_cnn, attribute_model->model, attribute_model->model_io_buffers);
				}

			
				if (!strcmp(recognition_model->post_process_type, "LPR")) {
					char tmp[20]="";
					name = (char*)tmp;
					int8_t* output_buffer_int8 = (int8_t*)(uintptr_t)recognition_model->model_output_buffer[0];
					confidence = post_process_lpr_int8(output_buffer_int8, recognition_model->model, name);
				} else {
					// normalize the recognition output embedding
					
					fix16_t embedding[EMBEDDING_LENGTH]={0};
					embedding_calc(embedding,recognition_model);						
					// Match the recognized objects with the ones in the database
					matchEmbedding(embedding,&confidence,&name);
				}
				// Filter recognition output
				updateRecognition(recognition_model->pTracks, recognition_model->pTracker->recognitionTrackInd, confidence, name, recognition_model->pTracker);
				if(use_attribute_model) {
					err = vbx_cnn_model_poll(the_vbx_cnn); // Wait for the attribute model
					while(err > 0) {
						for(int i =0;i<1000;i++);
						err = vbx_cnn_model_poll(the_vbx_cnn);
					}
					if(err < 0) return err;
					// Update gender+age of object tracks
					fix16_t age = 100*attribute_model->model_output_buffer[0][0];
					fix16_t gender = attribute_model->model_output_buffer[1][0];
					updateAttribution(recognition_model->pTracks[recognition_model->pTracker->recognitionTrackInd],gender,age,recognition_model->pTracker, is_frontal_view);
				}

				// Draw boxes for detected and label the recognized
				for(int t = 0; t < recognition_model->pTracker->tracksLength; t++) {
					track_t* track = recognition_model->pTracks[t];
					if(track->object == NULL)
						continue;
					fix16_t box[4];
					int box_thickness=5;
					// Calculate the box coordinates of the tracked object
					boxCoordinates(box, &track->filter);
					// Compute the offsets
					int x = fix16_to_int(box[0]) + screen_x_offset;
					int y = fix16_to_int(box[1]) + screen_y_offset;
					int w = fix16_to_int(box[2]) - fix16_to_int(box[0]);
					int h = fix16_to_int(box[3]) - fix16_to_int(box[1]);
					// Adding labels to the recognized
					if(strlen(track->name) > 0 && fix16_to_int(100*track->similarity)>40) {
						label_colour_e text_color= GREEN;
						if(fix16_to_int(100*track->similarity)>60){					
							colour = GET_COLOUR(0, 250, 0, 255);
							text_color = GREEN;
						}
						else{
							colour = GET_COLOUR(250, 250, 0, 255);
							text_color = WHITE;
						}
						if (!strcmp(recognition_model->post_process_type, "LPR")) {
							snprintf(label,sizeof(label),"%s", track->name);
						} else {
							snprintf(label,sizeof(label),"%s  (%d) ", track->name, fix16_to_int(100*track->similarity));
						}
						
						draw_label(label,x,fix16_to_int(box[3])+screen_y_offset+box_thickness,overlay_draw_frame,2048,1080,text_color);
						if(use_attribute_model){
							if(track->gender > F16(0.2)){
								gender_char = 'F';
							} else if(track->gender < F16(-0.6)){
								gender_char = 'M';
							} else{
								gender_char = '?';
							}
							snprintf(label,sizeof(label),"%c %d", gender_char, fix16_to_int(track->age));
							draw_label(label,x,fix16_to_int(box[1])-32,overlay_draw_frame,2048,1080,text_color);
						}
					} else {
						colour = GET_COLOUR(250, 250, 250,255);
						if(use_attribute_model){
							if(track->gender > F16(0.2)){
								gender_char = 'F';
							} else if(track->gender < F16(-0.6)){
								gender_char = 'M';
							} else{
								gender_char = '?';
							}
							snprintf(label,sizeof(label),"%c %d", gender_char, fix16_to_int(track->age));
							draw_label(label,x,fix16_to_int(box[1])-32,overlay_draw_frame,2048,1080,WHITE);
						}
					}

					if( x > 0 &&  y > 0 && w > 0 && h > 0) {
						draw_box(x,y,w,h,box_thickness,colour,overlay_draw_frame,2048,1080);
						if (!strcmp(recognition_model->post_process_type, "LPR")) {
						} else {
							// Draw the points of detected objects
							for(int p = 0; p < 5; p++) {
								draw_rectangle(fix16_to_int(track->object->points[p][0])+screen_x_offset,
										fix16_to_int(track->object->points[p][1])+screen_y_offset, 4, 4, colour,
										overlay_draw_frame, 2048,1080);
							}
						}
					}
				}
			}
		} else {
			matchTracks(objects, length, recognition_model->pTracker, MAX_TRACKS, recognition_model->pTracks, tracks,use_plate);
			if(capture_embedding){
				printf("No valid face embeddings captured\n");
				capture_embedding = 0;
			}
		}
		detect_model->is_running = 0;
		detect_model->buf_idx = !detect_model->buf_idx;
	}
	return 0;
}

void matchEmbedding(fix16_t embedding[],fix16_t* similarity, char** name) {
	// Match the detected objects with the objects in database
	*similarity = fix16_minimum;
	for(int d = 0; d < db_end_idx; d++){
		fix16_t dotProd = 0;
		for(int n = 0; n < EMBEDDING_LENGTH; n++)
			dotProd += (embedding[n] * db_embeddings[d][n])>>16;
		if(dotProd > *similarity){
			*similarity = dotProd;
			*name = db_nameStr[d];
		}
	}
}

void recognizeObject(vbx_cnn_t* the_vbx_cnn, struct model_descr_t* model, object_t* object, const char* post_process_type, int screen_height, int screen_width, int screen_stride, int screen_y_offset, int screen_x_offset) {
	fix16_t xy[6], ref[6];

	if(!strcmp(post_process_type,"LPD")) {
		xy[0] = object->box[0];
		xy[1] = object->box[1];
		xy[2] = object->box[2];
		xy[3] = object->box[1];
		xy[4] = object->box[0];
		xy[5] = object->box[3];
	} else if(!strcmp(post_process_type,"RETINAFACE") || !strcmp(post_process_type, "SCRFD") || !strcmp(post_process_type, "YUNET")) {
		xy[0] = object->points[0][0];
		xy[1] = object->points[0][1];
		xy[2] = object->points[1][0];
		xy[3] = object->points[1][1];
		// Mean of mouth points of Retinaface
		xy[4] = (object->points[3][0] + object->points[4][0])/2;
		xy[5] = (object->points[3][1] + object->points[4][1])/2;
	} else{
		xy[0] = object->points[0][0];
		xy[1] = object->points[0][1];
		xy[2] = object->points[1][0];
		xy[3] = object->points[1][1];
		xy[4] = object->points[3][0];
		xy[5] = object->points[3][1];
	}
	if (screen_x_offset > 0) {
		xy[0] += fix16_from_int(screen_x_offset);
		xy[2] += fix16_from_int(screen_x_offset);
		xy[4] += fix16_from_int(screen_x_offset);
	}
	if (screen_y_offset > 0) {
		xy[1] += fix16_from_int(screen_y_offset);
		xy[3] += fix16_from_int(screen_y_offset);
		xy[5] += fix16_from_int(screen_y_offset);
	}
	// Model reference coordinates
	ref[0] = model->coord4[0];
	ref[1] = model->coord4[1];
	ref[2] = model->coord4[2];
	ref[3] = model->coord4[3];
	ref[4] = model->coord4[6];
	ref[5] = model->coord4[7];
	warp_image_with_points(SCALER_BASE_ADDRESS,
		WARP_BASE_ADDRESS,
		(uint32_t*)(intptr_t)(*PROCESSING_FRAME_ADDRESS),
		(uint8_t*)virt_to_phys(the_vbx_cnn, (void*)model->model_input_buffer),
		(uint8_t*)virt_to_phys(the_vbx_cnn, (void*)warp_temp_buffer),
		xy, ref,
		screen_width, screen_height, screen_stride,
		model_get_input_shape(model->model, 0)[3],model_get_input_shape(model->model, 0)[2]);
}
