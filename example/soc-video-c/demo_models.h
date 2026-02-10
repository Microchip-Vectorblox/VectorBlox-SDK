#ifndef __DEMO_MODELS_H
#define __DEMO_MODELS_H

#if UCOMP
struct model_descr_t models[] = {
	{"Yolov5n_70s", "/home/root/samples_V1000_UCOMP_3.0/yolov5n_70s_512x512.ucomp", 0, "YOLOV5"},
	{"Yolov8n_50s_25p", "/home/root/samples_V1000_UCOMP_3.0/yolov8n_50s_25p_512x512.ucomp", 0, "ULTRALYTICS"},
	{"Yolov8n_50s_25p_aspect", "/home/root/samples_V1000_UCOMP_3.0/yolov8n_50s_25p_512x288.ucomp", 0, "ULTRALYTICS"},
	{"Yolov9s_70s_15p", "/home/root/samples_V1000_UCOMP_3.0/yolov9s_70s_15p_512x512.ucomp", 0, "ULTRALYTICS"},
	{"Yolov8n_70s_15p_aspect", "/home/root/samples_V1000_UCOMP_3.0/yolov9s_70s_15p_512x288.ucomp", 0, "ULTRALYTICS"},
	{"Yolov8n_pose_50s_25p", "/home/root/samples_V1000_UCOMP_3.0/yolov8n_pose_50s_25p_512x288_split.ucomp", 0, "ULTRALYTICS_POSE"},
	{"Resnet18_86s_07p", "/home/root/samples_V1000_UCOMP_3.0/resnet18_86s_07p.ucomp", 0, "CLASSIFY"},
};
#elif COMP
struct model_descr_t models[] = {
	{"Yolo v8n Compressed 66", "/home/root/samples_V1000_COMP_3.0/yolov8n_comp66_V1000_comp.vnnx", 0, "ULTRALYTICS"},		
	{"Yolo v8s Compressed 68", "/home/root/samples_V1000_COMP_3.0/yolov8s_comp68_V1000_comp.vnnx", 0, "ULTRALYTICS"},	
};

#else //NCOMP
struct model_descr_t models[] = {		
	{"Yolo v8n", "/home/root/samples_V1000_NCOMP_3.0/yolov8n_512x288_argmax_V1000_ncomp.vnnx", 0, "ULTRALYTICS"},		
	{"SCRFD", "/home/root/samples_V1000_NCOMP_3.0/scrfd_500m_bnkps_V1000_ncomp.vnnx", 0, "SCRFD"},
	{"mobileface-arcface", "/home/root/samples_V1000_NCOMP_3.0/mobilefacenet-arcface_V1000_ncomp.vnnx", 0, "ARCFACE"},
	{"MobileNet V2", "/home/root/samples_V1000_NCOMP_3.0/mobilenet-v2_V1000_ncomp.vnnx", 0, "CLASSIFY"},	
	{"Yolov8 Pose", "/home/root/samples_V1000_NCOMP_3.0/yolov8n-pose_512x288_split_V1000_ncomp.vnnx", 0, "ULTRALYTICS_POSE"},	
	{"FFNet 122NS", "/home/root/samples_V1000_NCOMP_3.0/FFNet-122NS-LowRes_512x288_V1000_ncomp.vnnx", 0, "PIXEL"},	
	{"Midas V2", "/home/root/samples_V1000_NCOMP_3.0/Midas-V2-Quantized_V1000_ncomp.vnnx", 0, "PIXEL"},		
};
#endif

#endif