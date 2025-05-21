# Currently Supported C Postprocessing


### int decodeMultiplePoses_int8()
Post processing for posenet networks that should return number of detected poses with keypoint, score, and displacement information.

    int decodeMultiplePoses_int8(poses_t poses[],
                            int8_t *	scores,
                            int8_t *	offsets,
                            int8_t *	displacementsFwd,
                            int8_t *	displacementsBwd,
                            int	outputStride,
                            int	maxPoseDetections,
                            fix16_t	scoreThreshold,
                            int	nmsRadius,
                            fix16_t	minPoseScore,
                            int	height,
                            int	width,
                            int	zero_points[],
                            fix16_t	scale_outs[] )



    Parameters
        poses	            Array of poses found
        scores	            Output buffer from model
        offsets	            Output buffer from model
        displacementsFwd	Output buffer from model
        displacementsBwd	Output buffer from model
        outputStride	    Stride used from model
        maxPoseDetections	Max number of detected poses
        scoreThreshold	    Score threshold ranged from 0-1.0
        nmsRadius	        Non-maximal suppression radius in pixels
        minPoseScore	    Threshold for individual points ranged from 0-1.0
        height	            Height of model output
        width	            Width of model output
        zero_points	        Mean offset for int8
        scale_outs	        fix16 scaling multiplier for int8
    Returns
        int                 Number of detected poses

### post_process_classifier_int8()
Post processing for classifiers such as MNIST or ImageNet networks. Returns the topk indices of the outputs, sorted from highest to lowest.

    void post_process_classifier_int8(int8_t *	outputs,
                                    const int	output_size,
                                    int16_t *	output_index,
                                    int	topk )


    Parameters
        outputs	            Unsorted outputs obtained from the model
        output_size	        Number of outputs
        output_index	    The retrned indicies, sorted by scores lowest to highest
        topk	            Number of indices to return sorted

### post_process_scrfd_int8()
Postprocessing on detected objects that stores bounding box and keypoints within the face object.

    int post_process_scrfd_int8	(object_t	faces[],
                                int	max_faces,
                                int8_t *	network_outputs[9],
                                int	zero_points[],
                                fix16_t	scale_outs[],
                                int	image_width,
                                int	image_height,
                                fix16_t	confidence_threshold,
                                fix16_t	nms_threshold,
                                model_t *	model )
    

    Parameters
        faces	                Array of faces found, one per detection
        max_faces	            Max number of faces that can be found in a model
        network_outputs	        Output buffers obtained from VectorBlox
        zero_points	            Mean offset for int8
        scale_outs	            fix16 scaling multiplier for int8
        image_width	            Width of input image sent to the network
        image_height	        Height of input image sent to the network
        confidence_threshold	Confidence threshold, ranged from 0-1.0
        nms_threshold	        Non-max suppression for overlapping detectiongs, ranged from 0-1.0
        model	                Network model
    Returns
        int                     Number of detected objects

### post_process_ssd_torch_int8()
Post-processing for SSD V2. Returns number of detected objects, along with boxes, confidence, and class information.

    int post_process_ssd_torch_int8(fix16_box *	boxes,
                                    int	max_boxes,
                                    int8_t *	network_outputs[12],
                                    fix16_t	network_scales[12],
                                    int32_t	network_zeros[12],
                                    int	num_classes,
                                    fix16_t	confidence_threshold,
                                    fix16_t	nms_threshold )


    Parameters
        boxes	                Array of object detciong boxes, one per detection
        max_boxes	            Max number of boxes that can be found
        network_outputs	        Output buffers obtained from VectorBlox
        network_scales	        fix16 scaling multipliers for int8
        network_zeros	        Mean offsets for int8
        num_classes	            Number of classes in the model
        confidence_threshold	Confidence threshold, from 0-1.0
        nms_threshold	        Non-max suppression overlap threshold, from 0-1.0
    Returns
        int                     Number of detected objects

### post_process_ultra_int8()
Returns number of detected objects and their respective classes.

    int post_process_ultra_int8(int8_t **	outputs,
                                int *	outputs_shape[],
                                fix16_t *	post,
                                fix16_t	thresh,
                                int	zero_points[],
                                fix16_t	scale_outs[],
                                const int	max_boxes,
                                const int	is_obb,
                                const int	is_pose )


    Parameters
        outputs	            Output buffers obtained from VectorBlox
        outputs_shape	    Index of the shapes corresponding to the outputs
        post	            Boxes obtained from postprocessing
        thresh	            Condifence threshold, ranged from 0-1.0
        zero_points	        Mean offset for int8
        scale_outs	        fix16 scaling multiplier for int8
        max_boxes	        Max number of boxes
        is_obb	            Check if postprocess is done for oriented-bounding boxes
        is_pose	            Check if postprocess is done for pose detection
    Returns
        int                 Number of detected boxes
### post_process_ultra_nms()
Performs non-maximal suppression on detected objects.

    int post_process_ultra_nms(fix16_t *	output,
                                int	output_boxes,
                                int	input_h,
                                int	input_w,
                                fix16_t	thresh,
                                fix16_t	overlap,
                                fix16_box	fix16_boxes[],
                                poses_t	poses[],
                                int	boxes_len,
                                const int	num_classes,
                                const int	is_obb,
                                const int	is_pose )


    Parameters
        output	        Array of boxes obtained from postprocessing
        output_boxes	Max number of boxes for outputs
        input_h	        Input height of the model
        input_w	        Input width of the model
        thresh	        Condifence threshold, ranged from 0-1.0
        overlap	        Non-max suppression fo overlapping detections, ranging from 0-1.0
        fix16_boxes	    Boxes obtained from non-maximal suppression
        poses	        Poses obtained from non-maximal suppression if pose
        boxes_len	    Max length of boxes to be returned
        num_classes	    Number of classes for detection
        is_obb	        Check if nms is done for oriented-bounding boxes
        is_pose	        Check if nms is done for pose detection
    Returns
        int             Number of valid detections

### post_process_ultra_nms_int8()
Performs non-maximal suppression on detected objects.

    int post_process_ultra_nms_int8(int8_t *	output,
                                    int	output_boxes,
                                    int	input_h,
                                    int	input_w,
                                    fix16_t	f16_scale,
                                    int32_t	zero_point,
                                    fix16_t	thresh,
                                    fix16_t	overlap,
                                    fix16_box	fix16_boxes[],
                                    int	boxes_len,
                                    const int	num_classes )


    Parameters
        output	        Array of boxes obtained from postprocessing
        output_boxes	Max number of boxes for outputs
        input_h	        Input height of the model
        input_w	        Input width of the model
        f16_scale	    Scale value multiplierin fix16 format for int8
        zero_point	    Mean offset for int8
        thresh	        Condifence threshold, ranged from 0-1.0
        overlap	        Non-max suppression fo overlapping detections, ranging from 0-1.0
        fix16_boxes	    Boxes obtained from non-maximal suppression
        boxes_len	    Max length of boxes to be returned
        num_classes	    Number of classes for detection
    Returns
        int             Number of valid detections

### post_process_yolo_int8()
Post-processing for Yolov2/V3/V4/V5. Returns number of detected objects, along with boxes containing coordinates, confidence, and class information.

    int post_process_yolo_int8(int8_t **	outputs,
                                const int	num_outputs,
                                int	zero_points[],
                                fix16_t	scale_outs[],
                                yolo_info_t *	cfg,
                                fix16_t	thresh,
                                fix16_t	overlap,
                                fix16_box	fix16_boxes[],
                                int	max_boxes )


    Parameters
        outputs	        Output buffers obtained from VectorBlox
        num_outputs	    Number of outputs the model contains
        zero_points	    Mean offset for int8
        scale_outs	    fix16 scaling multiplier for int8
        cfg	            Configuration containing output sizes and anchors
        thresh	        Confidence threshold, ranged from 0-1.0
        overlap	        IOU overlap threshold, ranged from 0-1.0
        fix16_boxes	    Array of object detectiong boxes, one per detection
        max_boxes	    Limit to the number of possible detection boxes found
    Returns
        int             Number of detected boxes

### pprint_post_process()
Post process wrapper function.

    int pprint_post_process(const char *	name,
                            const char *	pptype,
                            model_t *	model,
                            vbx_cnn_io_ptr_t *	o_buffers,
                            int	int8_flag,
                            int	fps,
                            vbx_cnn_t *	the_vbx_cnn )


    Parameters
        name	        Model name
        pptype	        Model postprocessing type
        model	        Model artefact
        o_buffers	    Output buffers obtained from the model
        int8_flag	    Flag indicating whether outputs are int8 or not
        fps	            FPS calculation for the model (set to 1 for simulation)
        the_vbx_cnn     VectorBlox cnn information
    Returns
        int             Returns -1 if postprocessing type is invalid, otherwise returns 0 for success.