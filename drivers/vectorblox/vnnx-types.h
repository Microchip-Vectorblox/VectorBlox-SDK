/*!
 * \file
 * \brief Data structure definitions for VectorBlox ONNX
 */

#ifndef VNNX_TYPES_H
#define VNNX_TYPES_H
#include <stdlib.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t obj_off_t;
#if defined(__GNUC__)
#define STRUCT_PACKED struct __attribute__((packed,aligned(4)))
#else
#define STRUCT_PACKED struct
#pragma pack(push,1)
#pragma warning(disable : 4200)
#endif

#define SHAPE_DIMS 6

typedef enum {
	CALC_TYPE_UINT8,
	CALC_TYPE_INT8,
	CALC_TYPE_INT16,
	CALC_TYPE_INT32,
	CALC_TYPE_UNKNOWN
} calc_type_e;

typedef enum{
	IDENTITY=300,
	ELTWISE=301,
	PREFETCH=302,
	LUT=303,
	PIXEL_SHUFFLE=304,
	UNKNOWN_SUBGRAPH=305
} VNNXOperator;

typedef enum{
    RESIZE_NEAREST,
    RESIZE_LINEAR
} resize_mode_e;


typedef enum{
	ELTWISE_ADD,
	ELTWISE_MUL,
	ELTWISE_SUB,
	ELTWISE_DIV,
	ELTWISE_GREATER,
	ELTWISE_GREATER_EQUAL,
	ELTWISE_LESS,
	ELTWISE_LESS_EQUAL,
	ELTWISE_EQUAL,
	ELTWISE_NOT_EQUAL,
	ELTWISE_MINIMUM,
	ELTWISE_MAXIMUM,
	ELTWISE_SQUARED_DIFFERENCE,
} eltwise_type_e;


typedef enum {
    ADD = 0,
    AVERAGE_POOL_2D = 1,
    CONCATENATION = 2,
    CONV_2D = 3,
    DEPTHWISE_CONV_2D = 4,
    DEPTH_TO_SPACE = 5,
    DEQUANTIZE = 6,
    EMBEDDING_LOOKUP = 7,
    FLOOR = 8,
    FULLY_CONNECTED = 9,
    HASHTABLE_LOOKUP = 10,
    L2_NORMALIZATION = 11,
    L2_POOL_2D = 12,
    LOCAL_RESPONSE_NORMALIZATION = 13,
    LOGISTIC = 14,
    LSH_PROJECTION = 15,
    LSTM = 16,
    MAX_POOL_2D = 17,
    MUL = 18,
    RELU = 19,
    RELU_N1_TO_1 = 20,
    RELU6 = 21,
    RESHAPE = 22,
    RESIZE_BILINEAR = 23,
    RNN = 24,
    SOFTMAX = 25,
    SPACE_TO_DEPTH = 26,
    SVDF = 27,
    TANH = 28,
    CONCAT_EMBEDDINGS = 29,
    SKIP_GRAM = 30,
    CALL = 31,
    CUSTOM = 32,
    EMBEDDING_LOOKUP_SPARSE = 33,
    PAD = 34,
    UNIDIRECTIONAL_SEQUENCE_RNN = 35,
    GATHER = 36,
    BATCH_TO_SPACE_ND = 37,
    SPACE_TO_BATCH_ND = 38,
    TRANSPOSE = 39,
    MEAN = 40,
    SUB = 41,
    DIV = 42,
    SQUEEZE = 43,
    UNIDIRECTIONAL_SEQUENCE_LSTM = 44,
    STRIDED_SLICE = 45,
    BIDIRECTIONAL_SEQUENCE_RNN = 46,
    EXP = 47,
    TOPK_V2 = 48,
    SPLIT = 49,
    LOG_SOFTMAX = 50,
    DELEGATE = 51,
    BIDIRECTIONAL_SEQUENCE_LSTM = 52,
    CAST = 53,
    PRELU = 54,
    MAXIMUM = 55,
    ARG_MAX = 56,
    MINIMUM = 57,
    LESS = 58,
    NEG = 59,
    PADV2 = 60,
    GREATER = 61,
    GREATER_EQUAL = 62,
    LESS_EQUAL = 63,
    SELECT = 64,
    SLICE = 65,
    SIN = 66,
    TRANSPOSE_CONV = 67,
    SPARSE_TO_DENSE = 68,
    TILE = 69,
    EXPAND_DIMS = 70,
    EQUAL = 71,
    NOT_EQUAL = 72,
    LOG = 73,
    SUM = 74,
    SQRT = 75,
    RSQRT = 76,
    SHAPE = 77,
    POW = 78,
    ARG_MIN = 79,
    FAKE_QUANT = 80,
    REDUCE_PROD = 81,
    REDUCE_MAX = 82,
    PACK = 83,
    LOGICAL_OR = 84,
    ONE_HOT = 85,
    LOGICAL_AND = 86,
    LOGICAL_NOT = 87,
    UNPACK = 88,
    REDUCE_MIN = 89,
    FLOOR_DIV = 90,
    REDUCE_ANY = 91,
    SQUARE = 92,
    ZEROS_LIKE = 93,
    FILL = 94,
    FLOOR_MOD = 95,
    RANGE = 96,
    RESIZE_NEAREST_NEIGHBOR = 97,
    LEAKY_RELU = 98,
    SQUARED_DIFFERENCE = 99,
    MIRROR_PAD = 100,
    ABS = 101,
    SPLIT_V = 102,
    UNIQUE = 103,
    CEIL = 104,
    REVERSE_V2 = 105,
    ADD_N = 106,
    GATHER_ND = 107,
    COS = 108,
    WHERE = 109,
    RANK = 110,
    ELU = 111,
    REVERSE_SEQUENCE = 112,
    MATRIX_DIAG = 113,
    QUANTIZE = 114,
    MATRIX_SET_DIAG = 115,
    ROUND = 116,
    HARD_SWISH = 117,
    IF = 118,
    WHILE = 119,
    NON_MAX_SUPPRESSION_V4 = 120,
    NON_MAX_SUPPRESSION_V5 = 121,
    SCATTER_ND = 122,
    SELECT_V2 = 123,
    DENSIFY = 124,
    SEGMENT_SUM = 125,
    BATCH_MATMUL = 126,
    PLACEHOLDER_FOR_GREATER_OP_CODES = 127,
    CUMSUM = 128,
    CALL_ONCE = 129,
    BROADCAST_TO = 130,
    RFFT2D = 131,
    CONV_3D = 132,
    IMAG=133,
    REAL=134,
    COMPLEX_ABS=135,
    HASHTABLE = 136,
    HASHTABLE_FIND = 137,
    HASHTABLE_IMPORT = 138,
    HASHTABLE_SIZE = 139,
    REDUCE_ALL = 140,
    CONV_3D_TRANSPOSE = 141,
    VAR_HANDLE = 142,
    READ_VARIABLE = 143,
    ASSIGN_VARIABLE = 144,
    BROADCAST_ARGS = 145,
    RANDOM_STANDARD_NORMAL = 146,
    BUCKETIZE = 147,
    RANDOM_UNIFORM = 148,
    MULTINOMIAL = 149,
    GELU = 150,
    DYNAMIC_UPDATE_SLICE = 151,
    RELU_0_TO_1 = 152,
    UNSORTED_SEGMENT_PROD = 153,
    UNSORTED_SEGMENT_MAX = 154,
    UNSORTED_SEGMENT_SUM = 155,
    ATAN2 = 156,
    UNSORTED_SEGMENT_MIN = 157,
    SIGN = 158,
    BITCAST = 159,
    BITWISE_XOR = 160,
    RIGHT_SHIFT = 161,
    STABLEHLO_LOGISTIC = 162, // WARNING: Do not have runtime support
    STABLEHLO_ADD = 163, // WARNING: No runtime support yet
    STABLEHLO_DIVIDE = 164, // WARNING: No runtime support yet
    STABLEHLO_MULTIPLY = 165, // WARNING: No runtime support yet
    STABLEHLO_MAXIMUM = 166, // WARNING: No runtime support yet
    STABLEHLO_RESHAPE = 167, // WARNING: No runtime support yet
    STABLEHLO_CLAMP = 168, // WARNING: No runtime support
    STABLEHLO_CONCATENATE = 169, // WARNING: No runtime support
    STABLEHLO_BROADCAST_IN_DIM = 170, // WARNING: No runtime support
    STABLEHLO_CONVOLUTION = 171, // WARNING: No runtime support
    STABLEHLO_SLICE = 172, // WARNING: No runtime support
    STABLEHLO_CUSTOM_CALL = 173, // WARNING: No runtime support
    STABLEHLO_REDUCE = 174, // WARNING: No runtime support
    STABLEHLO_ABS = 175, // WARNING: No runtime support
    STABLEHLO_AND = 176, // WARNING: No runtime support
    STABLEHLO_COSINE = 177, // WARNING: No runtime support
    STABLEHLO_EXPONENTIAL = 178, // WARNING: No runtime support
    STABLEHLO_FLOOR = 179, // WARNING: No runtime support
    STABLEHLO_LOG = 180, // WARNING: No runtime support
    STABLEHLO_MINIMUM = 181, // WARNING: No runtime support
    STABLEHLO_NEGATE = 182, // WARNING: No runtime support
    STABLEHLO_OR = 183, // WARNING: No runtime support
    STABLEHLO_POWER = 184, // WARNING: No runtime support
    STABLEHLO_REMAINDER = 185, // WARNING: No runtime support
    STABLEHLO_RSQRT = 186, // WARNING: No runtime support
    STABLEHLO_SELECT = 187, // WARNING: No runtime support
    STABLEHLO_SUBTRACT = 188, // WARNING: No runtime support
    STABLEHLO_TANH = 189, // WARNING: No runtime support
    STABLEHLO_SCATTER = 190,
    STABLEHLO_COMPARE = 191, // WARNING: No runtime support
    STABLEHLO_CONVERT = 192, // WARNING: No runtime support
    STABLEHLO_DYNAMIC_SLICE = 193, // WARNING: No runtime support
    STABLEHLO_DYNAMIC_UPDATE_SLICE = 194, // WARNING: No runtime support
    STABLEHLO_PAD = 195, // WARNING: No runtime support
    STABLEHLO_IOTA = 196, // WARNING: No runtime support
    STABLEHLO_DOT_GENERAL = 197, // WARNING: No runtime support
    STABLEHLO_REDUCE_WINDOW = 198, // WARNING: No runtime support
    STABLEHLO_SORT = 199, // WARNING: No runtime support
    STABLEHLO_WHILE = 200, // WARNING: No runtime support
    STABLEHLO_GATHER = 201, // WARNING: No runtime support
    STABLEHLO_TRANSPOSE = 202, // WARNING: No runtime support
    DILATE = 203,
    STABLEHLO_RNG_BIT_GENERATOR = 204,
    REDUCE_WINDOW = 205
} BuiltinOperator;


typedef STRUCT_PACKED {
	uint32_t base_ptr_offset_from_sp_start;
	uint32_t offset;
} indirect_ptr_t;

/**
 * @brief Describes tensor used by nodes
 */
typedef STRUCT_PACKED {
	int32_t type;
	int32_t shape[SHAPE_DIMS];
	int32_t dims;
	int32_t external_producer;
	int32_t external_consumer;
	float scale;
	int32_t scale_f16;
	int32_t zero;
	int32_t multiplier;
	int32_t shift;
	indirect_ptr_t indirect;
	obj_off_t direct;
} vnnx_tensor_t;


/**
 * @brief Parameter to minor mode function
 */
typedef struct {
	/* int maps,r,col,n,m,c,channels,y,x; */
	int c,maps,channels;
	int y,r,m; // y=row coord in tiled matrix, r=num of row in a tile, m=rows in the input
	int x,col,n; // x=col coord in tiled matrix, col=num of cols in a tile, m=cols in the input
	int b,batch,batches;
	int32_t idx[SHAPE_DIMS];
	int32_t step[SHAPE_DIMS];
	int32_t total[SHAPE_DIMS];
	int32_t dims;
	void* sp_in;
	void* sp_out;
	void* sp_prefetch;
	int32_t sp_bytes;
	int32_t sp_prefetch_bytes_per_map;
	int64_t graph_pointer;
} vnnx_layer_info_t;

struct vnnx_layer;
typedef int (*layer_run_func)(const struct vnnx_layer*,
                               vnnx_layer_info_t* /*inout*/);
/**
 * @brief Describes minor node in graph
 */

typedef STRUCT_PACKED vnnx_layer{
	int32_t type; //operator enum
	int32_t input_data_type;
	int32_t output_data_type;
	int32_t strides[2];
	int32_t kernel_shape[2];
	int32_t dilations[2];
	int32_t pads[6];
	int32_t maps;
	int32_t num_inputs;
	int32_t num_outputs;
	int32_t num_tensors;
	int32_t activation_min;
	int32_t activation_max;
	obj_off_t input_multiplier;
	obj_off_t input_shift;
	int32_t input_offset;
	obj_off_t output_multiplier;
	obj_off_t output_shift;
	int32_t output_offset;
	int32_t nop;
	obj_off_t tensors;
	union{
		STRUCT_PACKED {
			int32_t kernels;
			int32_t stride_width;
			int32_t stride_height;
			int32_t dilation_width_factor;
			int32_t dilation_height_factor;
			int32_t padding_width;
			int32_t padding_height;
			int32_t filter_shape_dims[4];
			int32_t group;
			int32_t imaps;
			int32_t conv_rows;
			int32_t use_vector;
			int32_t use_fia;
			int32_t use_db;
			int32_t use_depthwise;
			int32_t use_strided;
			int32_t fit_weights;
			int32_t split_weight_shaper_buffers;
			int32_t direct_dma;
			int32_t mxp_double_buffer;
			obj_off_t filter_data;
			obj_off_t bias_data;
			obj_off_t quantization_records;
		} Conv2DOptions;
		STRUCT_PACKED {
			int32_t axis;
		} ConcatOptions;
		STRUCT_PACKED {
			obj_off_t input2_multiplier;
			obj_off_t input2_shift;
			int32_t input2_offset;
			obj_off_t bias_data;
			int32_t swap;
			int32_t optimized;
			int32_t isize;
			int32_t left_shift;
			int32_t type;
		} eltwise8;
		STRUCT_PACKED {
			int32_t filter_shape_dims[4];
			obj_off_t filter_multiplier;
			obj_off_t filter_shift;
			int32_t filter_offset;
			obj_off_t filter_data;
			obj_off_t bias_data;
			float iscale;
			float fscale;
			float oscale;
			int32_t broadcast;
			int32_t optimized;
			int32_t isize;
			int32_t left_shift;
			int32_t sub;
			int32_t swap_inputs;
		} broadcast8;
		STRUCT_PACKED {
			int32_t axis;
			int32_t arg_max;
			obj_off_t axis_list;
		} reduce8;
		STRUCT_PACKED{ 
			int32_t value;
			int32_t transpose_dilate_w;
			int32_t transpose_dilate_h;
		} PadOptions;
		STRUCT_PACKED{
			float min;
			float max;
		}clip;
		STRUCT_PACKED{
			int32_t unsigned_input;
			int32_t unsigned_output;
			obj_off_t weights;
		}depthwise;
		STRUCT_PACKED{
			obj_off_t alpha_multiplier;
			obj_off_t alpha_shift;
			int32_t alpha_offset;
			int32_t optimized;
			int32_t maps_at_once;
			float iscale;
			float ascale;
			float oscale;
			obj_off_t vci_int8;
			obj_off_t alpha_data;
			int32_t alpha_shape[4];
		}prelu;
		STRUCT_PACKED{
			obj_off_t alpha_multiplier;
			obj_off_t alpha_shift;
		}leakyrelu;
		STRUCT_PACKED{
			int32_t diff_min;
			int32_t axis;
			int32_t depth;
			int32_t count;
			obj_off_t vci_int8;
			obj_off_t lut_int32;
			obj_off_t idx_int8;
		}SoftmaxOptions;
		STRUCT_PACKED{
			int32_t input_range_radius;
			int32_t count;
			int32_t lut_count;
			obj_off_t vci_int8;
			obj_off_t lut_int8;
			obj_off_t idx_int8;
		}ActivationOptions;
		STRUCT_PACKED{
			int32_t diff_min;
			int32_t outer_size;
			int32_t depth;
			int32_t reverse_scaling_divisor;
			int32_t reverse_scaling_right_shift;
			int32_t axis;
		}LogSoftmaxOptions;
		STRUCT_PACKED{
			obj_off_t memory_offset;
		}prefetch;
		STRUCT_PACKED{
			int32_t begin[4];
			int32_t end[4];
			int32_t stride[4];
		}SliceOptions;
		STRUCT_PACKED{
			int32_t axis;
			int32_t batch_dims;
			obj_off_t coord_data;
			int32_t batch_size;
			int32_t outer_size;
			int32_t axis_size;
			int32_t inner_size;
			int32_t coord_size;
			int32_t swap_input_order;
		}GatherOptions;
		STRUCT_PACKED{
			obj_off_t block_shape_data;
			obj_off_t paddings_data;
		}SpaceToBatchNDOptions;
		STRUCT_PACKED{
			obj_off_t block_shape_data;
			obj_off_t crop_data;
		}BatchToSpaceNDOptions;
		STRUCT_PACKED{
			int32_t mode;
		}MirrorPadOptions;
		STRUCT_PACKED{
			int32_t mode;
		}ReshapeOptions;
		STRUCT_PACKED{
			int32_t r;
		}PixelShuffleOptions;
		STRUCT_PACKED{
			int32_t axis;
			int32_t count;
			int32_t dims;
		}PackOptions;
		STRUCT_PACKED{
			float scale[2];
            int32_t mode;
            int32_t num_c_inc;
            obj_off_t c_inc;
		}ResizeOptions;
		STRUCT_PACKED{
			int32_t permutation[3];
                        int32_t out_maps_at_once;
                        int32_t out_rows_at_once;
		}TransposeOptions;
		STRUCT_PACKED{
			int32_t axis;
			obj_off_t splits;
		}SplitOptions;
		STRUCT_PACKED{
			int32_t colar_map_dims[2];
			obj_off_t colar_map_data;
		}embedding;
		STRUCT_PACKED{
			int32_t max;
			int32_t filter_shape_dims[4];
            obj_off_t filter_multiplier;
            obj_off_t filter_shift;
            int32_t filter_offset;
            obj_off_t filter_data;
		}MinMaxOptions;
		
	};
} vnnx_layer_t;

struct vnnx_subgraph_node;
struct vnnx_graph;
typedef int (*subgraph_run_func)(const struct vnnx_graph* g,struct vnnx_subgraph_node*, const int n, const int cores, const int core_start, const int core_stop);

/**
 * @brief Describes major node in graph
 */
typedef STRUCT_PACKED vnnx_subgraph_node {
	int32_t type;
	int32_t input_data_type;
	int32_t output_data_type;
	int32_t offloaded;
	int32_t input_strides[2];
	int32_t output_strides[2];
	int32_t channels;
	int32_t m;
	int32_t n;
	int32_t maps;
	int32_t rows;
	int32_t cols;
	int32_t skip;
	int32_t scratchpad_bytes;

	char input_description[48];
	char output_description[48];
	obj_off_t sublayers;
	int32_t num_sublayers;
	int32_t row_start;
	int32_t row_last;
	int32_t row_inc;
	int32_t row_inc0;
	int32_t rows_0;
	int32_t rows_final;
	int32_t col_start;
	int32_t col_last;
	int32_t col_inc;
	int32_t col_inc0;
	int32_t cols_0;
	int32_t cols_final;
	int32_t prefetch_bytes_per_map;
	int32_t use_replay;
	obj_off_t replay_buffer;
	int32_t replay_buffer_size;
	int32_t num_inputs;
	int32_t num_outputs;
	int32_t num_tensors;
	obj_off_t tensors;
	int32_t activation_min;
	int32_t activation_max;
	obj_off_t input_multiplier;
	obj_off_t input_shift;
	int32_t input_offset;
	obj_off_t output_multiplier;
	obj_off_t output_shift;
	int32_t output_offset;
	union {
		STRUCT_PACKED {
			int32_t kernels;
			int32_t stride_width;
			int32_t stride_height;
			int32_t dilation_width_factor;
			int32_t dilation_height_factor;
			int32_t padding_width;
			int32_t padding_height;
			int32_t filter_shape_dims[4];
			int32_t group;
			int32_t imaps;
			int32_t conv_rows;
			int32_t use_vector;
			int32_t use_fia;
			int32_t use_db;
			int32_t use_depthwise;
			int32_t use_strided;
			int32_t fit_weights;
			int32_t split_weight_shaper_buffers;
			int32_t direct_dma;
			int32_t mxp_double_buffer;
			int32_t first_fia;
			int32_t last_fia;
			int32_t fia_collision;
			obj_off_t filter_data;
			obj_off_t bias_data;
			obj_off_t quantization_records;
		} Conv2DOptions;
		STRUCT_PACKED {
			obj_off_t input2_multiplier;
			obj_off_t input2_shift;
			int32_t input2_offset;
			obj_off_t bias_data;
			int32_t swap;
			int32_t optimized;
			int32_t isize;
			int32_t left_shift;
			int32_t type;
		} eltwise8;
		STRUCT_PACKED {
			int32_t filter_shape_dims[2];
			int32_t input_stride;
			int32_t use_fia;
			int32_t first_fia;
			int32_t last_fia;
			int32_t fia_collision;
			int32_t mxp_double_buffer;
			obj_off_t filter_data;
			obj_off_t bias_data;
			obj_off_t quantization_records;
		} FullyConnectedOptions;
		STRUCT_PACKED {
			int32_t axis;
		} ConcatOptions;
		STRUCT_PACKED{
			int32_t axis;
			int32_t count;
			int32_t dims;
		}PackOptions;
		STRUCT_PACKED {
			int32_t pixels_per_loop;
		} argmax;
		STRUCT_PACKED{
			float alpha;
			float beta;
			float bias;
			float scale;
			int32_t size;
		}lrn;
		STRUCT_PACKED{
			int32_t tile[4];
		}TileOptions;
		STRUCT_PACKED{
			int32_t m0;
		}reduce;
		STRUCT_PACKED{
			int32_t stride;
		}reorg;
		STRUCT_PACKED{
			float scale[2];
            int32_t mode;
            int32_t num_c_inc;
            obj_off_t c_inc;
		}ResizeOptions;
		STRUCT_PACKED{
			int32_t permutation[3];
                        int32_t out_maps_at_once;
                        int32_t out_rows_at_once;
		}TransposeOptions;
		STRUCT_PACKED{
			int32_t axis;
			obj_off_t splits;
		}SplitOptions;
	};
} vnnx_subgraph_node_t;

/**
 * @brief Description of communication interface between client and host
 */

typedef STRUCT_PACKED {
	uint8_t is_host;

	uint32_t max_message_length;
  volatile uint8_t *send_base;
  volatile uint8_t *send_end;
  volatile uint32_t *send_write_offset_ptr;
  volatile uint32_t *send_read_offset_ptr;
  volatile uint8_t *recv_base;
  volatile uint8_t *recv_end;
  volatile uint32_t *recv_write_offset_ptr;
  volatile uint32_t *recv_read_offset_ptr;
} vbx_sm_comm_interface;

/**
 * @brief Description of core
 */
typedef struct{
	int is_host;
	vbx_sm_comm_interface comm;
}vnnx_core_t;


struct vnnx_shared_allocator;
/**
 * @defgroup vnnx Vectorblox ONNX Library.
 * @{
 */

/**
 * @brief Structure that describes how to allocate shared memory
 */
typedef struct vnnx_shared_allocator{
	void* (*alloc)(struct vnnx_shared_allocator*,size_t num_bytes);
	void (*free)(struct vnnx_shared_allocator*,void* ptr);
} vnnx_shared_allocator_t;

/**
 *  @brief External facing graph object
 */
typedef STRUCT_PACKED vnnx_graph{
	uint32_t version; 
	uint32_t vbx_nn_preset;
	uint32_t num_inputs;
	uint32_t num_outputs;
	/*
	 * 16 bytes in (4*word)
	((uint32_t *)fixed_replay_buffer)[0] = (((0) << (0)) | (((1 << 17) | (1 << 18))));
        ((uint32_t *)fixed_replay_buffer)[1] = (uint32_t)(((uintptr_t)model_replay_buffer)-((uintptr_t)model_address));
        ((uint32_t *)fixed_replay_buffer)[2] = (uint32_t)((((uintptr_t)model_replay_buffer_mxp_end)-16)-((uintptr_t)model_address));
         ((uint32_t *)fixed_replay_buffer)[3] = 0;
	*/
	uint32_t fixed_replay_buffer0; 
	uint32_t fixed_replay_buffer1; 
	uint32_t fixed_replay_buffer2; 
	uint32_t fixed_replay_buffer3; 

	uint32_t include_io_data;
	uint32_t data_bytes;
	uint32_t allocate_bytes;
	//after this the attributes are private
	obj_off_t io_nodes;
	obj_off_t io_offsets;
	int32_t num_layers;
	obj_off_t replay_buffer;
	int32_t replay_buffer_size;
	uint32_t magic;
	vnnx_subgraph_node_t subgraphs[0];
}vnnx_graph_t;

#if defined(__GNUC__)
#else
#pragma pack(pop)
#endif

typedef enum {UNINITIALIZED, INVALID, VALID} recording_status_e;

/**
 * @brief Function pointer for running user kernels
 */

typedef int (*vnnx_user_kernel)(uint32_t core,
                                uint32_t cores,
                                uint32_t message_length,
                                uint8_t* message_buffer);

/**
 @}*/

#ifdef __cplusplus
}
#endif
	
#endif //VNNX_TYPES_H
