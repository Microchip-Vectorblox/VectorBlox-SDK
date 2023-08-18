/*!
 * \file
 * \brief API for interacting with Core VectorBlox
 */

#ifndef VBX_CNN_API_H
#define VBX_CNN_API_H
#include <stdint.h>
#include <stdlib.h>
#include "vnnx-types.h"
#if defined(__riscv) && defined(__linux)
#define VBX_SOC_DRIVER 1
#endif
#ifdef __cplusplus
extern "C" {
#endif

/**
 * A Note on caches:
 * In these API functions there are several  blocks of memory that that are read and written to by
 * both the host processor and the VBX CNN IP Core. These blocks include the instruction blob,
 * the network blob, and any input or output buffers.
 * In order to reduce bugs users should also ensure that these buffers are not in the host's cachable memory.
 */
#define VBX_INSTRUCTION_SIZE (2*1024*1024)
typedef uint64_t vbx_cnn_io_ptr_t;
struct model_struct;
typedef struct model_struct model_t;


typedef enum {
	VBX_CNN_CALC_TYPE_UINT8,
	VBX_CNN_CALC_TYPE_INT8,
	VBX_CNN_CALC_TYPE_INT16,
	VBX_CNN_CALC_TYPE_INT32,
	VBX_CNN_CALC_TYPE_UNKNOWN
} vbx_cnn_calc_type_e;

typedef enum {
	VBX_CNN_SIZE_CONF_V250  = 0,
	VBX_CNN_SIZE_CONF_V500  = 1,
	VBX_CNN_SIZE_CONF_V1000 = 2,
	VBX_CNN_SIZE_CONF_V2000 = 3,
	VBX_CNN_SIZE_CONF_V4000 =  4,
}vbx_cnn_size_conf_e;

typedef enum {
              INVALID_FIRMWARE_ADDRESS        = 1,
              FIRMWARE_ADDRESS_NOT_READY      = 2,
              START_NOT_CLEAR                 = 3,
              OUTPUT_VALID_NOT_SET            = 4,
              FIRMWARE_BLOB_VERSION_MISMATCH  = 5,
              INVALID_NETWORK_ADDRESS         = 6,
              MODEL_BLOB_INVALID              = 7,
              MODEL_BLOB_VERSION_MISMATCH     = 8,
              MODEL_BLOB_SIZE_CONFIGURATION_MISMATCH      = 9,
              FIRMWARE_BLOB_STALE            = 10
}vbx_cnn_err_e;

#define MAX_IO_BUFFERS 10
typedef struct {
	int32_t initialized;
	uint32_t version;
	uint32_t size;/*vbx_cnn_size_conf_e*/
	volatile uint32_t output_valid;
	void* instruction_blob;
	volatile uint32_t* ctrl_reg;
  	int debug_print_ptr;
  	size_t  dma_phys_trans_offset;
#if defined(VBX_SOC_DRIVER) || defined(SPLASHKIT_PCIE)
    	int fd;
    	uint8_t* dma_buffer;
    	uint8_t* dma_buffer_end;
  	vbx_cnn_io_ptr_t *io_buffers;
#endif

}vbx_cnn_t;

#if VBX_SOC_DRIVER || SPLASHKIT_PCIE
  void* vbx_allocate_dma_buffer(vbx_cnn_t* vbx_cnn,size_t request_size,size_t align);
#else
  void* vbx_allocate_dma_buffer(vbx_cnn_t* vbx_cnn,size_t request_size,size_t align);
#endif
/**
 * Initialize vbx_cnn IP Core.
 * After this, the core will accept instructions.
 * If any other function in this file is run without a valid initialized vbx_cnn_t
 * the result is undefined.
 *
 * @param ctrl_reg_addr The address of the VBX CNN S_control port
 * @param firmware_blob A block of memory containing valid instructions for the IP Core
 *         Users must ensure that the instruction blob is reachable by the VBX CNN IP Core's
 *         M_AXI port
 *
 * @return A vbx_cnn_t structure. On success .initialized is set, otherwise it is zero;
 *
 */
vbx_cnn_t* vbx_cnn_init(void* ctrl_reg_addr,void* firmware_blob);

/**
 * Read error register and return the error
 *
 * @param vbx_cnn The vbx_cnn object to use
 *
 * @return Current value of Error Register
 */
vbx_cnn_err_e vbx_cnn_get_error_val(vbx_cnn_t* vbx_cnn);

/**
 * Run the model specified with the IO buffers specified
 * One Model can be queued while another model is running to achieve peak throughput.
 * In that case the calling code would look something like:
 * @code{.cpp}
 *  vbx_cnn_model_start(vbx_cnn,model,io_buffers);
 *  while (input = get_input()){
      io_buffers[0] = input;
      vbx_cnn_model_start(vbx_cnn,model,io_buffers);
      while(vbx_cnn_model_poll(vbx_cnn)>0);
   }
   while(vbx_cnn_model_poll(vbx_cnn)>0);
 * @endcode
 *
 * @param vbx_cnn The vbx_cnn object to use
 * @param model The model
 * @return nonzero if model not run. Occurs if vbx_cnn_get_state() returns FULL or ERROR
 */
int vbx_cnn_model_start(vbx_cnn_t* vbx_cnn,model_t* model,vbx_cnn_io_ptr_t io_buffers[]);

typedef enum{
             READY = 1,         //< Can accept model immediately
             RUNNING = 2,       //< Model Running,can accept model eventually
             RUNNING_READY = 3, //< Model Running,can accept model immediately
             FULL = 6,          //< Cannot accept model
             ERROR = 8          //< IP Core stopped.
}vbx_cnn_state_e;
/**
 * Query vbx_cnn to see if there is a model running
 *
 * @param vbx_cnn The vbx_cnn object to use
 * @return current state of the core. One of:
         READY =1 (Can accept model immediately)
         RUNNING = 2       (Model Running,can accept model eventually)
         RUNNING_READY = 3  (Model Running,can accept model immediately)
         FULL = 6          (Cannot accept model)
         ERROR = 8          (IP Core stopped.)

 */
vbx_cnn_state_e vbx_cnn_get_state(vbx_cnn_t* vbx_cnn);

/**
 * Wait for model to complete
 *
 * @param vbx_cnn The vbx_cnn object to use
 *
 * @return 1 if network running
 *         0 if network done
 *         -1 if error processing network
 *         -2 No Network Running
 */
int vbx_cnn_model_poll(vbx_cnn_t* vbx_cnn);

/**
 * Wait for output_valid interrupt
 *
 * @param vbx_cnn The vbx_cnn object to use
 *
 * @return 1 if network running
 *         0 if network done
 *         -1 if error processing network
 *         -2 No Network Running
 */
int vbx_cnn_model_wfi(vbx_cnn_t* vbx_cnn);

void vbx_cnn_model_isr(vbx_cnn_t *vbx_cnn);


/**
 * Model Parsing Function
 */

/**
 * Check to see if the memory pointed to by model
 * looks like a valid model.
 *
 * @param model The model to check
 * @return non zero if the model does not look valid
 */
int model_check_sanity(const model_t* model);

/**
 * Get length in elements of an output buffer
 * @param model The model to query
 * @param index The index of the output to get the length of
 * @return length in elements of output buffer
 */
size_t model_get_output_length(const model_t* model,int output_index);
/**
 * Get length in elements of an input buffer
 * @param model The model to query
 * @param index The index of the input to get the length of
 * @return length in elements of input buffer
 */

size_t model_get_input_length(const model_t* model,int input_index);

/**
 * Get dimensions of elements of an input buffer
 * @param model The model to query
 * @param index The index of the input to get the length of
 * @return dimensions of in elements of input buffer
 */
int* model_get_input_dims(const model_t* model,int input_index);

/**
 * Get dimensions of elements of an output buffer
 * @param model The model to query
 * @param index The index of the output to get the length of
 * @return dimensions of in elements of output buffer
 */
int* model_get_output_dims(const model_t* model,int output_index);

/**
 * Get the datatype of an output buffer
 * @param model The model to query
 * @param index The index of the output to get the datatype of
 * @return data type of model output
 */

vbx_cnn_calc_type_e model_get_output_datatype(const model_t* model,int output_index);
/**
 * Get the datatype of an input buffer
 * @param model The model to query
 * @param index The index of the output to get the datatype of
 * @return data type of model input
 */
vbx_cnn_calc_type_e model_get_input_datatype(const model_t* model,int input_index);
/**
 * Get the number of input buffers for the model
 * @param model The model to query
 * @return get number of inputs for the model
 */
size_t model_get_num_inputs(const model_t* model);
/**
 * Get the number of output buffers for then model
 * @param model The model to query
 * @return get number of outputs for the model
 */

size_t model_get_num_outputs(const model_t* model);

/**
 * Get size configuration that the model was generated for
 *   will be one of
 *   V250 0
 *   V500 1
 *   V1000 2
 *   V2000 3
 *   V4000 4
 *
 * @param model The model to query
 * @return sizeot that the model was generated for
 */
vbx_cnn_size_conf_e model_get_size_conf(const model_t* model);

/**
 * Get size required to store the data part of the model
 *
 * @param model The model to query
 * @param the size required to store the data part of the model
 */
size_t model_get_data_bytes(const model_t* model);
/**
 * Get size required to store the entire model, include temporary buffers
 * the temporary buffers must be contiguous with the model data
 *
 * @param model The model to query
 * @return The size required to store entire model in memory.
 */

size_t model_get_allocate_bytes(const model_t* model);

/**
 * Get a pointer to test input to run through the graph
 * for an input buffer
 *
 * @param model The model to query.
 * @param input_index The input for which to get the test input.
 * @return pointer to test input of model
 */
void* model_get_test_input(const model_t* model,int input_index);

//void* model_get_test_output(const model_t* model,int output_index);

/**
 * Get a the amount the the output values should be scaled to
 * get the true values
 *
 * @param model The model to query.
 * @param output_index The output for which to get the scale value.
 * @return scale value for model output
 */

float model_get_output_scale_value(const model_t* model,int index);

#ifdef __cplusplus
}
#endif

#endif //VBX_CNN_API_H
