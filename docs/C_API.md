# C API (Hardware and Simulator)

The following table lists the hardware and simulator enum types for C API

### Enum Types 

| Parameter             | Description                                                                                      |
|-----------------------|--------------------------------------------------------------------------------------------------|
| vbx_cnn_calc_type_e   | VBX_CNN_CALC_TYPE_UINT8,<br> VBX_CNN_CALC_TYPE_INT8, <br> VBX_CNN_CALC_TYPE_INT16, <br> VBX_CNN_CALC_TYPE_INT32, <br> VBX_CNN_CALC_TYPE_UNKNOWN |
| vbx_cnn_size_conf_e   | VBX_CNN_SIZE_CONF_V250 = 0, <br> VBX_CNN_SIZE_CONF_V500 = 1, <br> VBX_CNN_SIZE_CONF_V1000 = 2             |
| vbx_cnn_comp_conf_e   | VBX_CNN_CONF_NO_COMPRESSION = 0, <br> VBX_CNN_CONF_COMPRESSION = 1, <br> VBX_CNN_CONF_UNSTRUCTURED_COMPRESSION = 2 |
| vbx_cnn_err_e         | START_NOT_CLEAR = 3,<br> OUTPUT_VALID_NOT_SET = 4,<br> INVALID_NETWORK_ADDRESS = 6, <br> MODEL_BLOB_INVALID = 7, <br> MODEL_BLOB_VERSION_MISMATCH = 8, <br> MODEL_BLOB_CONFIGURATION_MISMATCH = 9 |
| vbx_cnn_state_e       | READY = 1, <br> RUNNING = 2, <br> RUNNING_READY = 3, <br> FULL = 6, <br> ERROR = 8 <br>                                  |

---

# Function Documentation


## int model_check_sanity

```int model_check_sanity(const model_t * model)```

Model parsing function checks if the model parameter looks like a valid model.

**Parameters**<br>
```model```: The model to check

**Returns**<br>
It returns non-zero, if the model does not look valid.

## size_t model_get_allocate_bytes

```size_t model_get_allocate_bytes(const model_t * model)```

It is used to get size required to store the entire model, include temporary buffers. The temporary buffers must be contiguous with the model data.

**Parameters**<br>
```model```: The model to query

**Returns**<br>
It returns the size required to store entire model in memory.

## size_t model_get_data_bytes

```size_t model_get_data_bytes(const model_t * model)```

It is used to get the size required to store only the data part of the model.

**Parameters**<br> 
```model```: The model to query

**Returns**<br>
It returns the size required to store the data part of the model.

## vbx_cnn_calc_type_e model_get_input_datatype

```vbx_cnn_calc_type_e model_get_input_datatype(const model_t * model, int input_index)```

It is used to get the datatype of an input buffer.

**Parameters**<br>
The following are the parameters:
- ```model```: The model to query
- ```input_index```: The index of the input to get the datatype.

**Returns**<br>
It returns the data type of model input.

## size_t model_get_input_dims

```size_t model_get_input_dims(const model_t* model, int input_index)```

It is used to get the dimensions of elements of an input buffer.

**Parameters**<br> 
The following is the list of parameters:
- ```model```: The model to query
- ```input_index```: Index used to grab the total number of input dimensions.

**Returns**<br>
It returns the number of dimensions of indexed input buffer.

## size_t model_get_input_length

```size_t model_get_input_length(const model_t * model, int input_index)```

It is used to get the length in elements of an input buffer.

**Parameters**<br> 
The following is the list of parameters:
- ```model```: The model to query
- ```input_index```: The index of the input to get the length of elements.

**Returns**<br>
It returns the length in elements of input buffer.

## size_t model_get_num_inputs

```size_t model_get_num_inputs(const model_t * model)```

It is used to get the number of input buffers for the model.

**Parameters**<br> 
```model```: The model to query

**Returns**<br>
It returns the number of inputs for the model.

## size_t model_get_num_outputs

```size_t model_get_num_outputs(const model_t * model)```

It is used to get the number of output buffers for the model.

**Parameters**<br>  
```model```: The model to query

**Returns**<br>
It returns the number of outputs for the model.

## vbx_cnn_calc_type_e model_get_output_datatype

```vbx_cnn_calc_type_e model_get_output_datatype(const model_t * model, int output_index)```

It is used to get the datatype of an output buffer.

**Parameters**<br> 
The following is the list of parameters:
- ```model```: The model to query
- ```output_index```: The index of the output to get the datatype.

**Returns**<br>
It returns the data type of model output.

## size_t model_get_output_dims

```size_t model_get_output_dims(const model_t* model, int index)```

It is used to get the number of dimensions of elements of an output buffer.

**Parameters**<br> 
The following is the list of parameters:
- ```model```: The model to query
- ```input```: The index of the output to get the length of output dimension index.

**Returns**<br>
It returns the number of dimensions of indexed input buffer.

## int* model_get_output_shape

```int* model_get_output_shape(const model_t* model, int index)```

It is used to get the dimensions of elements of a specific output buffer.

**Parameters**<br> 
The following is the list of parameters:
- ```model```: The model to query
- ```input```: The index of the input to get the length of output shape index.

**Returns**<br>
It returns the dimensions in elements of output buffer

## int* model_get_input_shape

```int* model_get_input_shape(const model_t* model, int index)```

It is used to get dimensions of elements of a specific output buffer.

**Parameters**<br> 
The following is the list of parameters:
- ```model```: The model to query
- ```input_index```: The index of the input to get the length of input shape index.

**Returns**<br>
It returns the dimensions in elements of output buffer.

## size_t model_get_output_length

```size_t model_get_output_length(const model_t * model, int index)```

It is used to get the length in elements of an output buffer

**Parameters**<br> 
The following is the list of parameters:
- ```model```: The model to query
- ```output_index```: The index of the output to get the output length.

**Returns**<br>
It returns the length in elements of output buffer.

## float model_get_output_scale_value

```float model_get_output_scale_value(const model_t * model, int index)```

It is used to get the amount of output values should be scaled to get the true values.

**Parameters**<br> 
The following is the list of parameters:
- ```model```: The model to query
- ```output_index```: The output for which to get the scale value.

**Returns**<br>
It returns the scale value for model output.

## int model_get_output_scale_fix16_value

```int model_get_output_scale_fix16_value(const model_t * model, int index)```

It is used to get the amount of output values should be scaled to get the true values.

**Parameters**<br> 
The following is the list of parameters:
- ```model```: The model to query
- ```index```: The output for which to get the scale value.

**Returns**<br>
It returns the scale value for model output in fix16 format.

## int model_get_input_scale_fix16_value

```int model_get_input_scale_fix16_value(const model_t * model, int index)```

It is used to get the amount of output values should be scaled to get the true values.

**Parameters**<br> 
The following is the list of parameters:
- ```model```: The model to query
- ```index```: The input for which to get the scale value.

**Returns**<br> 
It returns the scale value for model input in fix16 format.

## int model_get_output_zeropoint

```int model_get_output_zeropoint(const model_t * model, int index)```

It is used to get the amount of output values should be offset by to get the true values

**Parameters**<br> 
The following is the list of parameters:
- ```model```: The model to query
- ```index```: The output for which to get the zero point value.

**Returns**<br>
It returns the zero point for model output in fix16 format

## int model_get_input_zeropoint

```int model_get_input_zeropoint(const model_t * model, int index)```

It is used to get the amount of input values should be offset by to get the true values.

**Parameters**<br> 
The following is the list of parameters:
- ```model```: The model to query
- ```index```: The input for which to get the scale value.

**Returns**<br> 
It returns the zero point for model input in fix16 format.

## vbx_cnn_size_conf_e model_get_size_conf

```vbx_cnn_size_conf_e model_get_size_conf(const model_t * model)```

It is used to get the size configuration that the model was generated for. It will be one of the
following:
- 0 = V250
- 1 = V500
- 2 = V1000

**Parameters**<br> 
```model```: The model to query

**Returns**<br> 
It returns the size configuration the model was generated

## vbx_cnn_comp_conf_e model_get_comp_conf

```vbx_cnn_comp_conf_e model_get_comp_conf(const model_t* model)```

It is used to get compression configuration that the model was generated for, and it will be one of:
- 0 = No Compression
- 1 = Compression
- 2 = Unstructured Compression

**Parameters**<br> 
```model```: The model to query

**Returns**<br> 
It returns the size of that the model was generated for.

## void* model_get_test_input

```void* model_get_test_input(const model_t * model, int index)```

It is used to get a pointer to test input to run through the graph for an input buffer.

**Parameters**<br> 
The following is the list of parameters:
- ```model```: The model to query
- ```input_index```: The input from which to get the test input.

**Returns**<br> 
Pointer to test input of model.

## vbx_cnn_err_e vbx_cnn_get_error_val

```vbx_cnn_err_e vbx_cnn_get_error_val(vbx_cnn_t * vbx_cnn)```

It is used to read error register and return the error.

**Parameters**<br> 
```vbx_cnn```: The ```vbx_cnn``` object to use

**Returns**<br> 
It returns the current value of error register.

## vbx_cnn_state_e vbx_cnn_get_state

```vbx_cnn_state_e vbx_cnn_get_state(vbx_cnn_t * vbx_cnn)```

It is used to query ```vbx_cnn``` to see if there is a model running

**Parameters**<br> 
```vbx_cnn```: The ```vbx_cnn``` object to use

**Returns**<br> 
Current state of the core. It can be one of the following:
- READY = 1 (Can accept model immediately)
- RUNNING = 2 (Model Running, can accept model eventually)
- RUNNING_READY = 3 (Model Running, can accept model immediately)
- FULL = 6 (Cannot accept model)
- ERROR = 8 (IP Core stopped)

## vbx_cnn_t* vbx_cnn_init

```vbx_cnn_state_e vbx_cnn_get_state(volatile void * ctrl_reg_addr)```

It is used to initialize ```vbx_cnn``` IP Core. After this, the core accepts instructions. If any other function in this file runs without a valid initialized ```vbx_cnn_t```, the result is undefined.

**Parameters**<br> 
```ctrl_reg_addr```: The address of the ```VBX CNN S_control``` port.

**Returns**<br> 
A ```vbx_cnn_t``` structure. ```.initialized``` is set on success, otherwise it is zero.

## int vbx_cnn_model_poll

```int vbx_cnn_model_poll(vbx_cnn_t * vbx_cnn)```

Waits for model to complete.

**Parameters**<br> 
```vbx_cnn```: The ```vbx_cnn``` object to use

**Returns**<br> 
The returns can be one of the following:
- 1 = If network is running, 
- 0 = If network is done, 
- −1 = If there is an error during network processing, 
- −2 = No network is running

## int vbx_cnn_model_start

```int vbx_cnn_model_start(vbx_cnn_t * vbx_cnn, model_t * model, vbx_cnn_io_ptr_t * io_buffers)```

Runs the model with the I/O buffers specified. One model can be queued while another model is running to achieve peak throughput. In that case, the calling code looks like:
```cpp
vbx_cnn_model_start(vbx_cnn,model,io_buffers);
   while (input = get_input()){
       io_buffers[0] = input;
       vbx_cnn_model_start(vbx_cnn,model,io_buffers);
       while(vbx_cnn_model_poll(vbx_cnn)>0);
   }
while(vbx_cnn_model_poll(vbx_cnn)>0);
```

**Parameters**<br> 
The following is the list of parameters:
- ```vbx_cnn```: The ```vbx_cnn``` object to use; 
- ```model```: The model to query; 
- ```io_buffers```: Array of pointers to the input and output buffers. The pointers to the input buffers are first, and are followed by the pointers to the output buffers.

**Returns**<br> 
Non zero if model does not run. It occurs if ```vbx_cnn_get_state()``` returns FULL or ERROR.

## int vbx_cnn_model_wfi

```int vbx_cnn_model_wfi(vbx_cnn_t * vbx_cnn)```

It waits for interrupt to determine model completion.

**Parameters**<br> 
```vbx_cnn```: The vbx_cnn object to use

**Returns**<br> 
The following is the list of possible returns:
- 1 = If network is running
- 0 = If network done or no network is running
- −1 = If error in processing network

## void vbx_cnn_model_isr

```void vbx_cnn_model_isr(vbx_cnn_t * vbx_cnn)```

It interrupts service register

**Parameters**<br> 
```vbx_cnn```: The ```vbx_cnn``` object to use

**Returns**<br> 
If a model is running, it clears the ```output_valid``` register once model is done.



## void model_check_configuration

```int model_check_configuration(const model_t* model, vbx_cnn_t *vbx_cnn)```

Check to see if the model is generated with right Vectorblox Core configuration

**Parameters**<br> 
```model```: The model to check

**Returns**<br> 
Non zero if the model does not look valid.




## int vbx_tsnp_model_start

```int vbx_tsnp_model_start(vbx_cnn_t* vbx_cnn, model_t* model, model_t* tsnp_model, uint32_t input_offset, vbx_cnn_io_ptr_t io_buffers[])```

Run the model specified with the IO buffers specified. One Model can be queued while another model is running to achieve peak throughput. In that case the calling code would look something like:

```cpp
vbx_tsnp_model_start(vbx_cnn,model,io_buffers);
while (input = get_input()){
   io_buffers[0] = input;
   vbx_cnn_model_start(vbx_cnn,model,io_buffers);
   while(vbx_cnn_model_poll(vbx_cnn)>0);
}
while(vbx_cnn_model_poll(vbx_cnn)>0);
```

**Parameters**<br>
The following is the list of parameters:
- ```vbx_cnn```: The ```vbx_cnn``` object to use; 
- ```model```: The model; 
- ```tsnp_model```: The tsnp version of the model; 
- ```input_offset```: The input offset from the model base; 

**Returns**<br> 
Nonzero if model not run. Occurs if vbx_cnn_get_state() returns FULL or ERROR.


---




# Python API (Simulator Only)

### vbx.sim.Model
### Methods
The following is the list of methods:
- __init__(self,model_bytes): : Creates the model object from the bytes object passed into method.
- run(self,inputs): Runs the model with inputs passed as a list of numpy arrays. It returns a list of numpy arrays as output.
### Attributes
The following is the list of attributes:
- num_outputs: The number of outputs of this model.
- num_inputs: The number of inputs of this model.
- output_lengths: A list of lengths in number of elements for each output buffer of the model.
- output_dims: A list of dimensions for each output buffer of the model.
- input_lengths: A list of lengths in number of elements for each input buffer of the model.
- input_dims: A list of dimensions for each input buffer of the model.
- output_dtypes: A list of numpy.dtype for output describing the element type of each output buffer of the model.
- input_dtypes: A list of numpy.dtype for output describing the element type of each input buffer of the model.
- output_scale_factor: A list of floats describing how to scale each output buffer of the model.
- description: A string that the model is generated with describing the model.
- test_input: A list of inputs that can be passed into Model.run() as test data.