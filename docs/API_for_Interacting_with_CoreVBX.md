# API for Interacting with CoreVectorBlox

Refer to the bottom section of this document for information on the Python API.

## C API (Hardware and Simulator)

The following table lists the hardware and simulator enum types for the C API.

### Enum Types

| Parameter             | Description                                                                                      |
|-----------------------|--------------------------------------------------------------------------------------------------|
| vbx_cnn_calc_type_e   | VBX_CNN_CALC_TYPE_UINT8,<br> VBX_CNN_CALC_TYPE_INT8, <br> VBX_CNN_CALC_TYPE_INT16, <br> VBX_CNN_CALC_TYPE_INT32, <br> VBX_CNN_CALC_TYPE_UNKNOWN |
| vbx_cnn_size_conf_e   | VBX_CNN_SIZE_CONF_V250 = 0, <br> VBX_CNN_SIZE_CONF_V500 = 1, <br> VBX_CNN_SIZE_CONF_V1000 = 2             |
| vbx_cnn_comp_conf_e   | VBX_CNN_CONF_NO_COMPRESSION = 0, <br> VBX_CNN_CONF_COMPRESSION = 1, <br> VBX_CNN_CONF_UNSTRUCTURED_COMPRESSION = 2 |
| vbx_cnn_err_e         | START_NOT_CLEAR = 3,<br> OUTPUT_VALID_NOT_SET = 4,<br> INVALID_NETWORK_ADDRESS = 6, <br> MODEL_BLOB_INVALID = 7, <br> MODEL_BLOB_VERSION_MISMATCH = 8, <br> MODEL_BLOB_CONFIGURATION_MISMATCH = 9 |
| vbx_cnn_state_e       | READY = 1, <br> RUNNING = 2, <br> RUNNING_READY = 3, <br> FULL = 6, <br> ERROR = 8 <br>                                  |

---

## Function Documentation

### int model_check_sanity

```c
int model_check_sanity(const model_t * model)
```

Checks if the model parameter looks like a valid model.

**Parameters**

- `model`: The model to check

**Returns**

Non-zero if the model does not look valid.

### size_t model_get_allocate_bytes

```c
size_t model_get_allocate_bytes(const model_t * model)
```

Returns the size required to store the entire model, including temporary buffers. The temporary buffers must be contiguous with the model data.

**Parameters**

- `model`: The model to query

**Returns**

The size required to store the entire model in memory.

### size_t model_get_data_bytes

```c
size_t model_get_data_bytes(const model_t * model)
```

Returns the size required to store only the data part of the model.

**Parameters**

- `model`: The model to query

**Returns**

The size required to store the data part of the model.

### vbx_cnn_calc_type_e model_get_input_datatype

```c
vbx_cnn_calc_type_e model_get_input_datatype(const model_t * model, int input_index)
```

Returns the datatype of an input buffer.

**Parameters**

- `model`: The model to query
- `input_index`: The index of the input to get the datatype for.

**Returns**

The data type of the specified model input.

### size_t model_get_input_dims

```c
size_t model_get_input_dims(const model_t* model, int input_index)
```

Returns the number of dimensions of an input buffer.

**Parameters**

- `model`: The model to query
- `input_index`: The index of the input to get the number of dimensions for.

**Returns**

The number of dimensions of the specified input buffer.

### size_t model_get_input_length

```c
size_t model_get_input_length(const model_t * model, int input_index)
```

Returns the length in elements of an input buffer.

**Parameters**

- `model`: The model to query
- `input_index`: The index of the input to get the length for.

**Returns**

The length in elements of the specified input buffer.

### size_t model_get_num_inputs

```c
size_t model_get_num_inputs(const model_t * model)
```

Returns the number of input buffers for the model.

**Parameters**

- `model`: The model to query

**Returns**

The number of inputs for the model.

### size_t model_get_num_outputs

```c
size_t model_get_num_outputs(const model_t * model)
```

Returns the number of output buffers for the model.

**Parameters**

- `model`: The model to query

**Returns**

The number of outputs for the model.

### vbx_cnn_calc_type_e model_get_output_datatype

```c
vbx_cnn_calc_type_e model_get_output_datatype(const model_t * model, int output_index)
```

Returns the datatype of an output buffer.

**Parameters**

- `model`: The model to query
- `output_index`: The index of the output to get the datatype for.

**Returns**

The data type of the specified model output.

### size_t model_get_output_dims

```c
size_t model_get_output_dims(const model_t* model, int index)
```

Returns the number of dimensions of an output buffer.

**Parameters**

- `model`: The model to query
- `index`: The index of the output to get the number of dimensions for.

**Returns**

The number of dimensions of the specified output buffer.

### int* model_get_output_shape

```c
int* model_get_output_shape(const model_t* model, int index)
```

Returns the tensor dimensions of a specific output buffer.

**Parameters**

- `model`: The model to query
- `index`: The index of the output to get the shape for.

**Returns**

The tensor dimensions of the specified output buffer.

### int* model_get_input_shape

```c
int* model_get_input_shape(const model_t* model, int index)
```

Returns the tensor dimensions of a specific input buffer.

**Parameters**

- `model`: The model to query
- `input_index`: The index of the input to get the shape for.

**Returns**

The tensor dimensions of the specified input buffer.

### size_t model_get_output_length

```c
size_t model_get_output_length(const model_t * model, int index)
```

Returns the length in elements of an output buffer.

**Parameters**

- `model`: The model to query
- `output_index`: The index of the output to get the length for.

**Returns**

The length in elements of the specified output buffer.

### float model_get_output_scale_value

```c
float model_get_output_scale_value(const model_t * model, int index)
```

Returns the scale factor by which output values must be multiplied to obtain the true values.

**Parameters**

- `model`: The model to query
- `output_index`: The output for which to get the scale value.

**Returns**

The scale value for the specified model output.

### int model_get_output_scale_fix16_value

```c
int model_get_output_scale_fix16_value(const model_t * model, int index)
```

Returns the scale factor by which output values must be multiplied to obtain the true values, in fix16 format.

**Parameters**

- `model`: The model to query
- `index`: The output for which to get the scale value.

**Returns**

The scale value for the specified model output in fix16 format.

### int model_get_input_scale_fix16_value

```c
int model_get_input_scale_fix16_value(const model_t * model, int index)
```

Returns the scale factor by which input values must be multiplied to obtain the true values, in fix16 format.

**Parameters**

- `model`: The model to query
- `index`: The input for which to get the scale value.

**Returns**

The scale value for the specified model input in fix16 format.

### int model_get_output_zeropoint

```c
int model_get_output_zeropoint(const model_t * model, int index)
```

Returns the zero point offset that must be applied to output values to obtain the true values.

**Parameters**

- `model`: The model to query
- `index`: The output for which to get the zero point value.

**Returns**

The integer zero point for the specified model output.

### int model_get_input_zeropoint

```c
int model_get_input_zeropoint(const model_t * model, int index)
```

Returns the zero point offset that must be applied to input values to obtain the true values.

**Parameters**

- `model`: The model to query
- `index`: The input for which to get the zero point value.

**Returns**

The integer zero point for the specified model input.

### vbx_cnn_size_conf_e model_get_size_conf

```c
vbx_cnn_size_conf_e model_get_size_conf(const model_t * model)
```

Returns the size configuration that the model was compiled for. It will be one of the following:

- 0 = V250
- 1 = V500
- 2 = V1000
- 3 = V2000
- 4 = V4000

**Parameters**

- `model`: The model to query

**Returns**

The size configuration the model was compiled for.

### vbx_cnn_comp_conf_e model_get_comp_conf

```c
vbx_cnn_comp_conf_e model_get_comp_conf(const model_t* model)
```

Returns the compression configuration that the model was compiled for. It will be one of the following:

- 0 = No Compression
- 1 = Compression
- 2 = Unstructured Compression

**Parameters**

- `model`: The model to query

**Returns**

The compression configuration the model was compiled for.

### void* model_get_test_input

```c
void* model_get_test_input(const model_t * model, int index)
```

Returns a pointer to test input data to run through the graph for an input buffer.

**Parameters**

- `model`: The model to query
- `input_index`: The input from which to get the test input.

**Returns**

Pointer to test input of model.

### vbx_cnn_err_e vbx_cnn_get_error_val

```c
vbx_cnn_err_e vbx_cnn_get_error_val(vbx_cnn_t * vbx_cnn)
```

Reads the error register and returns the error.

**Parameters**

- `vbx_cnn`: The `vbx_cnn` object to use

**Returns**

The current value of the error register.

### vbx_cnn_state_e vbx_cnn_get_state

```c
vbx_cnn_state_e vbx_cnn_get_state(vbx_cnn_t * vbx_cnn)
```

Queries `vbx_cnn` to see if there is a model running.

**Parameters**

- `vbx_cnn`: The `vbx_cnn` object to use

**Returns**

Current state of the core. It can be one of the following:

- READY = 1 (Can accept model immediately)
- RUNNING = 2 (Model running, can accept model eventually)
- RUNNING_READY = 3 (Model running, can accept model immediately)
- FULL = 6 (Cannot accept model)
- ERROR = 8 (IP Core stopped)

### vbx_cnn_t* vbx_cnn_init

```c
vbx_cnn_t* vbx_cnn_init(void* ctrl_reg_addr)
```

Initializes the `vbx_cnn` IP Core. After this, the core accepts instructions. If any other function in this file runs without a valid initialized `vbx_cnn_t`, the result is undefined.

**Parameters**

- `ctrl_reg_addr`: The address of the `VBX CNN S_control` port.

**Returns**

A `vbx_cnn_t` structure. `.initialized` is set on success, otherwise it is zero.

### int vbx_cnn_model_poll

```c
int vbx_cnn_model_poll(vbx_cnn_t * vbx_cnn)
```

Polls for model completion.

**Parameters**

- `vbx_cnn`: The `vbx_cnn` object to use

**Returns**

The return value can be one of the following:

- 1 = Network is running
- 0 = Network is done
- -1 = Error during network processing
- -2 = No network is running

### int vbx_cnn_model_start

```c
int vbx_cnn_model_start(vbx_cnn_t * vbx_cnn, model_t * model, vbx_cnn_io_ptr_t io_buffers[])
```

Runs the model with the I/O buffers specified. One model can be queued while another model is running to achieve peak throughput. In that case, the calling code looks like:

```cpp
vbx_cnn_model_start(vbx_cnn, model, io_buffers);
while (input = get_input()) {
    io_buffers[0] = input;
    vbx_cnn_model_start(vbx_cnn, model, io_buffers);
    while (vbx_cnn_model_poll(vbx_cnn) > 0);
}
while (vbx_cnn_model_poll(vbx_cnn) > 0);
```

**Parameters**

- `vbx_cnn`: The `vbx_cnn` object to use
- `model`: The model to run
- `io_buffers`: Array of pointers to the input and output buffers. The pointers to the input buffers are first, followed by the pointers to the output buffers.

**Returns**

Non-zero if model does not run. This occurs if `vbx_cnn_get_state()` returns FULL or ERROR.

### int vbx_cnn_model_wfi

```c
int vbx_cnn_model_wfi(vbx_cnn_t * vbx_cnn)
```

Waits for interrupt to determine model completion.

**Parameters**

- `vbx_cnn`: The `vbx_cnn` object to use

**Returns**

The return value can be one of the following:

- 1 = Network is running
- 0 = Network is done
- -1 = Error during network processing
- -2 = No network is running

### void vbx_cnn_model_isr

```c
void vbx_cnn_model_isr(vbx_cnn_t * vbx_cnn)
```

Interrupt service routine for the CoreVectorBlox IP Core.

**Parameters**

- `vbx_cnn`: The `vbx_cnn` object to use

**Returns**

If a model is running, it clears the `output_valid` register once the model is done.

### int model_check_configuration

```c
int model_check_configuration(const model_t* model, vbx_cnn_t *vbx_cnn)
```

Checks if the model was generated with the correct VectorBlox Core configuration.

**Parameters**

- `model`: The model to check

**Returns**

Non-zero if the model does not look valid.

### int vbx_tsnp_model_start

```c
int vbx_tsnp_model_start(vbx_cnn_t* vbx_cnn, model_t* model, model_t* tsnp_model, uint32_t input_offset, vbx_cnn_io_ptr_t io_buffers[])
```

Runs the model specified with the I/O buffers specified. One model can be queued while another model is running to achieve peak throughput. In that case, the calling code looks like:

```cpp
vbx_tsnp_model_start(vbx_cnn, model, tsnp_model, input_offset, io_buffers);
while (input = get_input()) {
    io_buffers[0] = input;
    vbx_cnn_model_start(vbx_cnn, model, io_buffers);
    while (vbx_cnn_model_poll(vbx_cnn) > 0);
}
while (vbx_cnn_model_poll(vbx_cnn) > 0);
```

**Parameters**

- `vbx_cnn`: The `vbx_cnn` object to use
- `model`: The model
- `tsnp_model`: The TSNP version of the model
- `input_offset`: The input offset from the model base

**Returns**

Non-zero if model does not run. This occurs if `vbx_cnn_get_state()` returns FULL or ERROR.

---

## Python API (Simulator Only)

### vbx.sim.Model

#### Methods

- **\_\_init\_\_(self, model_bytes)**: Creates the model object from the bytes object passed into the method.
- **run(self, inputs)**: Runs the model with inputs passed as a list of numpy arrays. Returns a list of numpy arrays as output.

#### Attributes

- **num_outputs**: The number of outputs of this model.
- **num_inputs**: The number of inputs of this model.
- **output_lengths**: A list of lengths in number of elements for each output buffer of the model.
- **output_dims**: A list of dimensions for each output buffer of the model.
- **input_lengths**: A list of lengths in number of elements for each input buffer of the model.
- **input_dims**: A list of dimensions for each input buffer of the model.
- **output_dtypes**: A list of `numpy.dtype` for each output buffer describing the element type.
- **input_dtypes**: A list of `numpy.dtype` for each input buffer describing the element type.
- **output_scale_factor**: A list of floats describing how to scale each output buffer of the model.
- **description**: A string that the model was generated with, describing the model.
- **test_input**: A list of inputs that can be passed into `Model.run()` as test data.
