#include "vbx_cnn_api.h"
#include "graph_version.h"
#include <stdio.h>


static inline vnnx_layer_t* get_sublayers(const vnnx_graph_t* graph,const vnnx_subgraph_node_t* node){
	return (vnnx_layer_t*)((uintptr_t)graph+(uintptr_t)node->sublayers);
}

static inline vnnx_tensor_t* get_tensors(const vnnx_graph_t* graph,const vnnx_subgraph_node_t* node){
	return (vnnx_tensor_t*)((uintptr_t)graph+(uintptr_t)node->tensors);
}

static const vnnx_tensor_t* get_input_tensor(const vnnx_graph_t* graph, int index){
	int32_t* io_nodes = (int32_t*)(intptr_t)((intptr_t)graph +graph->io_nodes);
	int32_t* io_offset = (int32_t*)(intptr_t)((intptr_t)graph +graph->io_offsets);
	if ((unsigned)index > graph->num_inputs){
		return NULL;
	}
	const vnnx_subgraph_node_t* node = graph->subgraphs + io_nodes[index];
	vnnx_tensor_t* tensors = get_tensors(graph, node);
	return tensors + io_offset[index];
}

static const vnnx_tensor_t* get_output_tensor(const vnnx_graph_t* graph, int index){
	int32_t* io_nodes = (int32_t*)(intptr_t)((intptr_t)graph +graph->io_nodes);
	int32_t* io_offset = (int32_t*)(intptr_t)((intptr_t)graph +graph->io_offsets);
	if ((unsigned)index > graph->num_outputs){
		return NULL;
	}
	const vnnx_subgraph_node_t* node = graph->subgraphs + io_nodes[graph->num_inputs + index];
	vnnx_tensor_t* tensors = get_tensors(graph, node);
	return tensors + io_offset[graph->num_inputs + index];
}

int model_check_sanity(const model_t* model){
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	if (graph->magic != 0x1ABE11ED){
		return -1;
	}
	if (graph->version != VNNX_GRAPH_VERSION){
		return -1;
	}

	return 0;

}

size_t model_get_data_bytes(const model_t* model)
{
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	return graph->data_bytes;

}

size_t model_get_allocate_bytes(const model_t* model)
{
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	return graph->allocate_bytes;

}


size_t model_get_num_inputs(const model_t* model)
{
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	return graph->num_inputs;

}

size_t model_get_num_outputs(const model_t* model)
{
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	return graph->num_outputs;
}

vbx_cnn_size_conf_e model_get_size_conf(const model_t* model)
{
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	return graph->vbx_nn_preset;
}


vbx_cnn_calc_type_e model_get_input_datatype(const model_t* model,int index){
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	const vnnx_tensor_t* tensor = get_input_tensor(graph, index);
	if (tensor) {
		return (calc_type_e)tensor->type;
	}
	return CALC_TYPE_UNKNOWN;
}

vbx_cnn_calc_type_e model_get_output_datatype(const model_t* model,int index){
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	const vnnx_tensor_t* tensor = get_output_tensor(graph, index);
	if (tensor) {
		return (calc_type_e)tensor->type;
	}
	return CALC_TYPE_UNKNOWN;
}

size_t model_get_input_length(const model_t* model,int index){
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	const vnnx_tensor_t* tensor = get_input_tensor(graph, index);
	if (tensor) {
		int dims = tensor->dims;
		const int32_t *shape = tensor->shape;
		int size = 1;
		for (int i = 0; i < dims; i++) {
			if (shape[i] > 0) size *= shape[i];
		}
		return size;
	} 
	return -1;
}

size_t model_get_output_length(const model_t* model,int index){
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	const vnnx_tensor_t* tensor = get_output_tensor(graph, index);
	if (tensor) {
		int dims = tensor->dims;
		const int32_t *shape = tensor->shape;
		int size = 1;
		for (int i = 0; i < dims; i++) {
			if (shape[i] > 0) size *= shape[i];
		}
		return size;
	}
	return -1;
}

size_t model_get_input_dims(const model_t* model,int index){
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	const vnnx_tensor_t* tensor = get_input_tensor(graph, index);
	if (tensor) {
		return tensor->dims;
	}
	return -1;
}

size_t model_get_output_dims(const model_t* model,int index){
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	const vnnx_tensor_t* tensor = get_output_tensor(graph, index);
	if (tensor) {
		return tensor->dims;
	}
	return -1;
}

int* model_get_input_shape(const model_t* model,int index){
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	const vnnx_tensor_t* tensor = get_input_tensor(graph, index);
	if (tensor) {
		return (int*)(tensor->shape);
	}
	return NULL;
}

int* model_get_output_shape(const model_t* model,int index){
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	const vnnx_tensor_t* tensor = get_output_tensor(graph, index);
	if (tensor) {
		return (int*)(tensor->shape);
	}
	return NULL;
}

void* model_get_test_input(const model_t* model,int index){
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	const vnnx_tensor_t* tensor = get_input_tensor(graph, index);
	return (void*)((uintptr_t)graph+(uintptr_t)(tensor->direct));
}

void* model_get_test_output(const model_t* model,int index){
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	const vnnx_tensor_t* tensor = get_output_tensor(graph, index);
	return (void*)((uintptr_t)graph+(uintptr_t)(tensor->direct));
}

float model_get_output_scale_value(const model_t* model,int index){
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	const vnnx_tensor_t* tensor = get_output_tensor(graph, index);
	return tensor->scale;
}

int model_get_output_scale_fix16_value(const model_t* model,int index){
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	const vnnx_tensor_t* tensor = get_output_tensor(graph, index);
	return tensor->scale_f16;
}

float model_get_input_scale_value(const model_t* model,int index){
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	const vnnx_tensor_t* tensor = get_input_tensor(graph, index);
	return tensor->scale;
}

int model_get_input_scale_fix16_value(const model_t* model,int index){
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	const vnnx_tensor_t* tensor = get_input_tensor(graph, index);
	return tensor->scale_f16;
}

int model_get_output_zeropoint(const model_t* model,int index){
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	const vnnx_tensor_t* tensor = get_output_tensor(graph, index);
	return (int)tensor->zero;
}

int model_get_input_zeropoint(const model_t* model,int index){
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	const vnnx_tensor_t* tensor = get_input_tensor(graph, index);
	return (int)tensor->zero;
}

int model_get_debug_json(const model_t* model) {
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	vnnx_subgraph_node_t* nodes = graph->subgraphs;
	char layer_str[256];
	for (int n = 0; n < (int)graph->num_layers; n++) {
		vnnx_subgraph_node_t* node = nodes + n;
		FILE *fp;
		sprintf(layer_str, "node%03d.json", n);
		fp = fopen(layer_str, "w");
		fprintf(fp,"{");
		fprintf(fp,"\"inputs\":[\n");
		for (int i = 0; i < node->num_inputs; i++) {
			if (i == 0) fprintf(fp,"{");
			vnnx_tensor_t* pinput_tensor = ((vnnx_tensor_t*)((uintptr_t)graph+(uintptr_t)node->tensors)) + i;
			int32_t* shape = pinput_tensor->shape;
			int dims = pinput_tensor->dims;
			fprintf(fp,"\"zero\":%d,", (int)pinput_tensor->zero);
			fprintf(fp,"\"scale\":%d,", (int)pinput_tensor->scale_f16);
			fprintf(fp,"\"shape\":[");

			int size = 1;
			for (int d = 0; d < dims; d++) {
				if (shape[d] > 0) size *= shape[d];
				if(d == dims-1){
					fprintf(fp,"%d],\n",(int)shape[d]);
				} else fprintf(fp,"%d,",(int)shape[d]);
			}
			void* data =  (void*)((uintptr_t)graph+(uintptr_t)(pinput_tensor->direct));
			int8_t* d8 = (int8_t*)data;
			fprintf(fp,"\"data\":[ ");
			for (int s=0; s< size; s++){
				if(s == size-1){
					fprintf(fp,"%d]\n", d8[s]);
				} else {
					fprintf(fp,"%d,",d8[s]);
				}
			}
			if(i == node->num_inputs-1) {
				fprintf(fp,"}],\n");
			} else {
				fprintf(fp,"},\n{");
			}
		}
		fprintf(fp,"\"outputs\":[\n");
		for (int o = 0; o < node->num_outputs; o++) {
			if (o == 0) fprintf(fp,"{");

			int output_offset = node->num_tensors - node->num_outputs + o;
			if (node->num_sublayers > 0) {
				vnnx_layer_t* sublayers = get_sublayers(graph,node);
				output_offset = node->num_tensors - sublayers[node->num_sublayers - 1].num_outputs + o;
			}
			vnnx_tensor_t* poutput_tensor = ((vnnx_tensor_t*)((uintptr_t)graph+(uintptr_t)node->tensors)) + output_offset;
			int32_t* shape = poutput_tensor->shape;
			int dims = poutput_tensor->dims;
			fprintf(fp,"\"zero\":%d,", (int)poutput_tensor->zero);
			fprintf(fp,"\"scale\":%d,", (int)poutput_tensor->scale_f16);
			fprintf(fp,"\"shape\":[");

			int size = 1;
			for (int d = 0; d < dims; d++) {
				if (shape[d] > 0) size *= shape[d];
				if(d == dims-1){
					fprintf(fp,"%d],\n",(int)shape[d]);
				} else fprintf(fp,"%d,",(int)shape[d]);
			}
			void* data =  (void*)((uintptr_t)graph+(uintptr_t)(poutput_tensor->direct));
			int8_t* d8 = (int8_t*)data;
			fprintf(fp,"\"data\":[ ");
			for (int s=0; s< size; s++){
				if(s == size-1){
					fprintf(fp,"%d]\n", d8[s]);
				} else {
					fprintf(fp,"%d,",d8[s]);
				}
			}
			if(o == node->num_outputs-1) {
				fprintf(fp,"}]\n");
			} else {
				fprintf(fp,"},\n{");
			}
		}
		fprintf(fp,"}\n");
		fclose(fp);
	}
	return 0;
}
