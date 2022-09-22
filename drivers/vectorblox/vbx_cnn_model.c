#include "vbx_cnn_api.h"
#include "graph_version.h"
#include <stdio.h>
static const vnnx_subgraph_node_t* get_input_node(const vnnx_graph_t* graph, int index){
	int32_t* io_nodes = (int32_t*)(intptr_t)((intptr_t)graph +graph->io_nodes);
	if ((unsigned)index > graph->num_inputs){
		return NULL;
	}
	return graph->subgraphs+io_nodes[index];
}
static const vnnx_subgraph_node_t* get_output_node(const vnnx_graph_t* graph, int index){
	int32_t* io_nodes = (int32_t*)(intptr_t)((intptr_t)graph +graph->io_nodes);
	if ((unsigned)index > graph->num_outputs){
		return NULL;
	}
	return graph->subgraphs +io_nodes[graph->num_inputs+index];
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
	const vnnx_subgraph_node_t* node = get_input_node(graph,index);
	if(node){
		return (calc_type_e)node->input_data_type;
	}
	return CALC_TYPE_UNKNOWN;
}
vbx_cnn_calc_type_e model_get_output_datatype(const model_t* model,int index){
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	const vnnx_subgraph_node_t* node = get_output_node(graph,index);
	if(node){
		return node->output_data_type;
	}
	return CALC_TYPE_UNKNOWN;
}
size_t model_get_input_length(const model_t* model,int index){
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	const vnnx_subgraph_node_t* node = get_input_node(graph,index);
	return node?node->input_size:-1;
}
size_t model_get_output_length(const model_t* model,int index){
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	const vnnx_subgraph_node_t* node = get_output_node(graph,index);
	return node?node->output_size:-1;
}
int* model_get_input_dims(const model_t* model,int index){
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	const vnnx_subgraph_node_t* node = get_input_node(graph,index);
	return (int*)(node?node->input_shape:NULL);
}
int* model_get_output_dims(const model_t* model,int index){
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	const vnnx_subgraph_node_t* node = get_output_node(graph,index);
	return (int*)(node?node->output_shape:NULL);
}
void* model_get_test_input(const model_t* model,int index){
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	const vnnx_subgraph_node_t* node = get_input_node(graph,index);
	return (void*)(intptr_t)((uintptr_t) graph + node->test_input_data);
}
void* model_get_test_output(const model_t* model,int index){
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	const vnnx_subgraph_node_t* node = get_output_node(graph,index);
	return (void*)(intptr_t)((uintptr_t) graph + node->test_output_data);
}

float model_get_output_scale_value(const model_t* model,int index){
	vnnx_graph_t* graph = (vnnx_graph_t*)model;
	const vnnx_subgraph_node_t* node = get_output_node(graph,index);
	return node->output_scale_factor;
}
