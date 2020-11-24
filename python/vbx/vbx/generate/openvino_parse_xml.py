import xml.etree.ElementTree as ET
import numpy as np


class Node:
    def __init__(self, name, type, id):
        self.name = name
        self.type = type
        self.id = id
        self.data = None
        self.input = None
        self.output = None
        self.weights = None
        self.biases = None
        self.custom = None
        self._from = None
        self._to = None
        self.min = None
        self.max = None
        self.mean = None


    def set_edges(self, edges):
        _edges = [e for e in edges if e['to_layer'] == self.id]
        # _edges = sorted(_edges, key=lambda x: x['from_port'])
        _edges = sorted(_edges, key=lambda x: x['to_port'])
        self._from = [e['from_layer'] for e in _edges]

        _edges = [e for e in edges if e['from_layer'] == self.id]
        # _edges = sorted(_edges, key=lambda x: x['to_port'])
        _edges = sorted(_edges, key=lambda x: x['from_port'])
        self._to = [e['to_layer'] for e in _edges]


    def set_stats(self, stats):
        _stats = [s for s in stats if s['name'] == self.name]
        assert(len(_stats) <= 1)
        if len(_stats):
            self.min = _stats[0]['min']
            self.max = _stats[0]['max']
            if 'mean' in _stats[0]:
                self.mean = _stats[0]['mean']


    def set_stats_by_id(self, stats):
        _stats = [s for s in stats if s['id'] == str(self.id)]
        assert(len(_stats) <= 1)
        if len(_stats):
            self.min = _stats[0]['min']
            self.max = _stats[0]['max']
            if 'mean' in _stats[0]:
                self.mean = _stats[0]['mean']


    @staticmethod
    def _precision_to_dtype(blob):
        if 'precision' in blob:
            p = blob['precision']
        elif 'element_type' in blob:
            p = blob['element_type'].upper()
        else:
            p = "FP32"

        if p == "F32":
            return np.float32
        elif p == "F64":
            return np.float64
        elif p == "FP32":
            return np.float32
        elif p == "FP64":
            return np.float64
        elif p == "I64":
            return np.int64
        elif p == "I32":
            return np.int32
        else:
            raise RuntimeError("Unknown precision {}".format(p))

    @staticmethod
    def _element_type_to_dtype(blob):
        if 'element_type' not in blob:
            p = "FP32"
        else:
            p = blob['element_type'].upper()

        if p == "FP32":
            return np.float32
        elif p == "FP64":
            return np.float64
        elif p == "I64":
            return np.int64
        elif p == "I32":
            return np.int32
        else:
            raise RuntimeError("Unknown precision {}".format(p))

    def set_params(self, bin):
        if self.weights:
            with open(bin, 'rb') as f:
                size = int(self.weights['size'])
                offset = int(self.weights['offset'])
                f.seek(offset)
                buf = f.read(size)
                dtype = Node._precision_to_dtype(self.weights)
                count = size//dtype(0).itemsize
                self.weights['arr'] = np.frombuffer(buf, dtype=dtype, count=count, offset=0)
        if self.biases:
            with open(bin, 'rb') as f:
                size = int(self.biases['size'])
                offset = int(self.biases['offset'])
                f.seek(offset)
                buf = f.read(size)
                dtype = Node._precision_to_dtype(self.biases)
                count = size//dtype(0).itemsize
                self.biases['arr'] = np.frombuffer(buf, dtype=dtype, count=count, offset=0)
        if self.custom:
            with open(bin, 'rb') as f:
                size = int(self.custom['size'])
                offset = int(self.custom['offset'])
                f.seek(offset)
                buf = f.read(size)
                dtype = Node._precision_to_dtype(self.custom)
                count = size//dtype(0).itemsize
                self.custom['arr'] = np.frombuffer(buf, dtype=dtype, count=count, offset=0)
        if self.data:
            if 'size' in self.data and 'offset' in self.data:
                with open(bin, 'rb') as f:
                    size = int(self.data['size'])
                    offset = int(self.data['offset'])
                    f.seek(offset)
                    buf = f.read(size)
                    dtype = Node._precision_to_dtype(self.data)
                    count = size//dtype(0).itemsize
                    self.data['arr'] = np.frombuffer(buf, dtype=dtype, count=count, offset=0)

    def __str__(self):

        return "name: {}, type: {}, id: {}".format(self.name, self.type, self.id)


    def get_to(self, nodes):
        return [n.name for n in nodes if n.id in self._to]


    def get_from(self, nodes):
        return [n.name for n in nodes if n.id in self._from]


def parse_statistics(xml_stats):
    stats = []
    for stat in xml_stats:
        name, min, max = None, None, None
        for s in stat:
            if s.tag == 'name':
                name = s.text
            elif s.tag == 'min':
                try:
                    min = np.asarray([float(x) for x in s.text.split(',')])
                except:
                    pass
            elif s.tag == 'max':
                try:
                    max = np.asarray([float(x) for x in s.text.split(',')])
                except:
                    pass

        stats.append({'name': name, 'min': min, 'max': max})

    return stats


def parse_edges(xml_edges):
    edges = []
    for edge in xml_edges:
        from_layer = int(edge.attrib['from-layer'])
        to_layer = int(edge.attrib['to-layer'])
        from_port = int(edge.attrib['from-port'])
        to_port = int(edge.attrib['to-port'])
        edges.append({'from_layer': from_layer, 'to_layer': to_layer, 'from_port': from_port, 'to_port': to_port})

    return edges


def parse_ports(group):
    ports = []
    for subgroup in group:
        if subgroup.tag == 'port':
            ports.append(tuple([int(dim.text) for dim in subgroup]))
    return ports


def parse_nodes(xml_nodes):
    nodes = []
    for node in xml_nodes:
        n = Node(node.attrib['name'], node.attrib['type'], int(node.attrib['id']))
        for group in node:
            if group.tag == 'data':
                n.data = {k: group.attrib[k] for k in group.keys()}
            elif group.tag == 'input':
                n.input = parse_ports(group)
            elif group.tag == 'output':
                n.output = parse_ports(group)
            elif group.tag == 'blobs':
                for subgroup in group:
                    if subgroup.tag == 'weights':
                        n.weights = {k: subgroup.attrib[k] for k in subgroup.keys()}
                    elif subgroup.tag == 'biases':
                        n.biases = {k: subgroup.attrib[k] for k in subgroup.keys()}
                    elif subgroup.tag == 'custom':
                        n.custom = {k: subgroup.attrib[k] for k in subgroup.keys()}
        nodes.append(n)
    return nodes


def parse_openvino_xml(xml_file):
    xml_net = ET.parse(xml_file).getroot()
    version = xml_net.attrib['version']

    for group in xml_net:
        if group.tag == 'layers':
            nodes = parse_nodes(group)

    for group in xml_net:
        if group.tag == 'statistics':
            stats = parse_statistics(group)
            for n in nodes:
                n.set_stats(stats)

    for group in xml_net:
        if group.tag == 'edges':
            edges = parse_edges(group)
            for n in nodes:
                n.set_edges(edges)

    bin_file = xml_file.replace('.xml', '.bin')
    for n in nodes:
        n.set_params(bin_file)

    return sorted(nodes, key=lambda x: x.id), version


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('xml')
    args = parser.parse_args()

    nodes, version = parse_openvino_xml(args.xml)
