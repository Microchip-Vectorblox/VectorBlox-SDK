import os
import struct
import argparse
import math

# This script combines a compiled vnnx model with a compiled nx model to generate
# a unified model file for VBX 3.0. Currently it is a standalone command. In the
# future it may run automatically as part of vnnx_compile.
#
# Example usage:
# python generate_vbx3_model.py -v path/to/test.vnnx -n path/to/nx_ddr_content.bin


# Get the size in bytes of the binary file
def get_file_size(bin_file):
    try:
        return os.path.getsize(bin_file)
    except FileNotFoundError:
        print(f"Error: The file '{bin_file}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
    assert False

# Get the offset and size for each input and output tensor
def get_io_offset_and_size(path_to_vnnx):
    inputs = []
    outputs = []
    file_path = os.path.join(os.path.dirname(path_to_vnnx), "nx_engine", "vnnx_io_offsets.txt")
    with open(file_path, 'r') as file:
        next(file)  # Skip the header
        for line in file:
            columns = line.strip().split()
            assert len(columns) == 5

            size = int(columns[3])
            offset = int(columns[4])

            io_type = columns[0]
            assert io_type in ["input", "output"]
            if io_type == "input":
                inputs.append((size, offset))
            else:
                outputs.append((size, offset))
    
    return inputs, outputs

def get_header_section(vnnx_path, vnnx_num_bytes, nx_num_bytes):
    # First get the offset and size for each input and output
    inputs, outputs = get_io_offset_and_size(vnnx_path)

    # Header size is 5 (size of 3 sections, num inputs, num outputs) + 2 (size/offset) per
    # input and output
    num_ints = 5 + 2*len(inputs) + 2*len(outputs)

    # Add zero padding so header is always a multiple of 16 bytes
    num_ints_padded = (num_ints + 3) // 4 * 4
    padding = [0] * (num_ints_padded - num_ints)

    # Get final header size
    header_size = num_ints_padded*4

    # Convert from list of tuples to flat lists
    inputs_data = []
    for size_offset_tuple in inputs:
        for value in size_offset_tuple:
            inputs_data.append(value)
    outputs_data = []
    for size_offset_tuple in outputs:
        for value in size_offset_tuple:
            outputs_data.append(value)
                        
    # Make the header
    header = struct.pack(f'{num_ints_padded}I', header_size, vnnx_num_bytes, nx_num_bytes, len(inputs),
        len(outputs), *inputs_data, *outputs_data, *padding)

    return header

def main():
    # Parse inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vnnx', type=str, required=True, help='Path to .vnnx file')
    parser.add_argument('-n', '--nx_bin', type=str, required=True, help='Path to NX .bin file')
    parser.add_argument('-o', '--ucomp', type=str, required=True, help='Path to .ucomp .bin file')
    args = parser.parse_args()

    # Get the size of each file
    vnnx_num_bytes = get_file_size(args.vnnx)
    vnnx_num_bytes_padded = math.ceil(vnnx_num_bytes/16) * 16
    nx_num_bytes = get_file_size(args.nx_bin)

    # Create a new header section
    header = get_header_section(args.vnnx, vnnx_num_bytes_padded, nx_num_bytes)

    # Open the files and read their content
    with open(args.vnnx, 'rb') as f:
        vnnx_data = f.read()
    padded_vnnx_data = vnnx_data.ljust(vnnx_num_bytes_padded, b'\0')
    with open(args.nx_bin, 'rb') as f:
        nx_data = f.read()

    # Create the new combined file
    with open(args.ucomp, 'wb') as out_file:
        out_file.write(header)
        out_file.write(padded_vnnx_data)
        out_file.write(nx_data)

    print(f"Model file '{args.ucomp}' has been created.")

if __name__ == '__main__':
    main()
