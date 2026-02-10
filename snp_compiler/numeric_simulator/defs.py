import numpy as np
class qTensor:
    def __init__(self,data,scale=1.0,zero_point=0,folding_factor_x=0,folding_factor_y=0,x_slices=1):
        self.data = data
        self.scale = scale
        self.zero_point = zero_point
        self.folding_factor_x = folding_factor_x
        self.folding_factor_y = folding_factor_y
        self.x_slices = x_slices
        # This real tensor is used in cases where the actual tensor in FPGA looks different from the one we simulate. Cases:
        # 1) A tensor which is created by a concat node that also y fold on read. In real scenario the X inputs to the concat are first y folded and then concatenated
        #    But the next conv would expect first concat and then y fold. We need the "real" tensor if we want to generate a mac debug file for that conv since
        #    In order to re-arrange the tensor to be as expected by conv we manipulate the oc_processing order and mac debug use this data to generate the debug file
        #    in particular it uses the "per_ic_group_sorted_weight_activation_pairs"
        self.real_tensor_data = None 
    def get_float(self):
        float_tensor = (self.data.astype(np.int64)-self.zero_point)*self.scale #astype(np.int64) is needed so that calc is not made in uint8
        return float_tensor.astype(np.float32)
