import sys

from ptflops.pytorch_engine import add_flops_counting_methods, print_model_with_flops
from ptflops.utils import flops_to_string, params_to_string

from RESULTSONOurAPPROACH.network.segmentator_3d_asymm_spconv import Asymm_3d_spconv
import torch
from utils.load_save_util import load_checkpoint
model=Asymm_3d_spconv([480,360,32])
from torchstat import stat
from ptflops import *


def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num


def get_model_complexity_info(model, input_res,
                              print_per_layer_stat=True,
                              as_strings=True,
                              input_constructor=None, ost=sys.stdout,
                              verbose=False, ignore_modules=[]):
    assert type(input_res) is tuple
    assert len(input_res) >= 2
    flops_model = add_flops_counting_methods(model)
    flops_model.eval()
    flops_model.start_flops_count(ost=ost, verbose=verbose, ignore_list=ignore_modules)
    if input_constructor:
        input = input_constructor(input_res)
        _ = flops_model(**input)
    else:
        try:
            batch = torch.ones(()).new_empty((1, *input_res),
                                             dtype=next(flops_model.parameters()).dtype,
                                             device=next(flops_model.parameters()).device)
        except StopIteration:
            batch = torch.ones(()).new_empty((1, *input_res))

        _ = flops_model(batch)

    flops_count, params_count = flops_model.compute_average_flops_cost()
    if print_per_layer_stat:
        print_model_with_flops(flops_model, flops_count, params_count, ost=ost)
    flops_model.stop_flops_count()

    if as_strings:
        return flops_to_string(flops_count), params_to_string(params_count)

    return flops_count, params_count
PATH = "/media/ibrahim/v1/OurApproachPCUrban/RESULTSONOurAPPROACH/model_save.pt"
model = load_checkpoint(PATH, model).cuda(0)

print("Parameters",get_model_parameters_number(model))
checkpoint = torch.load(PATH)

size_model = 0
for param in checkpoint.values():
    if param.is_floating_point():
        size_model += param.numel() * torch.finfo(param.dtype).bits
    else:
        size_model += param.numel() * torch.iinfo(param.dtype).bits
print(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")
#
# from torchsummary import summary
# help(summary)
#
# summary(model,(1,100000,3))
# print(get_model_complexity_info(model,(480,360,32)))
# checkpoint = torch.load(PATH)
# print(list(checkpoint.keys()))
# model.load_state_dict(checkpoint)
# print(model)
# print(stat(model(), (480, 360,  32)))

# model.load_state_dict(checkpoint['model_state_dict'])


# model = TheModelClass(*args, **kwargs)
# optimizer = TheOptimizerClass(*args, **kwargs)
#
# checkpoint = torch.load(PATH)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
#
# model.eval()