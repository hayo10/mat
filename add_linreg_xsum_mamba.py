'''
Compute and save the linearly-regressed matrices approximating
a layer's representation given another layer's representation
'''
import torch
import utils.pickle as pck
import utils.torch as trc

from _aux_mamba import concat_iter, linreg, file, verbose
from utils.mamba_util import (create_mamba_model, \
                              extract_vector_for_a_random_position,
                              create_tensor_batch,
                               num_of_mamba_layers, reorg_vectors_for_src)

# from deconstructed_mamba import DeconstructedMamba, Rep

#############################

if __name__ != '__main__':
    raise RuntimeError('This script is not intended to be imported')

#############################

model_name = "state-spaces/mamba-130m-hf"
dataset = 'xsum'
batch_size = 8
final_device = 'gpu'
only_to_final = False

#############################

tokenized = pck.load(file('experiment', model_name.split('/')[-1],
                          dataset + '_tokenized_train', 'pickle'))
tokenized_sentences = tokenized['tokenized_sentences']
token_positions = tokenized['token_positions']

device = trc.get_cuda_if_possible()
if final_device == 'gpu':
    final_device = device
model = create_mamba_model(model_name, output_hidden_states=True)
# model = DeconstructedMamba(model_name, dataset)
# model._no_ln_f = True
model.to(device)
model.eval()

num_layers =  num_of_mamba_layers(model)
#
# v = Rep(tokenized_sentences,
#         token_positions=token_positions,
#         device=device)

verbose('finished loading')


def jump_params(layer, jump_mode):
    params = {'mode': 'jump',
              'jump_layer': layer
              }
    if isinstance(jump_mode, str):
        params |= {'jump_mode': jump_mode}
    else:
        params |= {'jump_mode': jump_mode[0]}
        params |= jump_mode[1]
    return params


instruction_list = []
for i in range(num_layers + 1):
    instruction_list.append((jump_params(i, 'stop'), 'save output'))

my_vectors = []
for batch_start in range(0, len(tokenized_sentences), batch_size):
    if batch_start % 10 == 0:
        batch_num = batch_start // batch_size
        num_batches = len(tokenized_sentences) // batch_size
        print(f'Batch num: {batch_num} out of {num_batches}')
    batch_end = batch_start + batch_size
    batch_ip_token_ids = tokenized_sentences[batch_start: batch_end]
    batch_random_positions = token_positions[batch_start: batch_end].tolist()
    batch_ip_token_ids = create_tensor_batch(batch_ip_token_ids).to(
        model.device)
    my_vectors += extract_vector_for_a_random_position(model,
                                                       batch_ip_token_ids,
                                                       batch_random_positions)
    del batch_ip_token_ids
    torch.cuda.empty_cache()

vectors = reorg_vectors_for_src(my_vectors,
                                 num_layers)
del my_vectors
torch.cuda.empty_cache()
x = 1
#
# vectors = []
# for layer, output in\
#     enumerate(
#         model.forward_detailed_bh(v,
#                                   what_to_return=['output'],
#                                   final_device=final_device,
#                                   batch_size=batch_size,
#                                   instruction_list=instruction_list)
#         ):
#     verbose(f'working on layer {layer} out of { num_layers}')
#     vectors.append(concat_iter(otpt['output'] for otpt in output))

linreg_list = []
if only_to_final:
    linreg_list += [(i,  num_layers) for i in range( num_layers)]
else:
    # this is for r2 score figure
    for i in range( num_layers + 1):
        for j in range(i + 1,  num_layers + 1):
            linreg_list.append((i, j))

verbose('starting linear regression computation and save')
for (i, j) in linreg_list:
    linreg(vectors[i].to(model.device), vectors[j].to(model.device),
           intercept=False,
           file_name=file('linreg', model_name.split('/')[-1], dataset,
                          str(i) + '_' + str(j), 'pickle'))

# for (i, j) in linreg_list:
#     linreg(vectors[i].v(), vectors[j].v(),
#            intercept=False,
#            file_name=file('linreg', model_name, dataset,
#                           str(i) + '_' + str(j), 'pickle'))
