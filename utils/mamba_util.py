import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


def create_mamba_model(model_name, output_hidden_states):
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 output_hidden_states=output_hidden_states)
    return model


def num_of_mamba_layers(model):
    return len(model.backbone.layers)


def extract_vector_for_a_random_position(model, input_ids, random_positions):
    with torch.no_grad():
        outputs = model(input_ids)  # num layers * bs * seq_len * dim
    all_layer_ops = []
    for ip_it, random_pos in enumerate(random_positions):
        selected_hidden_states = []
        for layer_num, layer_op in enumerate(outputs.hidden_states):
            selected_hidden_states.append(layer_op[ip_it][random_pos].to('cpu'))
        all_layer_ops.append(selected_hidden_states)
    del outputs
    torch.cuda.empty_cache()
    return all_layer_ops


def reorg_vectors_for_src(ip_level_vectors, num_layers):
    layer_level_list = [
        [] for _ in range(num_layers + 1)
    ]
    for ip_level_vec in ip_level_vectors:
        for layer_it, layer_op in enumerate(ip_level_vec):
            layer_level_list[layer_it].append(layer_op.unsqueeze(0))

    layer_level_tensors = []
    for ll in layer_level_list:
        layer_level_tensors.append(torch.cat(ll))
    return layer_level_tensors


def run_early_exiting(model, input_ids):
    with torch.no_grad():
        outputs = model(input_ids)  # num layers * bs * seq_len * dim
    all_layer_ops = []
    for layer_num, layer_op in enumerate(outputs.hidden_states):
        """
        (norm_f)
        (lm_head)
        """
        rms_op = model.backbone.norm_f(layer_op)
        lm_op = model.lm_head(rms_op)
        all_layer_ops.append(lm_op.to('cpu'))
    del outputs
    torch.cuda.empty_cache()
    return all_layer_ops


def create_tensor_batch(batch_ip_tensors):
    max_dim = max([len(ip) for ip in batch_ip_tensors])
    zero_tensor = torch.zeros(len(batch_ip_tensors), max_dim, dtype=torch.int64)
    for ip_it, ip_tensor in enumerate(batch_ip_tensors):
        zero_tensor[ip_it, :len(ip_tensor)] = ip_tensor
    return zero_tensor


def extract_vector_for_a_random_position_from_texts(model, tokenizer,
                                                    ip_texts, random_positions):
    input_ids_obj = tokenizer(ip_texts, return_tensors='pt').to(model.device)
    input_ids = input_ids_obj["input_ids"]
    outputs = model(input_ids)  # num layers * bs * seq_len * dim
    all_layer_ops = []
    for ip_it, random_pos in enumerate(random_positions):
        selected_hidden_states = []
        for layer_num, layer_op in enumerate(outputs):
            selected_hidden_states.append(layer_op[ip_it][random_pos])
        all_layer_ops.append(selected_hidden_states)
    return torch.tensor(all_layer_ops)
