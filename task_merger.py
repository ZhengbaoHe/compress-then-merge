import math
import time
from collections import OrderedDict, defaultdict
from copy import deepcopy
from functools import partial
import sys, os
# XXX
import einops
import torch
from torch import nn
from tqdm import tqdm

from masking_ops import masked_merge
from merging_functions import ties_merging, tv_merging, topk_values_mask, PCB_merge
from utils import get_mask_fn
import tensorly as tl
from tensorly.decomposition import partial_tucker
from tensorly.tenalg import multi_mode_dot
float32_dtype = torch.float32

COMPUTE_FIRST_RANK = False
COMPUTE_AVERAGE_RANK = False

def directions_to_reps(directions):
    if isinstance(directions, list):
        return [directions_to_reps(direction) for direction in directions]
    return torch.nn.utils.parameters_to_vector([value.reshape(-1) for value in directions.values()])


class VectorOps(nn.Module):
    def directions_to_reps(self, directions):
        if isinstance(directions, list):
            return [self.directions_to_reps(direction) for direction in directions]
        return torch.nn.utils.parameters_to_vector(
            [value.reshape(-1) for key, value in directions.items()]
        )

    def rep_to_state_dict(self, vector, state_dict, remove_keys=[]):
        if isinstance(vector, list) or len(vector.shape) == 2:
            return [self.rep_to_state_dict(v, state_dict, remove_keys) for v in vector]
        # create a reference dict to define the order of the vector
        reference_dict = deepcopy(state_dict)
        for key in remove_keys:
            if key in reference_dict:
                del reference_dict[key]
        sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

        # create a shared state dict using the refence dict
        torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

        # add back the encoder and decoder embedding weights.
        if "transformer.shared.weight" in sorted_reference_dict:
            for key in remove_keys:
                sorted_reference_dict[key] = sorted_reference_dict[
                    "transformer.shared.weight"
                ]
        return sorted_reference_dict

    def mask_to_state_dict(self, mask, state_dict, remove_keys=[]):
        if isinstance(mask, list):
            return [self.mask_to_state_dict(m, state_dict, remove_keys) for m in mask]
        return self.rep_to_state_dict(mask, state_dict, remove_keys)

    def forward(self, directions, merging_fn, merge_config):
        vectors = self.directions_to_reps(directions)
        merged_vector, rows_to_keep, topk_mask = merging_fn(vectors)

        ties_mask = [dict() for _ in range(len(rows_to_keep))]
        for idx in range(len(rows_to_keep)):
            ties_mask[idx] = self.rep_to_state_dict(rows_to_keep[idx], directions[0])
        sd = self.rep_to_state_dict(merged_vector, directions[0])

        return sd, ties_mask


class TaskMerger(nn.Module):
    def __init__(self, finetuned_models, pretrained_model, param_handler, device=0, merge_config=None):
        super().__init__()
        print("=====================================")
        print(f"Device: {device}")
        print("=====================================")

        self.device = device
        self.scaling_coeffs = torch.tensor([1.] * len(finetuned_models)).to(self.device)
        self.param_handler = param_handler
        
        self.finetuned_models = []
        for ft_model in finetuned_models:
            if isinstance(ft_model, dict):
                self.finetuned_models.append({k: v.to(self.device) for k, v in ft_model.items()})
            elif hasattr(ft_model, 'to'):
                self.finetuned_models.append(ft_model.to(self.device))
            else:
                self.finetuned_models.append(ft_model)

        self.ftms_params = [param_handler(ft_model) for ft_model in self.finetuned_models]
        self.pretrained_model = pretrained_model.to(self.device)
        self.pt_params = self.pretrained_model.state_dict()
        self.merge_config = merge_config

    def randbin(self, M, N, P):
        P = 1 - P
        return torch.randint(2, size=(M, N), dtype=torch.float32).bernoulli(P)

    def apply_dare(self, ftms_params, p, dare_seed=0):
        print("DARE seed: ", dare_seed)
        torch.manual_seed(dare_seed)
        finetuned_directions = []
        for ftm_params in ftms_params:
            direction_sd = {}
            for key, finetuned_val in ftm_params.items():
                direction_sd[key] = finetuned_val * self.randbin(finetuned_val.shape[0], finetuned_val.shape[1], p) * (1 / (1 - p))
            finetuned_directions += [OrderedDict(sorted(direction_sd.items()))]
        return finetuned_directions

    def get_task_directions(self, ptm_params, ftms_params):
        finetuned_directions = []
        for ftm_params in ftms_params:
            direction_sd = {}

            for key, finetuned_val in ftm_params.items():
                if key not in ptm_params:
                    ptm_val = torch.zeros_like(finetuned_val)
                else:
                    ptm_val = ptm_params[key]
                direction_sd[key] = finetuned_val - ptm_val
            finetuned_directions += [OrderedDict(sorted(direction_sd.items()))]
        return finetuned_directions

    def set_scaling_coeffs(self, scaling_coeffs):
        if isinstance(scaling_coeffs, float) or len(scaling_coeffs) == 1:
            self.scaling_coeffs = torch.tensor([scaling_coeffs] * len(self.ftms_params)).to(self.device)
        else:
            self.scaling_coeffs = torch.tensor(scaling_coeffs).to(self.device)

    def get_layer_names(self, state_dict):
        layer_names = defaultdict(lambda: dict())
        for key in state_dict:
            if ('.weight' in key) or ('_weight' in key):
                strip_key = key.replace('.weight', '').replace('_weight', '')
                layer_names[strip_key]['weight'] = key
            elif ('.bias' in key) or ('_bias' in key):
                strip_key = key.replace('.bias', '').replace('_bias', '')
                layer_names[strip_key]['bias'] = key
            else:
                layer_names[key]['other'] = key + ':other'
        return layer_names

    def add_task_parameters(self, base_model, parameters, concat_across_output=True, scaling_coeffs=1.):
        if isinstance(parameters, list):
            return [self.add_task_parameters(
                deepcopy(base_model),
                parameter,
                concat_across_output=concat_across_output,
                scaling_coeffs=scaling_coeffs
            ) for parameter in parameters]
        sd = base_model.state_dict()
        for key, val in parameters.items():
            if any('base_layer' in k for k in sd.keys()):
                key = '.'.join(key.split('.')[:-1] + ['base_layer'] + key.split('.')[-1:])
            if concat_across_output:
                sd[key].add_(val.to(self.device) * scaling_coeffs)
            else:
                sd[key].add_(val.T.to(self.device) * scaling_coeffs)
        return base_model

    def directions_to_matrices(self, directions, reference_layer_names=None):
        if isinstance(directions, list):
            return [self.directions_to_matrices(direction, reference_layer_names) for direction in directions]

        if reference_layer_names is None:
            layer_names = self.get_layer_names(directions)
        else:
            layer_names = reference_layer_names

        matrices = {}
        for layer_name, parameter_names in layer_names.items():
            if 'other' in parameter_names:
                other_parameter = directions[parameter_names['other'].replace(':other', '')].to(torch.float32)
                # Ensure parameters are always two dimensional
                if len(other_parameter.shape) == 1:  # e.g., class token, positional embeddings
                    other_parameter = other_parameter[None, :]
                elif len(other_parameter.shape) > 2:  # e.g., patch embeddings
                    other_parameter = other_parameter.flatten(1)
                matrices[layer_name + ':other'] = other_parameter
            elif 'weight' in parameter_names:
                weight_name = parameter_names['weight']
                weight = directions[weight_name]
                if 'norm' in layer_name or 'ln' in layer_name:
                    weight = torch.diag(weight)
                matrices[layer_name] = weight.flatten(1)
                if 'bias' in parameter_names:
                    bias = directions[parameter_names['bias']]
                    matrices[layer_name] = torch.concat((matrices[layer_name], bias.reshape(-1, 1)), dim=1)
        return matrices

    def matrix_to_state_dict(self, matrix, state_dict, remove_keys=[]):
        if isinstance(matrix, list):
            return [self.matrix_to_state_dict(m, state_dict) for m in matrix]

        reference_dict = deepcopy(state_dict)
        for key in remove_keys:
            if key in reference_dict:
                del reference_dict[key]

        layer_names = self.get_layer_names(reference_dict)
        merged_state_dict = {}
        for layer_name, value in matrix.items():

            parameter_types = layer_names[layer_name.replace(':other', '')]
            if 'other' in parameter_types:
                name = parameter_types['other'].replace(':other', '')
                merged_state_dict[name] = value.reshape(reference_dict[name].shape)
            else:
                if 'bias' in parameter_types:
                    bias_index = value.shape[1] - 1
                    value, bias = value[:, :bias_index], value[:, -1].flatten()
                    merged_state_dict[parameter_types['bias']] = bias
                if 'norm' in layer_name or 'ln' in layer_name:
                    value = torch.diagonal(value)
                name = parameter_types['weight']
                merged_state_dict[name] = value.reshape(*(reference_dict[name].shape))

        # add back the encoder and decoder embedding weights.
        if "transformer.shared.weight" in merged_state_dict:
            for key in remove_keys:
                merged_state_dict[key] = merged_state_dict[
                    "transformer.shared.weight"
                ]
        return merged_state_dict

    def transform(self, *args, **kwargs):
        return


class VectorMerger(TaskMerger):
    def __init__(self, finetuned_models, pretrained_model, param_handler, device=0, merge_config=None):
        super().__init__(
            finetuned_models=finetuned_models,
            pretrained_model=pretrained_model,
            param_handler=param_handler,
            device=device,
            merge_config=merge_config
        )

        self.representation_helper = VectorOps()

    def merge(self, merge_config={'merge_method': 'tv'}):
        print(merge_config['merge_method'])

        if merge_config.get('merge_method') in ('tv', 'sum'):
            merging_fn = partial(tv_merging, **merge_config, weights=self.scaling_coeffs)
        elif merge_config.get('merge_method') in ('ties', 'dare-ties'):
            merging_fn = partial(ties_merging, **merge_config, weights=self.scaling_coeffs)
        else:
            raise ValueError(f"Merge method {merge_config['merge_method']} is not defined for VectorMerger.")

        ptm_reference_params = self.param_handler(self.pretrained_model).get_ft_parameters()

        ftms_relevant_params = (ftm.get_ft_parameters() for ftm in self.ftms_params)
        finetuned_directions = self.get_task_directions(ptm_reference_params, ftms_relevant_params)

        if merge_config.get('dare', False) or merge_config.get('merge_method') == 'dare-ties':
            finetuned_directions = self.apply_dare(
                finetuned_directions, merge_config['dare_pruning_coeffs'], merge_config['dare_seed']
            )

        merged_vector, _, _ = merging_fn(directions_to_reps(finetuned_directions))

        merged_sd = OrderedDict(sorted(finetuned_directions[0].items()))
        torch.nn.utils.vector_to_parameters(merged_vector, merged_sd.values())

        if len(merged_sd) == 2:
            merged_sd, mask = merged_sd

        t0 = time.time()
        if merge_config.get('isotropize', False):
            print("Isotropizing merged state dict...")
            for key, val in merged_sd.items():
                U, S, V = torch.linalg.svd(val.to(float32_dtype), full_matrices=False)
                S_iso_value = S.mean()
                S_iso = S_iso_value * torch.ones_like(S)
                merged_sd[key] = U @ torch.diag(S_iso) @ V

        merged_base = deepcopy(self.pretrained_model)

        merged_model = self.add_task_parameters(merged_base, merged_sd)
        print(f"Time taken to merge parameters: {time.time() - t0:.2f} seconds")

        return merged_model


class SVDMerger(TaskMerger):
    def __init__(self, finetuned_models, pretrained_model, param_handler, device=0, merge_config=None):
        super().__init__(
            finetuned_models=finetuned_models,
            pretrained_model=pretrained_model,
            param_handler=param_handler,
            device=device,
            merge_config=merge_config
        )

        self.layer_names = self.get_layer_names(self.ftms_params[0].get_ft_parameters())
        self.representation_helper = VectorOps()
        self.ingredients = None

    def variable_extend_dim(self, elements, op_dim):
        if isinstance(elements, list):
            return [self.variable_extend_dim(element, op_dim) for element in elements]
        while len(elements.shape) < (op_dim + 1):
            elements = elements.unsqueeze(-1)
        return elements

    def dict_of_concat_matrices(self, list_of_dictmatrices, dim=0, concat_across_output=True):
        dict2matrix_stack = defaultdict(lambda: list())
        for dict2matrix in list_of_dictmatrices:
            for key, val in dict2matrix.items():
                if concat_across_output:
                    dict2matrix_stack[key] += [val]
                else:
                    dict2matrix_stack[key] += [val.T.to(self.device)]

        for key, list_of_vals in dict2matrix_stack.items():
            # Extend dim as necessary
            list_of_vals = self.variable_extend_dim(list_of_vals, op_dim=dim)
            dict2matrix_stack[key] = torch.concat(list_of_vals, dim=dim)
        return dict2matrix_stack

    def reconstruct_merged_sd(self, U_sd, sV_sd):
        if isinstance(sV_sd, list):
            if isinstance(U_sd, list):
                return [self.reconstruct_merged_sd(U, sV) for U, sV in zip(U_sd, sV_sd)]
            return [self.reconstruct_merged_sd(U_sd, sV) for sV in sV_sd]
        sd = {}
        for key, U in U_sd.items():
            sd[key] = (U @ sV_sd[key]).to(torch.float32)
        return sd

    def apply_svd(self, ft_params, concat_across_output=True):
        UsV_dict = {}
        basis_dict = {}  # basis for reconstruction
        s_compositions_dict = [dict() for _ in range(len(ft_params))]
        V_compositions_dict = [dict() for _ in range(len(ft_params))]  # basis composition information per task

        print(f'Calculating SVD over {len(ft_params)} models. S > 1e-5')
        concated_ft_params = self.dict_of_concat_matrices(ft_params, dim=1, concat_across_output=concat_across_output)
        for key, val in tqdm(concated_ft_params.items(), desc='Obtaining SVDs...'):
            U, s, V = torch.linalg.svd(val.to(float32_dtype), full_matrices=False)
            # Keep only supported basis components
            U = U[:, s > 1e-5].type(torch.float32)
            V = V[s > 1e-5].type(torch.float32)
            s = s[s > 1e-5].type(torch.float32)
            UsV_dict[key] = {'U': deepcopy(U), 's': deepcopy(s), 'V': deepcopy(V)}
            # Set all s to be the same scale
            s[s <= 1e-5] = 0
            cat_hidden_dim = V.shape[1] // len(ft_params)

            basis_dict[key] = U.to(self.device)
            sV_concat = V
            Vs = list(torch.split(sV_concat, cat_hidden_dim, dim=1))
            for idx, V in enumerate(Vs):
                V = torch.diag(s) @ V  # Simple and safe for all merging methods we use.
                s_model = s / s

                s_compositions_dict[idx][key] = s_model.to(self.device)
                V_compositions_dict[idx][key] = V.to(self.device)
        return basis_dict, s_compositions_dict, V_compositions_dict, UsV_dict

    def apply_Ss_on_Vs(self, task_Vs, task_Ss):
        task_sVs = [dict() for i in range(len(task_Vs))]
        for idx, (Vs, Ss) in enumerate(zip(task_Vs, task_Ss)):
            for key, V in Vs.items():
                if len(Ss[key].shape) == 2:
                    task_sVs[idx][key] = Ss[key] @ V
                else:
                    task_sVs[idx][key] = torch.diag(Ss[key]) @ V
        return task_sVs

    def remove_others(self, ftms_mats):
        other_mats = [dict() for i in range(len(ftms_mats))]
        transform_mats = [dict() for i in range(len(ftms_mats))]

        for m_idx, ftm_mats in enumerate(ftms_mats):
            for key, val in ftm_mats.items():
                if ':other' in key:
                    other_mats[m_idx][key] = val
                elif 'modules_to_save' in key:
                    other_mats[m_idx][key] = val
                else:
                    transform_mats[m_idx][key] = val
        print(f'Len other: {len(other_mats[0])}| len: transform: {len(transform_mats[0])}')
        return other_mats, transform_mats

    def add_others(self, ftms_mats, ftms_others):
        if isinstance(ftms_mats, list):
            return [self.add_others(ftms_mat, ftms_other) for ftms_mat, ftms_other in zip(ftms_mats, ftms_others)]

        for key, val in ftms_others.items():
            ftms_mats[key] = val
        return ftms_mats

    def transform(self, merge_config):
        # Setup parameters
        ptm_reference_params = deepcopy(self.param_handler(self.pretrained_model).get_ft_parameters())
        ftms_relevant_params = [ftm.get_ft_parameters() for ftm in self.ftms_params]
        ftms_task_dirs = self.get_task_directions(ptm_reference_params, ftms_relevant_params)

        ftms_task_mats = self.directions_to_matrices(ftms_task_dirs)
        ftms_others, ftms_mats = self.remove_others(ftms_task_mats)

        U, task_Ss, task_sVs, UsV_dict = self.apply_svd(
            ftms_mats,
            concat_across_output=merge_config.get('concat_across_output', True),
        )

        self.ingredients = {
            'ftms_relevant_params': ftms_relevant_params,
            'ftms_others': ftms_others,
            'ptm_reference_params': ptm_reference_params,
            'U': U,
            'task_Ss': task_Ss,
            'task_sVs': task_sVs,
            'UsV_dict': UsV_dict,
        }

        if merge_config.get('ingredients_path') is not None:
            torch.save(self.ingredients, merge_config['ingredients_path'])

    def merge(self, merge_config):
        if merge_config.get('ingredients_path') is not None:
            ingredients = torch.load(merge_config['ingredients_path'])
        else:
            ingredients = deepcopy(self.ingredients)

        ftms_others = ingredients['ftms_others']
        ptm_reference_params = ingredients['ptm_reference_params']
        U = ingredients['U']
        task_Ss = ingredients['task_Ss']
        task_sVs = ingredients['task_sVs']

        if merge_config.get('dare', False) or merge_config.get('merge_method') == 'dare-ties':
            print("Applying DARE")
            task_sVs = self.apply_dare(
                task_sVs, merge_config['dare_pruning_coeffs'], merge_config['dare_seed']
            )

        representations = self.representation_helper.directions_to_reps(task_sVs)
        ftms_reps = representations

        mask_fn = get_mask_fn(merge_config['merge_method'])
        masks = mask_fn(ftms_reps, **merge_config)
        ftms_reps = torch.vstack(ftms_reps).clone()
        masked_sVs = ftms_reps * masks
        pre_merge_sVs_dict = self.representation_helper.rep_to_state_dict(masked_sVs, task_sVs[0])
        rescaled_Vs = self.apply_Ss_on_Vs(pre_merge_sVs_dict, task_Ss)

        rescaled_Vs = torch.stack(self.representation_helper.directions_to_reps(rescaled_Vs), dim=0)
        merged_sV_ = masked_merge(
            merge_func=merge_config.get('merging_type'), vectors=rescaled_Vs, weights=self.scaling_coeffs
        )
        merged_sV_sd = self.representation_helper.rep_to_state_dict(merged_sV_, task_sVs[0])

        merged_sd = self.reconstruct_merged_sd(U, merged_sV_sd)

        if merge_config.get('merge_method') in ('tv', 'sum'):
            merging_fn = partial(tv_merging, **merge_config, weights=self.scaling_coeffs)
        elif merge_config.get('merge_method') in ('ties', 'dare-ties'):
            merging_fn = partial(ties_merging, **merge_config, weights=self.scaling_coeffs)

        if merge_config.get('merge_other_params', False):
            merged_others, _ = self.representation_helper(ftms_others, merging_fn=merging_fn)
            merged_sd = self.add_others(merged_sd, merged_others)

        merged_sd = self.matrix_to_state_dict(merged_sd, ptm_reference_params)
        # Add merged sd to the ptm
        merged_base = deepcopy(self.pretrained_model)
        if merge_config.get('isotropize', False):
            print("Isotropizing merged state dict...")
            for key, val in merged_sd.items():
                U, S, V = torch.linalg.svd(val.to(float32_dtype), full_matrices=False)
                S_iso_value = S.mean()
                S_iso = S_iso_value * torch.ones_like(S)
                merged_sd[key] = U @ torch.diag(S_iso) @ V

        merged_model = self.add_task_parameters(merged_base, merged_sd, concat_across_output=merge_config.get('concat_across_output', True))
        return merged_model


class MatrixPerLayerMerger(TaskMerger):
    def __init__(self, finetuned_models, pretrained_model, param_handler, device=0, merge_config=None, mode="normal"):
        if mode == "low_resource":
            self.low_resource_init(
                finetuned_models=finetuned_models,
                pretrained_model=pretrained_model,
                param_handler=param_handler,
                device=device,
                merge_config=merge_config
            )
        elif mode == "normal":
            self.normal_init(
                finetuned_models=finetuned_models,
                pretrained_model=pretrained_model,
                param_handler=param_handler,
                device=device,
                merge_config=merge_config
            )
        elif mode == "lora_to_device":
            self.lora_to_device_init(
                finetuned_models=finetuned_models,
                pretrained_model=pretrained_model,
                param_handler=param_handler,
                device=device,
                merge_config=merge_config
            )
    
    def low_resource_init(self, finetuned_models, pretrained_model, param_handler, device=0, merge_config=None):
        nn.Module.__init__(self)
        print("=====================================")
        print(f"Device: {device}")
        print("=====================================")

        self.device = device
        self.scaling_coeffs = torch.tensor([1.] * len(finetuned_models))
        self.param_handler = param_handler
        
        self.finetuned_models = []
        for ft_model in finetuned_models:
            if isinstance(ft_model, dict):
                self.finetuned_models.append(ft_model)
            else:
                self.finetuned_models.append(ft_model)

        self.ftms_params = [param_handler(ft_model) for ft_model in self.finetuned_models]
        self.pretrained_model = pretrained_model
        self.pt_params = self.pretrained_model.state_dict()
        self.merge_config = merge_config

        self.layer_names = self.get_layer_names(self.ftms_params[0].get_ft_parameters())
        self.ingredients = None
        self.cache = {}
    
    def lora_to_device_init(self, finetuned_models, pretrained_model, param_handler, device=0, merge_config=None):
        nn.Module.__init__(self)
        print("=====================================")
        print(f"Device: {device}")
        print("=====================================")

        self.device = device
        self.scaling_coeffs = torch.tensor([1.] * len(finetuned_models)).to(self.device)
        self.param_handler = param_handler
        
        self.finetuned_models = []
        for ft_model in finetuned_models:
            if isinstance(ft_model, dict):
                self.finetuned_models.append({k: v.to(self.device) for k, v in ft_model.items()})
            elif hasattr(ft_model, 'to'):
                self.finetuned_models.append(ft_model.to(self.device))
            else:
                self.finetuned_models.append(ft_model)

        self.ftms_params = [param_handler(ft_model) for ft_model in self.finetuned_models]
        self.pretrained_model = pretrained_model
        self.pt_params = self.pretrained_model.state_dict()
        self.merge_config = merge_config

        self.layer_names = self.get_layer_names(self.ftms_params[0].get_ft_parameters())
        self.ingredients = None
        self.cache = {}
        
    
    def normal_init(self, finetuned_models, pretrained_model, param_handler, device=0, merge_config=None):
        super().__init__(
            finetuned_models=finetuned_models,
            pretrained_model=pretrained_model,
            param_handler=param_handler,
            device=device,
            merge_config=merge_config
        )

        self.layer_names = self.get_layer_names(self.ftms_params[0].get_ft_parameters())
        self.ingredients = None
        self.cache = {}

    def set_scaling_coeffs(self, scaling_coeffs):
        self.scaling_coeffs = torch.tensor(scaling_coeffs)

    def _apply_delta(self, new_sd, key, delta_w):
        for name, param in new_sd.named_parameters():
            if name == key:
                param.data += self.scaling_coeffs.to(param.device) * delta_w.type_as(param.data)

    def _process_ties(self, tensor_list, topK=10):
        original_shape = tensor_list[0].shape
        tensor_list = list(map(torch.flatten, tensor_list))
        merged_tv, rows_to_keep, mask = ties_merging(tensor_list, topK=topK)
        return merged_tv.reshape(original_shape)

    def get_iso_matrix(self, ftms_task_dirs):
        summed_vectors = sum([ftms_task_dirs[i] for i in range(len(ftms_task_dirs))])
        return isotropize_matrix(summed_vectors)

    def get_tsv_delta_w(self, ftms_task_dirs):
        sv_reduction = 1 / len(ftms_task_dirs)
        for i, vec in enumerate(ftms_task_dirs):
            u, s, v = torch.linalg.svd(vec.to(float32_dtype), full_matrices=False)
            if i == 0:
                sum_u = torch.zeros_like(u)
                sum_s = torch.zeros_like(s)
                sum_v = torch.zeros_like(v)
            reduced_index_s = int(s.shape[0] * sv_reduction)
            # select only the first reduced_index_s columns of u and place them
            sum_u[:, i * reduced_index_s: (i + 1) * reduced_index_s] = u[
                :, :reduced_index_s
            ]
            sum_s[i * reduced_index_s: (i + 1) * reduced_index_s] = s[
                :reduced_index_s
            ]
            # select only the first reduced_index_s rows of v and place them
            sum_v[i * reduced_index_s: (i + 1) * reduced_index_s, :] = v[
                :reduced_index_s, :
            ]
        u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
        u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)

        return torch.linalg.multi_dot((u_u, v_u, torch.diag(sum_s), u_v, v_v)).type_as(ftms_task_dirs[0])
    
    def get_tsv_delta_w_low_rank(self, ftms_task_dirs, target_rank=16):
        sv_reduction = 1 / len(ftms_task_dirs)
        reduced_index_s = int(min(ftms_task_dirs[0].shape) * sv_reduction)
        targeted_reduced_rank = min(target_rank, reduced_index_s)
        for i, vec in enumerate(ftms_task_dirs):
            u, s, v = torch.svd_lowrank(vec.to(float32_dtype), q=target_rank*4, niter=10)
            vh = v.T
            if i == 0:
                sum_u = torch.zeros(u.shape[0], targeted_reduced_rank * len(ftms_task_dirs)).to(u.device)
                sum_s = torch.zeros(targeted_reduced_rank * len(ftms_task_dirs)).to(s.device)
                sum_vh = torch.zeros(targeted_reduced_rank * len(ftms_task_dirs), vh.shape[1]).to(vh.device)
            # select only the first reduced_index_s columns of u and place them
            sum_u[:, i * targeted_reduced_rank: (i + 1) * targeted_reduced_rank] = u[
                :, :targeted_reduced_rank
            ]
            sum_s[i * targeted_reduced_rank: (i + 1) * targeted_reduced_rank] = s[
                :targeted_reduced_rank
            ]
            # select only the first reduced_index_s rows of v and place them
            sum_vh[i * targeted_reduced_rank: (i + 1) * targeted_reduced_rank, :] = vh[
                :targeted_reduced_rank, :
            ]
        u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
        u_vh, s_vh, vh_vh = torch.linalg.svd(sum_vh, full_matrices=False)

        return torch.linalg.multi_dot((u_u, v_u, torch.diag(sum_s), u_vh, vh_vh)).type_as(ftms_task_dirs[0])
    
    def get_dare_delta_w(self, M_list, dare_coeff=0.3, merge=True):
        M_list = deepcopy(M_list)
        for i in range(len(M_list)):
            m = self.randbin(M_list[i].shape[0], M_list[i].shape[1], dare_coeff).to(M_list[i].device)
            M_list[i] = M_list[i] * m
            M_list[i] = M_list[i] / (1 - dare_coeff)
        return torch.stack(M_list).sum(dim=0) if merge else torch.stack(M_list)

    def get_cart_delta_w(self, ftms_task_dirs, pruning_rank=0.04, scaling_coeffs=1.):
        theta_avg = torch.stack(ftms_task_dirs).mean(dim=0)
        sum = torch.zeros_like(theta_avg)
        for i in range(len(ftms_task_dirs)):
            tau = ftms_task_dirs[i] - theta_avg
            U, S, Vh = torch.linalg.svd(tau.to(float32_dtype), full_matrices=False)
            pruning_rank_k = math.ceil(pruning_rank * S.shape[0])
            sum += U[:, :pruning_rank_k] @ torch.diag(S[:pruning_rank_k]) @ Vh[:pruning_rank_k, :]
        return theta_avg + scaling_coeffs * sum
    
    def iso_cts(self, tensors, merge_config):
        target_rank = merge_config.get('lora_rank', 16)
        num_of_tasks = len(tensors)
        s = 1
        k_for_common = target_rank - num_of_tasks * s
        assert k_for_common > 0, "Target rank too small for the number of tasks."
        result_TA = torch.stack(tensors, dim=0).sum(dim=0)

        # common subspace from TA
        U, S, Vh = torch.linalg.svd(result_TA, full_matrices=False)
        V = Vh.T
        
        U_common, S_common, V_common = U[:, :k_for_common], S[:k_for_common], V[:, :k_for_common]

        # task-specific (rank-s) from residuals
        u_list, v_list, s_list = [], [], []
        for t in range(num_of_tasks):
            Dt = tensors[t]
            Rt =  (Dt - U_common @ (U_common.T @ Dt))  # remove common (left projection)
            u, ss, vh = torch.linalg.svd(Rt, full_matrices=False)
            u_list.append(u[:, :s])
            v_list.append(vh[:s, :].T)
            s_list.append(ss[:s])

        U_star = torch.cat([U_common] + u_list, dim=1)
        V_star = torch.cat([V_common] + v_list, dim=1)
        
        # orthonormalize
        U,_, Vh = torch.linalg.svd(U_star, full_matrices=False)
        U_star = U @ Vh
        U,_, Vh = torch.linalg.svd(V_star, full_matrices=False)
        V_star = U @ Vh

        # isotropic scaling (mean singular value over selected directions)
        sigma = (S_common.sum() + torch.cat(s_list).sum()) / target_rank
        return sigma * (U_star @ V_star.T)
   
    
    def get_cart_delta_cache(self, ftms_task_dirs,cached_svd, pruning_rank=0.04, cart_scaling_coeffs=1.):
        theta_avg = torch.stack(ftms_task_dirs).mean(dim=0)
        sum_ = torch.zeros_like(theta_avg)
        # means = []
        for i in range(len(ftms_task_dirs)):
            tau = ftms_task_dirs[i] - theta_avg
            U, Vh = cached_svd[i]
            U, Vh = U.to(tau.device), Vh.to(tau.device)
            # means.append( (U@Vh - tau).abs().mean().item())
            pruning_rank_k = math.ceil(pruning_rank * U.shape[1])
            sum_ +=  U[:, :pruning_rank_k]  @ Vh[:pruning_rank_k, :]
        # print("Average reconstruction error per task:", sum(means)/len(means))
        return theta_avg + cart_scaling_coeffs * sum_
    
    def get_cart_cache(self, ftms_task_dirs):
        theta_avg = torch.stack(ftms_task_dirs).mean(dim=0)
        tmp_cache = []
        device = self.pretrained_model.device
        for i in range(len(ftms_task_dirs)):
            tau = ftms_task_dirs[i] - theta_avg
            U, S, Vh = torch.linalg.svd(tau.to(float32_dtype), full_matrices=False)
            U = U @ torch.diag(S)
            U, Vh = U.to(device), Vh.to(device)
            tmp_cache.append((U, Vh))
        return tmp_cache
    
    def get_pcb_delta_w(self, tensor_list, topK=30):
        if topK > 1:
            topK /= 100
        original_shape = tensor_list[0].shape
        tensor_list_ = list(map(torch.flatten, tensor_list))
        merged_tv, _, _ = PCB_merge(torch.stack(tensor_list_, dim=0), pcb_ratio=1-topK)
        return merged_tv.reshape(original_shape)
    
    def util_get_core_matrices(self, A_list, B_list):
        A_list = [a.to(self.device) for a in A_list]
        B_list = [b.to(self.device) for b in B_list]
        
        r, n = A_list[0].shape
        m, _ = B_list[0].shape

        A_stack = torch.cat(A_list, dim=0)  # shape: (T*r, n)
        B_stack = torch.cat(B_list, dim=1)  # shape: (m, T*r)

        Vh_A_ref = torch.linalg.svd(A_stack.to(float32_dtype), full_matrices=False)[2]  # shape: (T*r, n)
        U_B_ref = torch.linalg.svd(B_stack.to(float32_dtype), full_matrices=False)[0]  # shape: (m, T*r)

        M_list = []
        for i, (A, B) in enumerate(zip(A_list, B_list)):
            M_aligned = U_B_ref.T @ B @ A @ Vh_A_ref.T
            M_list.append(M_aligned)

        return M_list, U_B_ref, Vh_A_ref

    def get_core_matrices(self, ftms_params_ab, key, merge_config):
        if key in self.cache:
            return self.cache[key]
        # Extract A and B matrices from all tasks
        A_list, B_list = zip(*[ftm[key] for ftm in ftms_params_ab])
        
        # Move to device
        M_list, U_B_ref, Vh_A_ref = self.util_get_core_matrices(A_list, B_list)

        self.cache[key] = (M_list, U_B_ref, Vh_A_ref)
        return M_list, U_B_ref, Vh_A_ref

    def get_knots_components(self, ftms_task_dirs):
        stack = torch.cat(ftms_task_dirs, dim=1)  # shape: (n, T*r*n)
        U, S, Vh = torch.linalg.svd(stack.to(float32_dtype), full_matrices=False)
        # Keep only supported basis components
        U = U[:, S > 1e-5].type(torch.float32)
        Vh = Vh[S > 1e-5].type(torch.float32)
        S = S[S > 1e-5].type(torch.float32)

        S[S <= 1e-5] = 0
        Vs = einops.rearrange(Vh, 'Tr (b c) -> b Tr c', b=len(ftms_task_dirs))
        return U, S, list(Vs)
    

    
    @torch.no_grad()
    def get_low_rank_components_in_core_space(self, A_list, B_list, rank=16, beta=1.0):
        """
        核心函数，包括加速版本
        """
        tmp_time = time.time()
        A_list = [a.to(self.device) for a in A_list]
        B_list = [b.to(self.device) for b in B_list]
        M_list, U_B_ref, Vh_A_ref = self.util_get_core_matrices(A_list=A_list, B_list=B_list)
        # print("Time to get core matrices:", time.time() - tmp_time, "U_B_ref shape:", U_B_ref.shape, "Vh_A_ref shape:", Vh_A_ref.shape, "M_list[0] shape:", M_list[0].shape)
        tmp_time = time.time()
        # to reconstruct: U_B_ref @ M_merged @ Vh_A_ref
        # U_B_ref shape: (m, T*r), Vh_A_ref shape: (T*r, n)
        tl.set_backend('pytorch')
        stack = torch.stack(M_list, dim=-1).to(self.device)  # shape: (T*r, T*r, T)
        total_norm = torch.norm(stack, dim=(0,1), keepdim=True)
        stack_norm = (stack / total_norm) * (total_norm * beta + total_norm.mean() * (1 - beta))
        # beta = 1.0, no normalization
        # beta = 0.0，total normalization
        
        (core, factors), rec_errors = partial_tucker(
            stack_norm , 
            rank=[rank, rank],
            modes=[0, 1], 
            n_iter_max=1000,
            tol=1e-6,
            verbose=False,
            init='svd')
        U0, U1 = factors
        #core shape: r, r, T, change to a list of (r, r)
        # cores = [core[:, :, i] for i in range(core.shape[2])]
        U0 = U_B_ref @ U0   # m, r     
        U1 = Vh_A_ref.T @ U1 # n ,r
        # U, S, Vh = torch.linalg.svd(U0_new.to(float32_dtype), full_matrices=False)
        # U0 = U[:, :rank]          # m, rank
        # U, S, Vh = torch.linalg.svd(U1_new.to(float32_dtype), full_matrices=False)
        # U1 = U[:, :rank]          # n, rank
        cores = []
        for t in range(core.shape[2]):
            C_t = U0.T @ B_list[t] @ A_list[t] @ U1         # [rank, rank]
            cores.append(C_t)
        return U0, U1, cores  
    @torch.no_grad()
    def get_low_rank_components(self, A_list, B_list, rank=16, beta=1.0):
        # Low-rank decomposition directly in the original space
        A_list = [a.to(self.device) for a in A_list]
        B_list = [b.to(self.device) for b in B_list]
        ftms_task_dirs = [B@A for A, B in zip(A_list, B_list)]
        tl.set_backend('pytorch')
        stack = torch.torch.stack(ftms_task_dirs, dim=-1)  # shape: (n,n,T)
        total_norm = torch.norm(stack, dim=(0,1), keepdim=True)
        stack_norm = (stack / total_norm) * (total_norm * beta + total_norm.mean() * (1 - beta))
        
        (core, factors), rec_errors = partial_tucker(
            stack_norm, 
            rank=[rank, rank],
            modes=[0, 1], 
            n_iter_max=1000,
            tol=1e-6,
            verbose=False,
            init='svd')
        U0, U1 = factors
        #core shape: r, r, T, change to a list of (r, r)
        # cores = [core[:, :, i] for i in range(core.shape[2])]
        cores = []
        for t in range(core.shape[2]):
            W_t = stack[:, :, t]          # 原始 ΔW^{(t)}
            C_t = U0.T @ W_t @ U1         # [rank, rank]
            cores.append(C_t)
        return U0, U1, cores
    def _merge_tensors(self, tensors, merge_config, key=None):
        if merge_config.get('merge_method') == 'mean':
            return torch.stack(tensors).mean(dim=0)
        elif merge_config.get('merge_method') in ('sum', 'tv'):
            return torch.stack(tensors).sum(dim=0)
        elif merge_config.get('merge_method') == 'ties':
            return self._process_ties(tensors, merge_config.get('topK', 10))
        elif merge_config.get('merge_method') == 'dare':
            if self.lmc:
                tensors_dare = self.get_dare_delta_w(tensors, merge_config.get('dare_pruning_coeffs', 0.3), merge=False)
                tensors_dare = [t * self.scaling_coeffs[i] for i, t in enumerate(tensors_dare)]
                return torch.stack(tensors_dare).sum(dim=0)
            return self.get_dare_delta_w(tensors, merge_config.get('dare_pruning_coeffs', 0.3), merge=True)

        elif merge_config.get('merge_method') == 'dare-ties':
            tensors_dare = self.get_dare_delta_w(tensors, merge_config.get('dare_pruning_coeffs', 0.3), merge=False)
            return self._process_ties(tensors_dare, merge_config.get('topK', 10))
        elif merge_config.get('merge_method') == 'tsv':
            result = self.get_tsv_delta_w(tensors)
            return result.type_as(tensors[0]) if hasattr(result, 'type_as') else result
        elif merge_config.get('merge_method') == 'tsv_low_rank_svd':
            result = self.get_tsv_delta_w_low_rank(tensors, target_rank=merge_config.get('lora_rank', 16))
            return result.type_as(tensors[0]) if hasattr(result, 'type_as') else result
        elif merge_config.get('merge_method') == 'iso-cts':
            result = self.iso_cts(tensors, merge_config)
            return result.type_as(tensors[0]) if hasattr(result, 'type_as') else result
        elif merge_config.get('merge_method') == 'cart_cache':
            # cache version of cart
            if not hasattr(self, 'cart_cache'):
                    self.cart_cache = {}
            if key not in self.cart_cache: # calcualte cache
                cached_svd = self.get_cart_cache(tensors)
                self.cart_cache[key] = cached_svd
            else:
                cached_svd = self.cart_cache[key]
            return self.get_cart_delta_cache(
                tensors,
                cached_svd,
                merge_config.get('cart_pruning_rank', 0.04),
                merge_config.get('cart_scaling_coeffs', 0.1)
            )
        elif merge_config.get('merge_method') == 'cart':
            return self.get_cart_delta_w(
                tensors,
                merge_config.get('cart_pruning_rank', 0.04),
                merge_config.get('cart_scaling_coeffs', 0.1)
            )
        elif merge_config.get('merge_method') == 'pcb':
            return self.get_pcb_delta_w(
                tensors,
                merge_config.get('topK', 40),
            )
        else:
            raise ValueError(f"Unknown merge_method: {merge_config.get('merge_method')}")

    def truncated_svd(self, matrix, target_rank):
        U, S, Vh = torch.linalg.svd(matrix.to(float32_dtype), full_matrices=False)
        return U[:, :target_rank] @ torch.diag(S[:target_rank]) @ Vh[:target_rank, :]
    
    def merge_MSUs(self, A_list, B_list, rank=16):
        import numpy as np
        from sklearn.cluster import KMeans
        A_list = [a.to(self.device) for a in A_list]
        B_list = [b.to(self.device) for b in B_list]
        input_dim = A_list[0].shape[1]
        output_dim = B_list[0].shape[0]
        device = A_list[0].device
        dtype = A_list[0].dtype

        MSUs = []
        for A, B in zip(A_list, B_list):
            r = A.shape[0]
            for i in range(r):
                a_i = A[i, :].reshape(-1)      # [input_dim]
                b_i = B[:, i].reshape(-1)      # [output_dim]
                MSUs.append(torch.cat([a_i, b_i], dim=0))  # [input_dim + output_dim]

        msu_mat = torch.stack([m.detach().cpu().float() for m in MSUs], dim=0)  # [N, D]
        msu_np = msu_mat.numpy()

        kmeans = KMeans(n_clusters=rank, n_init=10)
        kmeans.fit(msu_np)

        centers = torch.from_numpy(kmeans.cluster_centers_).float()  # [rank, D] on CPU
        labels = kmeans.labels_  # [N]

        eps = 1e-8
        for c in range(rank):
            idx = np.where(labels == c)[0]
            if idx.size == 0:
                raise ValueError(f"Cluster {c} has no members.")

            cluster = msu_mat[idx]  # [Nc, D]
            avg_inf = cluster.abs().amax(dim=1).mean()     # 平均∞范数（簇内）
            cen_inf = centers[c].abs().amax()              # 中心∞范数
            centers[c] *= (avg_inf / (cen_inf + eps))

        A_new = centers[:, :input_dim].to(device=device, dtype=dtype).contiguous()          # [rank, input_dim]
        B_cols = centers[:, input_dim:].to(device=device, dtype=dtype).contiguous()         # [rank, output_dim]
        B_new = B_cols.t().contiguous()                                                     # [output_dim, rank]

        out_scale = np.sqrt(len(MSUs) / rank)
        
        return A_new, B_new * out_scale
    
    def robustmerge(self, A_list, B_list, topK, eps=1e-8):
        A_list = [a.to(self.device) for a in A_list]
        B_list = [b.to(self.device) for b in B_list]
        A_shape = A_list[0].shape
        B_shape = B_list[0].shape
        device = A_list[0].device
        dtype = A_list[0].dtype
        
        original_A = torch.stack(A_list, dim=0)  # shape: (T, r, n)
        
        A_stacked_vectors = torch.stack([a.flatten() for a in A_list], dim=0)  # shape: (T, r*n)
        B_stacked_vectors = torch.stack([b.flatten() for b in B_list], dim=0)  # shape: (T, m*r)
        A_pruned, _, _ = topk_values_mask(A_stacked_vectors, K=topK, return_mask=True)
        B_pruned, _, _ = topk_values_mask(B_stacked_vectors, K=topK, return_mask=True)
        
        A_pruned = A_pruned.reshape((-1, A_shape[0], A_shape[1]))  # shape: (T, r, n)
        B_pruned = B_pruned.reshape((-1, B_shape[0], B_shape[1]))  # shape: (T, m, r)
        
        # S: (T, r)
        scaling = original_A.abs().sum(dim=-1) / (A_pruned.abs().sum(dim=-1) + eps)

        # cross-task normalization across tasks (dim=0): (T, r)
        scaling = scaling / (scaling.sum(dim=0, keepdim=True) + eps)

        # apply to B columns: (T, m, r)
        B_pruned = B_pruned * scaling.unsqueeze(1)  # (T, 1, r) broadcast over m

        return A_pruned.sum(dim=0), B_pruned.sum(dim=0)

    
       
    @torch.no_grad()
    def merge(self, merge_config):
        if hasattr(self, 'merged_task_vector'):
            del self.merged_task_vector
            del self.cached_merge_config
        merged_task_vector = self._merge_into_task_vector(merge_config)
        new_sd = deepcopy(self.pretrained_model)
        for key, delta_w in merged_task_vector.items():
            self._apply_delta(new_sd, key, delta_w)
        del merged_task_vector
        return new_sd

    @torch.no_grad()
    def _merge_into_task_vector(self, merge_config):
        print(f"Merging using {merge_config.get('merge_space')} - {merge_config.get('merge_method')}")
        print(f"Isotropizing = {merge_config.get('isotropize', False)}")
        print("Config:", merge_config)
        # Determine merge space and prepare parameters
        if not hasattr(self, "ftms_params_ab"):
            self.ftms_params_ab = [ftm.get_ft_ab_parameters() for ftm in self.ftms_params]
            self.relevant_ab_keys = self.ftms_params[0].get_ft_ab_parameters().keys()
        ftms_params_ab = self.ftms_params_ab
        relevant_ab_keys = self.relevant_ab_keys

        # with newer peft versions, the keys may not have '.base_layer' in them
        # so we handle it by replacing it with an empty string (a cleaner solution would be nice)
        all_keys = self.pretrained_model.state_dict().keys()
        
        merged_task_vector = {}

        avg_ranks = []
        
        MERGE_SPACE = merge_config.get('merge_space')
        if MERGE_SPACE is None or MERGE_SPACE == '':
            raise ValueError(f"Unknown merge_space: {merge_config.get('merge_space')}")
        
        if MERGE_SPACE == 'full':
            desc = "Merging full space"
            if merge_config.get("low_rank", False):
                desc += ", Using low-rank approximation"
                desc += f" (rank={merge_config.get('lora_rank', 16)})"
            
            for key in tqdm(all_keys, desc=desc):
                key_base = key.replace('.base_layer', '')
                if key_base in ftms_params_ab[0]:
                    tensor_list = [(ft_dir[key_base][1] @ ft_dir[key_base][0]).to(self.device) for ft_dir in ftms_params_ab ]
                    delta_w = self._merge_tensors(tensor_list, merge_config, key=key_base)
                    
                    if merge_config.get("low_rank", False):
                        delta_w = self.truncated_svd(delta_w, merge_config.get('lora_rank', 16))
                        
                    if merge_config.get('isotropize', False):
                        delta_w = isotropize_matrix(delta_w)
                        
                    if COMPUTE_AVERAGE_RANK:
                        avg_ranks.append(torch.linalg.matrix_rank(delta_w).item())
                    elif COMPUTE_FIRST_RANK:
                        if not avg_ranks:
                            avg_ranks.append(torch.linalg.matrix_rank(delta_w).item())
                        else:
                            avg_ranks.append(avg_ranks[-1])
                    else:
                        avg_ranks.append(-1)
                    merged_task_vector[key] = delta_w.to(self.pretrained_model.device)

        elif MERGE_SPACE == 'knots':
            desc = "Merging knots space"
            if merge_config.get("low_rank", False):
                desc += ", Using low-rank approximation"
                desc += f" (rank={merge_config.get('lora_rank', 16)})"
            for key in tqdm(all_keys, desc=desc):
                key_base = key.replace('.base_layer', '')
                if key_base in ftms_params_ab[0]:
                    if key_base in self.cache:
                        U, S, Vs = self.cache[key_base]
                    else:
                        tensor_list = [(ft_dir[key_base][1] @ ft_dir[key_base][0]).to(self.device) for ft_dir in ftms_params_ab ]
                        U, S, Vs = self.get_knots_components(tensor_list)
                        self.cache[key_base] = (U, S, Vs)

                    Vs_merged = self._merge_tensors(Vs, merge_config, key=key_base)
                    
                    if merge_config.get('isotropize', False):
                        # if not isotropize here, the result will be the same as full, 
                        if merge_config.get("low_rank", False):
                            Vs_merged = self.truncated_svd(Vs_merged, merge_config.get('lora_rank', 16))
                        Vs_merged = isotropize_matrix(Vs_merged)
                        delta_w = U @ torch.diag(S) @ Vs_merged
                    else:
                        delta_w = U @ torch.diag(S) @ Vs_merged
                        if merge_config.get("low_rank", False):
                            delta_w = self.truncated_svd(delta_w, merge_config.get('lora_rank', 16))
                        
                    if COMPUTE_AVERAGE_RANK:
                        avg_ranks.append(torch.linalg.matrix_rank(delta_w).item())
                    elif COMPUTE_FIRST_RANK:
                        if not avg_ranks:
                            avg_ranks.append(torch.linalg.matrix_rank(delta_w).item())
                        else:
                            avg_ranks.append(avg_ranks[-1])
                    else:
                        avg_ranks.append(-1)
                    merged_task_vector[key] = delta_w.to(self.pretrained_model.device)
            
        elif MERGE_SPACE == 'core':
            desc = "Merging core space"
            if merge_config.get("low_rank", False):
                desc += ", Using low-rank approximation"
                desc += f" (rank={merge_config.get('lora_rank', 16)})"
            for key in tqdm(all_keys, desc=desc):
                key_base = key.replace('.base_layer', '')
                if key_base in relevant_ab_keys:
                    M_list, U_B_ref, Vh_A_ref = self.get_core_matrices(ftms_params_ab, key_base, merge_config)

                    M_merged = self._merge_tensors(M_list, merge_config, key=key_base)
                    
                    if merge_config.get("low_rank", False):
                        M_merged = self.truncated_svd(M_merged, merge_config.get('lora_rank', 16))
                    if merge_config.get('isotropize', False):
                        M_merged = isotropize_matrix(M_merged)
                        
                    if COMPUTE_AVERAGE_RANK:
                        avg_ranks.append(torch.linalg.matrix_rank(M_merged).item())
                    elif COMPUTE_FIRST_RANK:
                        if not avg_ranks:
                            avg_ranks.append(torch.linalg.matrix_rank(M_merged).item())
                        else:
                            avg_ranks.append(avg_ranks[-1])
                    else:
                        avg_ranks.append(-1)
                    
                    delta_w = U_B_ref @ M_merged @ Vh_A_ref
                
                    
                    merged_task_vector[key] = delta_w.to(self.pretrained_model.device)
                    
        elif MERGE_SPACE.startswith('low_rank'):
            desc = "Merging Low-rank space"
            assert merge_config.get("low_rank", False), "Low-rank merging requires 'low_rank' to be True in merge_config."
            desc += f" (rank={merge_config.get('lora_rank', 16)})"
            beta = merge_config.get('beta', 0.0)
            for key in tqdm(all_keys, desc=desc):
                key_base = key.replace('.base_layer', '')
                if key_base in relevant_ab_keys:
                    low_rank_cache_key = f"{merge_config.get('merge_space')}_{key_base}_{beta:.2f}"
                    if low_rank_cache_key in self.cache:
                        U0, U1, cores = deepcopy(self.cache[low_rank_cache_key])
                    else:
                        A_list, B_list = zip(*[ftm[key_base] for ftm in ftms_params_ab])
                        if merge_config.get('merge_space') == 'low_rank_core':
                            U0, U1, cores = self.get_low_rank_components_in_core_space(
                                rank=merge_config.get('lora_rank', 16), A_list=A_list, B_list=B_list, beta=beta)
                        elif merge_config.get('merge_space') == 'low_rank':
                            U0, U1, cores = self.get_low_rank_components(
                                rank=merge_config.get('lora_rank', 16), A_list=A_list, B_list=B_list, beta=beta)
                        self.cache[low_rank_cache_key] = (U0, U1, cores)
                        print("U0 shape:", U0.shape, "U1 shape:", U1.shape, "Number:", len(cores), "Distance to Id U0:", (U0.T@U0 - torch.eye(U0.shape[1], device=U0.device)).norm().item(), "Distance to Id U1:", (U1.T@U1 - torch.eye(U1.shape[1], device=U1.device)).norm().item())
                    core_merged = self._merge_tensors(cores, merge_config, key=key_base)
                    
                    delta_w = U0 @ core_merged @ U1.T
                    
                    if merge_config.get('isotropize', False):
                        delta_w = isotropize_matrix(delta_w)
                        
                    if COMPUTE_AVERAGE_RANK:
                        avg_ranks.append(torch.linalg.matrix_rank(delta_w).item())
                    elif COMPUTE_FIRST_RANK:
                        if not avg_ranks:
                            avg_ranks.append(torch.linalg.matrix_rank(delta_w).item())
                        else:
                            avg_ranks.append(avg_ranks[-1])
                    else:
                        avg_ranks.append(-1)
                    merged_task_vector[key] = delta_w.to(self.pretrained_model.device)
        
        elif MERGE_SPACE == 'lego':
            desc = "Merging Using LEGO"
            if merge_config.get("low_rank", False):
                desc += ", Using low-rank approximation"
                desc += f" (rank={merge_config.get('lora_rank', 16)})"
            for key in tqdm(all_keys, desc=desc):
                key_base = key.replace('.base_layer', '')
                if key_base in ftms_params_ab[0]:
                    if key_base in self.cache:
                        A_merged, B_merged = self.cache[key_base]
                    else:
                        A_list, B_list = zip(*[ftm[key_base] for ftm in ftms_params_ab])
                        A_merged, B_merged = self.merge_MSUs(A_list, B_list, rank=merge_config.get('lora_rank', 16))
                        self.cache[key_base] = (A_merged, B_merged)
                        
                    delta_w = B_merged @ A_merged
                    
                    if merge_config.get('isotropize', False):
                        delta_w = isotropize_matrix(delta_w)
                    if COMPUTE_AVERAGE_RANK:
                        avg_ranks.append(torch.linalg.matrix_rank(delta_w).item())
                    elif COMPUTE_FIRST_RANK:
                        if not avg_ranks:
                            avg_ranks.append(torch.linalg.matrix_rank(delta_w).item())
                        else:
                            avg_ranks.append(avg_ranks[-1])
                    else:
                        avg_ranks.append(-1)
                    merged_task_vector[key] = delta_w.to(self.pretrained_model.device)
                    
                    
        elif MERGE_SPACE == 'robustmerge':
            desc = "Merging Using RobustMerge"
            if merge_config.get("low_rank", False):
                desc += ", Using low-rank approximation"
                desc += f" (rank={merge_config.get('lora_rank', 16)})"
            for key in tqdm(all_keys, desc=desc):
                key_base = key.replace('.base_layer', '')
                if key_base in ftms_params_ab[0]:
                    
                    A_list, B_list = zip(*[ftm[key_base] for ftm in ftms_params_ab])
                    A_merged, B_merged = self.robustmerge(A_list, B_list, topK=merge_config.get('topK', 10))
                        
                    delta_w = B_merged @ A_merged
                    
                    if merge_config.get('isotropize', False):
                        delta_w = isotropize_matrix(delta_w)
                        
                    if COMPUTE_AVERAGE_RANK:
                        avg_ranks.append(torch.linalg.matrix_rank(delta_w).item())
                    elif COMPUTE_FIRST_RANK:
                        if not avg_ranks:
                            avg_ranks.append(torch.linalg.matrix_rank(delta_w).item())
                        else:
                            avg_ranks.append(avg_ranks[-1])
                    else:
                        avg_ranks.append(-1)
                    merged_task_vector[key] = delta_w.to(self.pretrained_model.device)
                        
        else:
            raise ValueError(f"Unknown merge_space: {merge_config.get('merge_space')}")
        if avg_ranks:
            if COMPUTE_AVERAGE_RANK:
                STRING = "The rank is computed for each layer and averaged at the end."
            elif COMPUTE_FIRST_RANK:
                STRING = "The rank is computed only for the first merged layer"
            else:
                STRING = "Rank computation is disabled."
            print(f"Average rank of delta_W across all layers: {sum(avg_ranks)/len(avg_ranks):.2f}", f"[{STRING}]")
        return merged_task_vector
    
    @torch.no_grad()
    def reuse_merged_task_vector(self, merge_config):
        if not hasattr(self, 'merged_task_vector'):
            merged_task_vector = self._merge_into_task_vector(merge_config)
            self.merged_task_vector = merged_task_vector
            self.cached_merge_config = deepcopy(merge_config)
        else:
            for key in merge_config:
                if merge_config[key] != self.cached_merge_config.get(key):
                    if key != "scaling_coeffs":
                        raise ValueError("Cached merged task vector config does not match the provided config.")
                    
        new_sd = deepcopy(self.pretrained_model)
        for key, delta_w in self.merged_task_vector.items():
            self._apply_delta(new_sd, key, delta_w)
        return new_sd
    

@torch.no_grad()
def grassmann_geodesic_distance(Q1, Q2):
    Q1, _ = torch.linalg.qr(Q1, mode="reduced")
    Q2, _ = torch.linalg.qr(Q2, mode="reduced")

    M = Q1.T @ Q2                               # (r, r)
    s = torch.linalg.svdvals(M)                  # singular values = cos(theta)
    s = torch.clamp(s, 0.0, 1.0)                 # numerical safety
    theta = torch.acos(s)                        # principal angles in [0, pi/2]
    return torch.linalg.vector_norm(theta, ord=2).item() # sqrt(sum theta^2)

@torch.no_grad()
def grassmann_chordal_distance_fast(Q1, Q2):
    Q1, _ = torch.linalg.qr(Q1, mode="reduced")
    Q2, _ = torch.linalg.qr(Q2, mode="reduced")

    M = Q1.T @ Q2
    r = M.shape[0]
    dc2 = r - torch.sum(M * M)  # ||M||_F^2
    dc = torch.sqrt(torch.clamp(dc2, min=0.0))
    return dc.item()

@torch.no_grad()
def simple_distance(Q1, Q2):
    def canon_by_positive_absmax(Q: torch.Tensor) -> torch.Tensor:
        """
        Q: (n, r)
        return: (n, r)
        """
        idx = Q.abs().argmax(dim=0)                 # (r,)
        anchor = Q[idx, torch.arange(Q.shape[1], device=Q.device)]  # (r,)
        signs = torch.sign(anchor)                  # (r,) in {-1, 0, +1}
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)

        return Q * signs.unsqueeze(0)
    Q1c = canon_by_positive_absmax(Q1)
    Q2c = canon_by_positive_absmax(Q2)
    
    return (Q1c-Q2c).abs().mean().item() # sqrt(sum theta^2)

def compute_subspace_cover(tensor_list, delta_w, lora_rank, avg_energy_list, total):
    u, s, vh = torch.linalg.svd(delta_w.to(float32_dtype), full_matrices=False)
    u = u[:, :lora_rank]
    vh = vh[:lora_rank, :]
    
    
    projected_matrices = [u@(u.T@i@vh.T)@vh for i in tensor_list ]
    
    preserved_energy = [(p*i).sum() / torch.norm(p) / torch.norm(i) for p, i in zip(projected_matrices, tensor_list)]
    for i , e in enumerate(preserved_energy):
        avg_energy_list[i] += e.item()
    return avg_energy_list, total + 1

def isotropize_matrix(matrix):
    # effective rank
    rank = torch.linalg.matrix_rank(matrix).item()
    U, S, Vh = torch.linalg.svd(matrix.to(float32_dtype), full_matrices=False)
    U, S, Vh = U[:, :rank], S[:rank], Vh[:rank, :]
    S_iso = S.mean() * torch.ones_like(S)
    return U @ torch.diag(S_iso) @ Vh


def get_merge_handler(rep_type):
    if rep_type == 'vector':
        return VectorMerger
    elif rep_type == 'svd':
        return SVDMerger
    elif rep_type == 'matrix_per_layer':
        return MatrixPerLayerMerger
