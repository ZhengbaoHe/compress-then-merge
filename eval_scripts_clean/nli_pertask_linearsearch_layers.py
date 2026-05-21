import sys, os
# XXX
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from copy import deepcopy
import time
from pprint import pformat

import numpy as np
import torch
import wandb

from task_merger import get_merge_handler
from utils import evaluate_logits, get_config_from_name, prepare_experiment_config, set_seed, parse_eval_args, merge_args_into_task_merge_config, get_strides

# Set TOKENIZERS_PARALLELISM to true
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import transformers

transformers.utils.logging.set_verbosity(transformers.logging.ERROR)


def run_BIG_function(args):
    EVAL_SPLIT = 'val'
    EVAL_TEST = True
    BIGSEED = args.seed

    print("Seed : ", BIGSEED)
    set_seed(BIGSEED)
    
    if args.file_path == "":
        file_path = f"results/llama-tmp"
    else:
        file_path = args.file_path
    os.makedirs(file_path, exist_ok=True)
    print("Results will be saved to : ", file_path)
    with open(os.path.join(file_path, "record.txt"), "a") as f:
        f.write(f"Time : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
    
    # Get config
    config_name = args.config
    print("Config name : ", config_name)
    EARLY_STOPPING_STEPS = 2

    TASK_HEADS_PATH = "data/llama-3.2-1B/heads.pt" if '1B' in config_name else "heads.pt"
    # TASK_HEADS_PATH = "heads.pt"
    from utils import DEFAULT_DEVICE;  device = DEFAULT_DEVICE
    print(device)
    
    
    raw_config = get_config_from_name(config_name, device=device)
    print("Raw config: ", raw_config, "task_merge_config: ", raw_config['task_merge_config'])
    config = prepare_experiment_config(raw_config)
    config['task_merge_config'] = merge_args_into_task_merge_config(config['task_merge_config'], args)
    dataset_names = np.array([i['name'] for i in raw_config['dataset']])
    dataloaders = np.array([i for i in config['data']])
    mask_class = np.array([i['mask_class'] for i in config['dataset']])
    print(f"mask_class labels: {mask_class}")

    # Parameters are tuned in the order specified in search_config
    default_params = {'scaling_coeffs': 0.9,
                      'topK': 30,
                      'dare_pruning_coeffs': 0.05,
                      'beta': args.beta,
                      }  # Default config
    
    order_of_processing_params = [
        'scaling_coeffs',
    ]
    search_config = {
        'scaling_coeffs': np.arange(0.0, 4.0, step=0.025).tolist(),
        'topK': np.arange(0, 101, step=5)[::-1].tolist() + [1],
        'dare_pruning_coeffs': [1e-5] + np.arange(0.00, 1.01, step=0.05).tolist(), 
        'beta': [0.00, 0.25, 0.5, 0.75],
        }
    
    strides_config_first_round = get_strides(search_config, rounds=1)
    
    strides_config_second_round = get_strides(search_config, rounds=2)
    
    print("Strides for first round: ", strides_config_first_round)
    print("Strides for second round: ", strides_config_second_round)

    if "low_rank" in args.merge_space and args.search_beta_first:
        order_of_processing_params = ['beta', 'scaling_coeffs']
    if 'dare' in config['task_merge_config']['merge_method']:
        order_of_processing_params.append('dare_pruning_coeffs')
    if 'ties' in config['task_merge_config']['merge_method'] or 'robustmerge' in config['task_merge_config']['merge_method']:
        order_of_processing_params.append('topK')

    task_heads = torch.load(TASK_HEADS_PATH)

    finetuned_llama3_8b = {
        'snli': 92.49796416938111, 'mnli': 90.30820173204279, 'sick': 91.58173664900122, 'qnli': 94.48512585812358, 'rte': 89.85507246376812, 'scitail': 96.51928504233303, }

    finetuned_llama32_1b = {"mnli": 84.093, "snli": 88.578, "qnli": 89.725, 'sick': 90.216, 'rte': 78.986, 'scitail': 94.967}

    print("Using Llama fine-tuned acc")
    fine_tuned_acc = finetuned_llama3_8b if '8B' in config_name else finetuned_llama32_1b

    print(f'Finetuned Accs: {fine_tuned_acc}')
    print("="*50 + "\n" + "Starting Linear Search over parameters: ", order_of_processing_params,"\n" + "="*50)

    def merge_and_eval(merger, EVAL_SPLIT='val', running_merge_config=None, reuse_scaling_coeffs=False):
        set_seed(BIGSEED)
        print("EVAL_SPLIT : ", EVAL_SPLIT)

        all_results_with_merge_config = {}
        print('Creating Merge')

        merger.set_scaling_coeffs(running_merge_config['scaling_coeffs'])

        t0 = time.time()
        if reuse_scaling_coeffs:
            merged_model = merger.reuse_merged_task_vector(running_merge_config)
        else:
            merged_model = merger.merge(running_merge_config)
        print(f"Time taken to merge: {time.time() - t0}")

        merged_model.config.pad_token_id = 128001
        merged_model.config.use_cache = False
        merged_model.config.pretraining_tp = 1

        print('Evaluate Merged Model on Each Dataset')
        from utils import DEFAULT_DEVICE;  device = DEFAULT_DEVICE
        avg_accuracy = 0.
        avg_norm_accuracy = 0.
        t0 = time.time()
        for i, loader_dict in enumerate(dataloaders):
            loader = loader_dict['test'][EVAL_SPLIT]
            with torch.no_grad():
                for name, param in merged_model.named_parameters():
                    # Inject task head into model
                    if 'modules_to_save' in name:
                        param.copy_(task_heads[dataset_names[i]])

            acc = evaluate_logits(merged_model, loader, device, mask_class[i], silent=True)
            print(f"{dataset_names[i]} Normalized accuracy is {np.round((acc * 100)/ fine_tuned_acc[dataset_names[i]] *100, 3)}", f"\t{dataset_names[i]} accuracy is {np.round(acc * 100, 3)}")
            all_results_with_merge_config[dataset_names[i]] = acc * 100
            all_results_with_merge_config[dataset_names[i] + '_norm_acc'] = (acc * 100) / fine_tuned_acc[dataset_names[i]] * 100
            avg_accuracy += acc * 100
            avg_norm_accuracy += (acc * 100) / fine_tuned_acc[dataset_names[i]] * 100
        print(f"Time taken to evaluate: {time.time() - t0}")
        avg_accuracy /= len(dataloaders)
        avg_norm_accuracy /= len(dataloaders)
        print(f'Average Accuracy is {np.round(avg_accuracy, 3)}')
        print(f'Average Normalized Accuracy is {np.round(avg_norm_accuracy, 3)}')
        all_results_with_merge_config['Average_acc'] = avg_accuracy
        all_results_with_merge_config['Average_norm_acc'] = avg_norm_accuracy
        
        all_results_with_merge_config.update(running_merge_config)
        
        return all_results_with_merge_config


    with torch.no_grad():
        lora_state_dicts = np.array([i for i in config['models']['bases']])
        MergeClass = get_merge_handler(config['task_merge_config']['representation'])
        merger = MergeClass(
            lora_state_dicts,
            pretrained_model=config['models']['new'],
            param_handler=config['param_handler'],
            device=device,
            merge_config=config['task_merge_config'],
            mode="lora_to_device"
        )

        if config['task_merge_config']['ingredients_path'] is None or not os.path.exists(config['task_merge_config']['ingredients_path']):
            merger.transform(config['task_merge_config'])

        print(config['task_merge_config'])
        if args.only_eval:
            order_of_processing_params = []
        for param in order_of_processing_params:
            print("="*50 , "\nStarting Linear Search over parameter: ", param, "\n" + "="*50)
            best_val_results_with_merge_config = {'Average_norm_acc': 0.0}
            best_val_results_with_merge_config.update(config['task_merge_config'])
            best_val_results_with_merge_config.update(default_params)

            def run_search_loop(values, tmp_best_val_results_with_merge_config, reuse_scaling_coeffs=False):
                early_stopping = EARLY_STOPPING_STEPS
                for value in values:
                    current_searched_parameters = deepcopy(default_params)
                    current_searched_parameters[param] = value
                    config['task_merge_config'].update(current_searched_parameters)
                    print(f'Search Run with: {current_searched_parameters}')
                    
                    all_results_with_merge_config = merge_and_eval(merger, EVAL_SPLIT=EVAL_SPLIT, running_merge_config=config['task_merge_config'], reuse_scaling_coeffs=reuse_scaling_coeffs)
                    
                    if (all_results_with_merge_config['Average_norm_acc'] >= tmp_best_val_results_with_merge_config['Average_norm_acc']):
                        tmp_best_val_results_with_merge_config = deepcopy(all_results_with_merge_config)
                        early_stopping = EARLY_STOPPING_STEPS
                    else:
                        early_stopping -= 1
                        if early_stopping <= 0:
                            print("Early stopping")
                            break
                return tmp_best_val_results_with_merge_config

            # Prepare for multi-level search
            original_values = search_config[param]
            
            N = len(original_values)
            visited_indices = set()

            # Define strides for 3 levels
            stride1 = strides_config_first_round[param]
            stride2 = strides_config_second_round[param]
            
            def get_best_idx(val, values):
                if val is None: return -1
                for i, v in enumerate(values):
                    if abs(v - val) < 1e-8:
                        return i
                return -1

            # Level 1: Coarse Search
            indices_round_1 = list(range(0, N, stride1))
            values_round_1 = [original_values[i] for i in indices_round_1 if i not in visited_indices]
            if values_round_1:
                print("="*50 ,f"\nLevel 1 Search (Stride {stride1}):", " ".join([f"{v:.3f}" for v in values_round_1]), "\n"+"="*50)
                tmp_best_val_results_with_merge_config = run_search_loop(values_round_1, tmp_best_val_results_with_merge_config={'Average_norm_acc': 0.0}, reuse_scaling_coeffs=param=='scaling_coeffs')
                if tmp_best_val_results_with_merge_config['Average_norm_acc'] >= best_val_results_with_merge_config['Average_norm_acc']:
                    best_val_results_with_merge_config = deepcopy(tmp_best_val_results_with_merge_config)
                visited_indices.update(indices_round_1)

            # Level 2: Medium Search
            best_val = best_val_results_with_merge_config.get(param)
            best_idx = get_best_idx(best_val, original_values)
            
            if best_idx != -1 and stride1 > 1:
                start_idx = max(0, best_idx - stride1)
                end_idx = min(N, best_idx + stride1 + 1)
                
                indices_round_2 = [i for i in range(start_idx, end_idx, stride2) if i not in visited_indices]
                values_round_2 = [original_values[i] for i in indices_round_2]
                
                if values_round_2:
                    print("="*50 ,f"\nLevel 2 Search (Stride {stride2}) around {best_val}:", " ".join([f"{v:.3f}" for v in values_round_2]), "\n"+"="*50)
                    tmp_best_val_results_with_merge_config = run_search_loop(values_round_2, tmp_best_val_results_with_merge_config={'Average_norm_acc': 0.0}, reuse_scaling_coeffs=param=='scaling_coeffs')
                    if tmp_best_val_results_with_merge_config['Average_norm_acc'] >= best_val_results_with_merge_config['Average_norm_acc']:
                        best_val_results_with_merge_config = deepcopy(tmp_best_val_results_with_merge_config)
                    visited_indices.update(indices_round_2)

            # Level 3: Fine Search
            best_val = best_val_results_with_merge_config.get(param)
            best_idx = get_best_idx(best_val, original_values)
            
            if best_idx != -1 and stride2 > 1:
                start_idx = max(0, best_idx - stride2)
                end_idx = min(N, best_idx + stride2 + 1)
                
                indices_round_3 = [i for i in range(start_idx, end_idx, 1) if i not in visited_indices]
                values_round_3 = [original_values[i] for i in indices_round_3]
                
                if values_round_3:
                    print("="*50 ,f"\nLevel 3 Search (Stride 1) around {best_val}:", " ".join([f"{v:.3f}" for v in values_round_3]), "\n"+"="*50)
                    tmp_best_val_results_with_merge_config = run_search_loop(values_round_3, tmp_best_val_results_with_merge_config={'Average_norm_acc': 0.0}, reuse_scaling_coeffs=param=='scaling_coeffs')
                    if tmp_best_val_results_with_merge_config['Average_norm_acc'] >= best_val_results_with_merge_config['Average_norm_acc']:
                        best_val_results_with_merge_config = deepcopy(tmp_best_val_results_with_merge_config)
                    visited_indices.update(indices_round_3)

            default_params[param] = best_val_results_with_merge_config[param]
            
        def eval_test():
            detailed_test_results = merge_and_eval(merger, EVAL_SPLIT='test', running_merge_config=config['task_merge_config'])
            datasets = ['snli', 'mnli', 'sick', 'qnli', 'rte', 'scitail']
            output_result = " & ".join([f"{np.round(detailed_test_results[dataset+'_norm_acc'], 2)}" for dataset in datasets]) + f" & {np.round(detailed_test_results['Average_norm_acc'], 2)} "
            print(f"Normalized Test results: {output_result}")
            print(detailed_test_results)
            return detailed_test_results, output_result

        if args.only_eval:
            detailed_test_results, output_result = eval_test()
            best_params = deepcopy(default_params)
            for key in default_params.keys():
                if hasattr(args, key):
                    best_params[key] = getattr(args, key)
            
            with open(f"results/eval-final/llama/{args.merge_method}-{args.merge_space}-LowRank-{args.low_rank}-Iso-{args.isotropize}.txt", "a") as f:
                f.write("=" * 80 + "\n")
                # 记录时间
                f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
                f.write(f"Method: {args.merge_method}, Merge Space: {args.merge_space}, isotropize: {args.isotropize}, Low Rank: {args.low_rank}\n")
                f.write(f"Args:\n{pformat(vars(args))}\n")
                f.write(f"Normalized Test results: {output_result}\n")
                f.write(f"Test result dict:\n{pformat(detailed_test_results)}\n")
                f.write(f"Best parameters:\n{pformat(best_params)}\n")
                f.write(f"Seed: {args.seed}\n")
                f.write(f"LoRA rank: {args.lora_rank}\n")
                f.write("=" * 80 + "\n\n")
                
        elif EVAL_TEST:
            best_params = deepcopy(default_params)
            print("Best params :", best_params)
            
            config['task_merge_config'].update(best_params)
            
            detailed_test_results, output_result = eval_test()
            
            
            with open(os.path.join(file_path, f"{args.merge_method}-{args.merge_space}-LowRank-{args.low_rank}-Iso-{args.isotropize}.txt"), "a") as f:
                f.write("=" * 80 + "\n")
                # 记录时间
                f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
                f.write(f"Method: {args.merge_method}, Merge Space: {args.merge_space}, isotropize: {args.isotropize}, Low Rank: {args.low_rank}\n")
                f.write(f"Args:\n{pformat(vars(args))}\n")
                f.write(f"Normalized Test results: {output_result}\n")
                f.write(f"Test result dict:\n{pformat(detailed_test_results)}\n")
                f.write(f"Best parameters:\n{pformat(best_params)}\n")
                f.write(f"LoRA rank: {args.lora_rank}\n")
                f.write("=" * 80 + "\n\n")

            

if __name__ == "__main__":
    args = parse_eval_args()
    # if args.file_path == "":
    #     file_path = f"results/llama-tmp"
    # else:
    #     file_path = args.file_path
    # if not os.path.exists(file_path):
    #     os.makedirs(file_path, exist_ok=True)
    # file = os.path.join(file_path, f"{args.merge_method}-{args.merge_space}-LowRank-{args.low_rank}-Iso-{args.isotropize}.log")
    # sys.stdout = open(file, "w", encoding="utf-8")
    run_BIG_function(args)
