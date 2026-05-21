def get_vision_accuracies(model, rank, peft_type, n_tasks=8, dataset_names=None):
    if isinstance(rank, list):
        assert dataset_names is not None, "dataset_names must be provided when rank is a list"
        d = {}
        
        for i, r in enumerate(rank):
            print(dataset_names[i], r)
            d[dataset_names[i]] = get_vision_accuracies(model, r, peft_type)[dataset_names[i]]
        return d

    if model == "openai/clip-vit-base-patch32" and rank == 16 and peft_type == "lora":
        return {
            'food101': 84.86633663366336,
            'fer2013': 69.55234279742205,
            'fashionmnist': 94.0,
            'caltech101': 95.43165467625899,
            'cub': 62.5324114088159,
            'flowers': 85.39599934948772,
            'officehome': 88.61031518624641,
            'pet': 90.45996592844975,
            'stanford_cars': 74.0,
            'dtd': 58.3,
            'eurosat': 99.0,
            'gtsrb': 92.7,
            'mnist': 99.3,
            'resisc45': 88.4,
            'sun397': 64.5,
            'svhn': 96.2
        }
    else:
        raise ValueError(f"Accuracy for model {model}, rank {rank}, peft_type {peft_type} not found.")