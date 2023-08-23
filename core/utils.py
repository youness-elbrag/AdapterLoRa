import Attentions 

def make_lora_replace(model, quantized , Rank ,  depth=1, path="", verbose=True):
    if depth > 10:
        return model
    
    if isinstance(model, nn.Linear) and ([Linear for Attentions in path ]):
        if verbose:
            print(f"Find linear {path}:", type(model))
        return quantized(model,Rank)
    
    for key, module in model.named_children():  # Using named_children() for cleaner iteration
        if isinstance(module, nn.Linear) and ([Linear for Attentions in path ]):
            layer = quantized(module,Rank)
            setattr(model, key, layer)
            if verbose:
                print(f"Find linear {path}:{key} :", type(module))
                
        elif isinstance(module, nn.ModuleList):
            for i, elem in enumerate(module):
                layer = make_lora_replace(elem, depth+1, f"{path}:{key}[{i}]", verbose=verbose)
                if layer is not None:
                    module[i] = layer
                
        elif isinstance(module, nn.ModuleDict):
            for module_key, item in module.items():
                layer = make_lora_replace(item, depth+1, f"{path}:{key}:{module_key}", verbose=verbose)
                if layer is not None:
                    module[module_key] = layer
                
        else:
            layer = make_lora_replace(module, depth+1, f"{path}:{key}", verbose=verbose)
            if layer is not None:
                setattr(model, key, layer)
    
    return model