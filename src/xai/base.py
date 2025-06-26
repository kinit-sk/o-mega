import torch
from typing import Union

from architecture import SentenceTransformerToHF


def parse_module_path_more_wildcard(
    model: torch.nn.Module, 
    module_path_expressions: Union[list[str], dict]
) -> list[list[str]]:
    if module_path_expressions is None:
        return []
    if type(module_path_expressions) == list:
        return_module_list = []
        path_expressions = module_path_expressions
    elif type(module_path_expressions) == dict:
        return_module_list = []
        path_expressions = list(module_path_expressions.keys())
        attention_types = list(module_path_expressions.values())

    all_attention_types = []
    for path_it, path_expression in enumerate(path_expressions):
        module_list = [[]]
        module_strings = path_expression.split(".")
        for m_name in module_strings:
            if m_name != "*":
                for i in range(len(module_list)):
                    module_list[i].append(m_name)
            else:
                expanded_modules = []
                for m in module_list:
                    try:
                        sub_model = model
                        for sub_part in m:
                            sub_model = getattr(sub_model, sub_part)  # Ensure validity

                        for child_name, _ in sub_model.named_children():
                            expanded_modules.append(m + [child_name])
                    except AttributeError:
                        raise ValueError(f"Invalid path: {'.'.join(m)} does not exist.")
                module_list=expanded_modules
        return_module_list.extend(module_list)
        if type(module_path_expressions) == dict:
            all_attention_types.extend([attention_types[path_it]]*len(module_list))

    if type(module_path_expressions) == list:
        return return_module_list
    return return_module_list, all_attention_types



def parse_module_path(
    model: torch.nn.Module, 
    module_path_expressions: Union[list[str], dict]
) -> list[list[str]]:
    if module_path_expressions is None:
        return []
    if type(module_path_expressions) == list:
        assert sum([expr.count("*") > 1 for expr 
            in module_path_expressions]) == 0, "Only one wildcard is supported for now"
        return_module_list = []
        path_expressions = module_path_expressions
    elif type(module_path_expressions) == dict:
        assert sum([expr.count("*") > 1 for expr 
            in module_path_expressions.keys()]) == 0, "Only one wildcard is supported for now"
        return_module_list = []
        path_expressions = list(module_path_expressions.keys())
        attention_types = list(module_path_expressions.values())

    all_attention_types = []
    for path_it, path_expression in enumerate(path_expressions):
        module_list = [[]]
        module_strings = path_expression.split(".")
        for m_name in module_strings:
            if m_name != "*":
                for i in range(len(module_list)):
                    module_list[i].append(m_name)
            else:
                help_model = model

                for mod_iter in module_list[0]:
                    help_model = getattr(help_model, mod_iter)

                for it, (ch_name, _) in enumerate(help_model.named_children()):
                    if it == 0:
                        module_list[0].append(ch_name)
                    else:
                        module_list.append(module_list[0][:-1] + [ch_name])
        return_module_list.extend(module_list)
        if type(module_path_expressions) == dict:
            all_attention_types.extend([attention_types[path_it]]*len(module_list))

    if type(module_path_expressions) == list:
        return return_module_list
    return return_module_list, all_attention_types

def semantic_search_forward_function(
    model: SentenceTransformerToHF, encoding: dict[str], embedding: torch.Tensor
) -> dict[str, torch.Tensor]:
    model_output = model._forward_lrp_gae_method(**encoding)    
    
    output = {
        "prediction": torch.nn.functional.cosine_similarity(
            model_output["prediction"], embedding, dim=1
        )
    }
    if model_output.get("attentions", None) is not None:
        output["attentions"] = model_output["attentions"]
    return output