import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Union, Literal, Optional, Callable
from collections.abc import Iterable
from transformers import PreTrainedTokenizer

from .base import parse_module_path, parse_module_path_more_wildcard
import utils

class ConservativeLRP:
    activations_to_store = []

    def __init__(
        self, 
        store_A_path_expressions: list[str], 
        attent_path_expressions: list[str], 
        norm_layer_path_expressions: list[str],
        apply_normalization: bool = True,
        normalization_approach: Literal["min-max", "l2", "max-abs"] = "min-max",
    ) -> None:
        self.store_A_path_expressions = store_A_path_expressions
        self.attent_path_expressions = attent_path_expressions
        self.norm_layer_path_expressions = norm_layer_path_expressions

        self.store_A_module_list = []
        self.attent_module_list = []
        self.norm_layer_module_list = []

        self.store_A_handles = []
        self.attent_handles = []
        self.norm_layer_handles = []

        self.apply_normalization = apply_normalization
        self.normalization_approach = normalization_approach

    def get_module_from_module_path(self, model: torch.nn.Module, module_path: list[str]):
        for module in module_path:
            try:
                idx = int(module)
                model = model[idx]
            except:
                model = getattr(model, module)
        return model

    def attach_hooks(
        self, model: torch.nn.Module, 
        store_A_hook: Optional[Callable] = None, 
        attent_hook: Optional[Callable] = None, 
        norm_layer_hook: Optional[Callable] = None, 
    ) -> None:
        if attent_hook is None:
            attent_hook = ConservativeLRP.detach_attention_map_hook
        if norm_layer_hook is None:
            norm_layer_hook = ConservativeLRP.detach_norm_layer_hook
        if store_A_hook is None:
            store_A_hook = ConservativeLRP.store_activations_hook

        all_module_lists = [
            self.attent_module_list,
            self.norm_layer_module_list,
            self.store_A_module_list
        ]
        all_hook_funcs = [
            attent_hook,
            norm_layer_hook,
            store_A_hook
        ]
        all_handles = [
            self.attent_handles,
            self.norm_layer_handles,
            self.store_A_handles
        ]
        for module_list, hook_func, handles in zip(all_module_lists, all_hook_funcs, all_handles):
            for module_path in module_list:
                mod = self.get_module_from_module_path(model, module_path)
                handles.append(mod.register_forward_hook(hook_func))

    def prepare_model(self, model:torch.nn.Module) -> torch.nn.Module:
        self.store_A_module_list = parse_module_path(model, self.store_A_path_expressions)
        a = parse_module_path_more_wildcard(model, self.store_A_path_expressions)
        self.attent_module_list = parse_module_path(model, self.attent_path_expressions)
        self.norm_layer_module_list = parse_module_path(model, self.norm_layer_path_expressions)

        self.attach_hooks(model)
        return model
         
    def cleanup(self) -> None:
        ConservativeLRP.activations_to_store = []
    
        for handle_list in [self.store_A_handles, self.attent_handles, 
                        self.norm_layer_handles]:
            for handle in handle_list:
                handle.remove()
            handle_list.clear()
        
    @staticmethod
    def store_activations_hook(model, inp, out):
        if type(out) == tuple:
            detached_out = out[0].detach()
            detached_out.requires_grad = True

            ConservativeLRP.activations_to_store.append(
                (out[0], detached_out)
            )
            return (detached_out, out[1:])

        detached_out = out.detach()
        detached_out.requires_grad = True 

        ConservativeLRP.activations_to_store.append(
            (out, detached_out)
        )
        return detached_out
        
    @staticmethod
    def detach_attention_map_hook(model, inp, out):
        return out.detach()

    @staticmethod
    def detach_norm_layer_hook(model, inp, out):
        tensor_inp = inp[0]
        
        mean = tensor_inp.mean(dim=-1, keepdim=True).detach()
        std = tensor_inp.std(dim=-1, keepdim=True).detach()

        input_norm = ((tensor_inp - mean) / (std + model.eps)) * model.weight + model.bias
        return input_norm

    def explain(
        self, 
        model: torch.nn.Module, 
        tokenizer: PreTrainedTokenizer, 
        loader: DataLoader
    ) -> Union[list[torch.Tensor], list[torch.Tensor]]:
        # TODO later
        pass

    def _explain_batch(
        self, 
        model: torch.nn.Module, 
        tokenizer: PreTrainedTokenizer, 
        texts: Union[str, list[str]], 
        index: Optional[int] = None,
        forward_function: Optional[Callable[
            [dict[str, torch.Tensor]], 
            torch.Tensor
        ]] = None, 
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ConservativeLRP.activations_to_store = []
        was_train = model.training
        model.eval()
        
        encoding = tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(utils.get_device())
        special_tokens_mask = torch.isin(
            encoding["input_ids"], 
            torch.tensor(tokenizer.all_special_ids, device=utils.get_device())
        )

        if forward_function is None:
            predictions = model(**encoding)[0]
        else:
            predictions = forward_function(model, encoding)
        # if isinstance(predictions,dict):
        #     attention=predictions['attentions']
            predictions=predictions['prediction']
        if predictions.ndim == 2:
            if index is not None and type(index) == int:
                idx = torch.ones(len(encoding["input_ids"]), dtype=torch.long) * index
            elif index is not None and isinstance(index, Iterable):
                idx = index
            else:
                idx = predictions.argmax(dim=1)

            mask = torch.zeros_like(predictions)
            mask[[np.arange(len(predictions)), idx]] = 1
        elif predictions.ndim == 1:
            mask = torch.ones_like(predictions)
        else:
            raise ValueError("We have yet to support three-or-more dimensional prediction tensors")
        
        mask.requires_grad = True
        model.zero_grad()
        
        masked_predictions = predictions * mask
        masked_predictions.sum().backward()

        for i in range(len(self.activations_to_store))[::-1]:
            act, act_detached = self.activations_to_store[i]
            if i != 0:
                R_layer = ((act_detached.grad)*act).sum()
                R_layer.backward()
            else:
                explanation = (act_detached.grad)*act
                explanation = explanation.sum(axis=2)
        
        explanation[special_tokens_mask] = 0
        
        if self.apply_normalization and self.normalization_approach == "min-max":
            for i in range(len(explanation)):
                max_r = torch.quantile(explanation, 0.95)
                min_r = torch.quantile(explanation, 0.05)
                explanation[i] = (explanation[i] - min_r) / (max_r - min_r + 1e-9)
                explanation[i] = torch.clip(explanation[i], min=0, max=1)
        elif self.apply_normalization and self.normalization_approach == "l2":
            explanation = explanation / torch.norm(explanation, p=2)
        elif self.apply_normalization and self.normalization_approach == "max-abs":
            for i in range(len(explanation)):
                max_abs_r = torch.quantile(explanation.absolute(), 0.95)
                explanation[i] = explanation[i] / (max_abs_r + 1e-9)
                explanation[i] = torch.clip(explanation[i], min=-1, max=1)
        elif self.apply_normalization:
            raise ValueError("Invalid normalization approach")

        if was_train:
            model.train()
        return explanation.detach(), predictions.detach()
