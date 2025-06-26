import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Union, Literal, Optional, Callable
from collections.abc import Iterable
from transformers import PreTrainedTokenizer

from .base import parse_module_path
import utils


class GAE_Explain:
    all_A = []
    all_A_grad = []

    shared_cross_att_first_calls_is_qtext = False

    def __init__(
        self, 
        module_path_expressions: Union[list[str], dict, None] = None,
        apply_normalization: bool = False, 
        normalization_approach: Literal["min-max", "l2", "max-abs"] = "min-max",
        relevance_pooling: Literal["CLS_token", "mean"] = "CLS_token",
        custom_module_paths: dict = {}, custom_hooks: dict = {},
        shared_cross_att_first_calls_is_qtext: bool = True,
        last_shared_cross_att_text_only: bool = True,
        return_only_class_relevance: bool = True,
    ) -> None:
        # custom hooks specify hooks to apply for group of path expressions which are defined
        # in custom_module_paths
        # the correspondance between specific custom hooks and their path expressions is modeled using
        # the same dictionary keys
        self.custom_hooks = custom_hooks
        self.custom_module_paths = custom_module_paths
        self.custom_module_list = {}
        self.custom_handles = []

        # if module_path_expressions is an array, were saving attention maps in self-attention encoder architecture only
        # if module_path_expressions a dict, then key will be the path to the module we want to apply the hook to, and values
        # will define the type of that particular attention -> whether its self-attention of image modality, or for example
        # a cross-attention (query - image, key - text)...
        self.use_output_attentions_arg = False
        self.module_path_expressions = module_path_expressions
        if module_path_expressions is None:
            self.use_output_attentions_arg = True

        self.module_list = []
        self.attention_list = []

        # If the same weights are applied to both directions of cross attention, we need to know the order in which
        # the module is invoked, so that we know whether the odd or even calls correspond to for example cross_qtext_kimage 
        GAE_Explain.shared_cross_att_first_calls_is_qtext = shared_cross_att_first_calls_is_qtext
        self.last_shared_cross_att_text_only = last_shared_cross_att_text_only 

        self.distinguish_attention_types = True if type(module_path_expressions) == dict else False
        self.apply_normalization = apply_normalization
        self.normalization_approach = normalization_approach
        self.relevance_pooling = relevance_pooling
        self.return_only_class_relevance = return_only_class_relevance

        self.forward_hook_handles = []
        self.backward_hook_handles = []

        self.multimodal_hooks = {
            "self_text": [GAE_Explain.forward_hook_self_text, GAE_Explain.backward_hook_self_text],
            "self_image": [GAE_Explain.forward_hook_self_image, GAE_Explain.backward_hook_self_image],
            "cross_qtext_kimage": [GAE_Explain.forward_hook_qtext_kimage, GAE_Explain.backward_hook_qtext_kimage],
            "cross_qimage_ktext": [GAE_Explain.forward_hook_qimage_ktext, GAE_Explain.backward_hook_qimage_ktext],
            "shared_cross": [GAE_Explain.forward_hook_shared_cross, GAE_Explain.backward_hook_shared_cross],
        }

    def attach_hooks(self, model: torch.nn.Module) -> None:        
        custom_keys = list(self.custom_module_list.keys())
        for module_list_iter in range(len(custom_keys) + 1):
            if module_list_iter == 0:
                module_list = self.module_list
            else:
                custom_key = custom_keys[module_list_iter-1]
                module_list = self.custom_module_list[custom_key]

            for path_it, module_path in enumerate(module_list):
                helper_model = model
                for module in module_path:
                    try:
                        idx = int(module)
                        helper_model = helper_model[idx]
                    except:
                        helper_model = getattr(helper_model, module)

                if module_list_iter == 0:
                    if self.distinguish_attention_types == False:
                        self.forward_hook_handles.append(
                            helper_model.register_forward_hook(GAE_Explain.forward_hook)
                        )
                        self.backward_hook_handles.append(
                            helper_model.register_full_backward_hook(GAE_Explain.backward_hook)
                        )
                    else: 
                        attention_type = self.attention_list[path_it]
                        if attention_type not in ["self_text", "self_image", "cross_qtext_kimage", "cross_qimage_ktext", "shared_cross"]:
                            raise "Invalid attention type. For this specific attention type theres no default hook implemented"
                        
                        hooks = self.multimodal_hooks[attention_type]
                        self.forward_hook_handles.append(
                            helper_model.register_forward_hook(hooks[0])
                        )
                        self.backward_hook_handles.append(
                            helper_model.register_full_backward_hook(hooks[1])
                        )
                else:
                    if custom_key[-3:] == "_FW":
                        self.custom_handles.append(
                            helper_model.register_forward_hook(
                                self.custom_hooks[custom_key]
                            )
                        )
                    elif custom_key[-3:] == "_BW":
                        self.custom_handles.append(
                            helper_model.register_full_backward_hook(
                                self.custom_hooks[custom_key]
                            )
                        )
                    else:
                        raise ( 
                            "Key values in custom dictionaries need to end with either " +
                            "'_FW' or '_BW' to signify whether specified hook is applied " +
                            "to forward pass or backward pass."
                        )

    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.use_output_attentions_arg:
            return model
        
        self.module_list = parse_module_path(model, self.module_path_expressions)
        if self.distinguish_attention_types:
            self.module_list, self.attention_list = self.module_list

        for k in self.custom_module_paths.keys():
            self.custom_module_list[k] = parse_module_path(model, self.custom_module_paths[k])
        
        self.attach_hooks(model)
        return model

    @staticmethod
    def forward_hook(model, inp, out):
        GAE_Explain.all_A.append(
            inp[0].detach()
        )

    @staticmethod
    def backward_hook(model, grad_inp, grad_out):
        GAE_Explain.all_A_grad.append(
            grad_inp[0].detach()
        )

    ######

    @staticmethod
    def forward_hook_self_text(model, inp, out):
        layer_id = list(model._forward_hooks.keys())[0]
        GAE_Explain.all_A.append(
            (layer_id, "self_text", inp[0].detach())
        )

    @staticmethod
    def forward_hook_self_image(model, inp, out):
        layer_id = list(model._forward_hooks.keys())[0]
        GAE_Explain.all_A.append(
            (layer_id, "self_image", inp[0].detach())
        )

    @staticmethod
    def forward_hook_qimage_ktext(model, inp, out):
        layer_id = list(model._forward_hooks.keys())[0]
        GAE_Explain.all_A.append(
            (layer_id, "qimage_ktext", inp[0].detach())
        )

    @staticmethod
    def forward_hook_qtext_kimage(model, inp, out):
        layer_id = list(model._forward_hooks.keys())[0]
        GAE_Explain.all_A.append(
            (layer_id, "qtext_kimage", inp[0].detach())
        )

    ######

    @staticmethod
    def backward_hook_self_text(model, inp, out):
        layer_id = list(model._forward_hooks.keys())[0]
        GAE_Explain.all_A_grad.append(
            (layer_id, "self_text", inp[0].detach())
        )

    @staticmethod
    def backward_hook_self_image(model, inp, out):
        layer_id = list(model._forward_hooks.keys())[0]
        GAE_Explain.all_A_grad.append(
            (layer_id, "self_image", inp[0].detach())
        )

    @staticmethod
    def backward_hook_qimage_ktext(model, inp, out):
        layer_id = list(model._forward_hooks.keys())[0]
        GAE_Explain.all_A_grad.append(
            (layer_id, "qimage_ktext", inp[0].detach())
        )

    @staticmethod
    def backward_hook_qtext_kimage(model, inp, out):
        layer_id = list(model._forward_hooks.keys())[0]
        GAE_Explain.all_A_grad.append(
            (layer_id, "qtext_kimage", inp[0].detach())
        )

    ######

    @staticmethod
    def forward_hook_shared_cross(model, inp, out):
        layer_id = list(model._forward_hooks.keys())[0]
        first_qtext = GAE_Explain.shared_cross_att_first_calls_is_qtext 
        
        layer_ids = [a[0] for a in GAE_Explain.all_A]
        if layer_id in layer_ids:
            idx = np.where(np.array(layer_ids) == layer_id)[0][0]
            value_name = "shared_qimage_ktext" if first_qtext else "shared_qtext_kimage"
            GAE_Explain.all_A[idx].extend([value_name, inp[0].detach()])
        else:
            value_name = "shared_qtext_kimage" if first_qtext else "shared_qimage_ktext"
            GAE_Explain.all_A.append(
                [layer_id, value_name, inp[0].detach()]
            )
        
    @staticmethod
    def backward_hook_shared_cross(model, inp, out):
        layer_id = list(model._forward_hooks.keys())[0]
        first_qtext = GAE_Explain.shared_cross_att_first_calls_is_qtext 
    
        layer_ids = [a[0] for a in GAE_Explain.all_A_grad]
        if layer_id in layer_ids:
            idx = np.where(np.array(layer_ids) == layer_id)[0][0]
            value_name = "shared_qtext_kimage" if first_qtext else "shared_qimage_ktext"
            GAE_Explain.all_A_grad[idx].extend([value_name, inp[0].detach()])
        else:
            value_name = "shared_qimage_ktext" if first_qtext else "shared_qtext_kimage"
            GAE_Explain.all_A_grad.append(
                [layer_id, value_name, inp[0].detach()]
            )

    def explain_top_n_predictions(
        self, model: torch.nn.Module, encoding: dict[str, torch.Tensor], top_n: int = 5, 
        forward_function: Optional[Callable[
            [dict[str, torch.Tensor]], 
            dict[str, torch.Tensor]
        ]] = None, 
        token_idx_to_predict: Optional[int] = None
    ):
        all_relevance = []

        if forward_function is None:
            predictions = model(**encoding)[0]
        elif token_idx_to_predict is None:
            predictions = forward_function(model, encoding)["prediction"]
        else:
            predictions = forward_function(model, encoding, token_idx_to_predict)["prediction"]

        # computing explanation on seperate datapoints of batch
        for i in range(len(predictions)):

            pred = predictions[i]
            indices = torch.topk(pred, top_n, largest=True, sorted=True)[1]

            #preparing encoding data consisting of only one datapoint
            new_enc = {
                k: v[i][None]
                for k,v in encoding.items()
                if isinstance(v, torch.Tensor)
            }
            new_enc.update({
                k: v for k,v in encoding.items() if not isinstance(v, torch.Tensor)
            })

            datapoint_relevance = []
            for idx in indices:
                rel, _ = self._explain_batch(
                    model, new_enc, index=idx, 
                    forward_function=forward_function, 
                    token_idx_to_predict=token_idx_to_predict if token_idx_to_predict is None or type(token_idx_to_predict) == int else token_idx_to_predict[i]
                )
                datapoint_relevance.append(rel)

            datapoint_relevance = {
                k: torch.vstack([data[k] for data in datapoint_relevance])
                for k in datapoint_relevance[0]
            }
            all_relevance.append(datapoint_relevance)

        all_relevance = {
            k: torch.stack([data[k] for data in all_relevance])
            for k in all_relevance[0]
        }
        return all_relevance, predictions

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
            dict[str, torch.Tensor]
        ]] = None, 
        token_idx_to_predict: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        GAE_Explain.all_A = []
        GAE_Explain.all_A_grad = []
        GAE_Explain.shared_cross_att_counter = 0

        was_train = False
        if model.training == True:
            model.eval()
            was_train = True

        encoding = tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(utils.get_device())
        special_tokens_mask = torch.isin(
            encoding["input_ids"], 
            torch.tensor(tokenizer.all_special_ids, device=utils.get_device())
        )
        if self.use_output_attentions_arg:
            encoding["output_attentions"] = True

        if forward_function is None:
            model_output = model(**encoding)
            predictions = model_output[0]
        elif token_idx_to_predict is None:
            model_output = forward_function(model, encoding)
            predictions = model_output["prediction"]
        else:
            model_output = forward_function(model, encoding, token_idx_to_predict)
            predictions = model_output["prediction"]
        
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

        if self.use_output_attentions_arg:
            for attn in model_output["attentions"]:
                attn.retain_grad()

        mask.requires_grad = True
        model.zero_grad()
        masked_predictions = predictions * mask
        masked_predictions.sum().backward()

        if self.use_output_attentions_arg:
            GAE_Explain.all_A = model_output["attentions"]
            GAE_Explain.all_A_grad = [att.grad for att in model_output["attentions"]][::-1]
        relevance = self.calc_relevance(special_tokens_mask)

        if was_train:
            model.train()
        return relevance, predictions.detach()

    def accumulate_relevance(self) -> torch.Tensor:
        if self.distinguish_attention_types == False:        
            all_A = GAE_Explain.all_A
            all_A_grad = GAE_Explain.all_A_grad[::-1] #flip the order

            batch_size, num_tokens = all_A[0].shape[0], all_A[0].shape[2]
            
            identity = torch.Tensor(np.identity(num_tokens)).to(utils.get_device())
            identity = identity.tile((batch_size, 1, 1))

            relevance = identity
            for A, A_grad in zip(all_A, all_A_grad):
                A_bar = torch.clamp(A * A_grad, 0)  # rule 5  
                A_bar = A_bar.mean(axis=1)          # rule 5

                relevance = relevance + torch.matmul(A_bar, relevance) #rule 6
            return relevance

        # sort gradients to their respecitve A
        # if theres no gradients to specific A, that A will be skipped, deleted
        sorted_all_A = []
        sorted_all_A_grad = []
        for A in GAE_Explain.all_A:
            layer_id = A[0]
            for A_grad in GAE_Explain.all_A_grad:
                if A_grad[0] == layer_id:
                    if len(A) == 5:
                        # shared cross attention
                        if len(A_grad) == 5:
                            correct_order = [0, 1, 2, 3, 4]
                            if A[1] != A_grad[1]:
                                correct_order = [0, 3, 4, 1, 2]

                            correct_order = [A_grad[idx] for idx in correct_order]
                            sorted_all_A.append(A)
                            sorted_all_A_grad.append(correct_order)

                        else:
                            if (
                                self.last_shared_cross_att_text_only == True and 
                                    GAE_Explain.shared_cross_att_first_calls_is_qtext == True or 
                                self.last_shared_cross_att_text_only == False and 
                                    GAE_Explain.shared_cross_att_first_calls_is_qtext == False 
                            ):
                                correct_order = [0, 1, 2]
                            else:
                                correct_order = [0, 3, 4]

                            value_name = (
                                "shared_qtext_kimage" 
                                if self.last_shared_cross_att_text_only 
                                else "shared_qimage_ktext"
                            )
                            correct_A = [A[idx] for idx in correct_order]
                            correct_A_grad = A_grad[:1] + [value_name] + A_grad[2:]

                            sorted_all_A.append(correct_A)
                            sorted_all_A_grad.append(correct_A_grad)
                    else:
                        sorted_all_A.append(A)
                        sorted_all_A_grad.append(A_grad)
                    # continue
            
        all_A = sorted_all_A
        all_A_grad = sorted_all_A_grad
        R_ii, R_tt, R_qi_kt, R_qt_ki = None, None, None, None

        for A, A_grad in zip(all_A, all_A_grad):
            attention_type = A[1]
            if len(A) != 5:
                A_bar = torch.clamp(A[2] * A_grad[2], 0)    # rule 5
                A_bar = A_bar.mean(axis=1)                  # rule 5

            # image self-attention 
            if attention_type == "self_image":
                if R_ii is None:
                    batch_size, num_image_tokens = A_bar.shape[:2]
                    identity = torch.Tensor(np.identity(num_image_tokens)).to(utils.get_device())
                    R_ii = identity.tile((batch_size, 1, 1))
                
                old_R_ii = R_ii
                R_ii = R_ii + torch.matmul(A_bar, R_ii) # rule 6

                if R_qi_kt is not None:
                    R_qi_kt = R_qi_kt + torch.matmul(A_bar, R_qi_kt) # rule 7

            # text self-attention 
            elif attention_type == "self_text":
                if R_tt is None:
                    batch_size, num_text_tokens = A_bar.shape[:2]
                    identity = torch.Tensor(np.identity(num_text_tokens)).to(utils.get_device())
                    R_tt = identity.tile((batch_size, 1, 1))
                
                old_R_tt = R_tt
                R_tt = R_tt + torch.matmul(A_bar, R_tt) # rule 6

                if R_qt_ki is not None:
                    R_qt_ki = R_qt_ki + torch.matmul(A_bar, R_qt_ki) # rule 7

            # cross-attention
            else:   
                image_identity = torch.Tensor(np.identity(num_image_tokens)).to(utils.get_device())
                image_identity = image_identity.tile((batch_size, 1, 1))
                text_identity = torch.Tensor(np.identity(num_text_tokens)).to(utils.get_device())
                text_identity = text_identity.tile((batch_size, 1, 1))
                
                R_ii_hat = old_R_ii - image_identity
                R_ii_bar = R_ii_hat / R_ii_hat.sum(axis=2, keepdim=True) + image_identity   # rule 8, 9

                R_tt_hat = old_R_tt - text_identity
                R_tt_bar = R_tt_hat / R_tt_hat.sum(axis=2, keepdim=True) + text_identity    # rule 8, 9

                if attention_type == "qimage_ktext" or attention_type == "shared_qimage_ktext" and len(A) == 3:
                    if R_qi_kt is None:
                        R_qi_kt = torch.zeros(batch_size, num_image_tokens, num_text_tokens).to(utils.get_device())

                    old_R_qi_kt = R_qi_kt
                    R_qi_kt = R_qi_kt + torch.matmul(torch.matmul(torch.transpose(R_ii_bar, 1, 2), A_bar), R_tt_bar)    # rule 10
                        
                    if R_qt_ki is not None:
                        R_ii = R_ii + torch.matmul(A_bar, old_R_qt_ki)      # rule 11
                
                elif attention_type == "qtext_kimage" or attention_type == "shared_qtext_kimage" and len(A) == 3:
                    if R_qt_ki is None:
                        R_qt_ki = torch.zeros(batch_size, num_text_tokens, num_image_tokens).to(utils.get_device())

                    old_R_qt_ki = R_qt_ki
                    R_qt_ki = R_qt_ki + torch.matmul(torch.matmul(torch.transpose(R_tt_bar, 1, 2), A_bar), R_ii_bar)    # rule 10

                    if R_qi_kt is not None:
                        R_tt = R_tt + torch.matmul(A_bar, old_R_qi_kt)      # rule 11

                #shared cross attention
                elif len(A) == 5:
                    if attention_type == "shared_qimage_ktext":
                        image_idx = 2
                        text_idx = 4
                    else:
                        image_idx = 4
                        text_idx = 2

                    A_bar_qimage = torch.clamp(A[image_idx] * A_grad[image_idx], 0)    # rule 5
                    A_bar_qimage = A_bar_qimage.mean(axis=1)                           # rule 5

                    A_bar_qtext = torch.clamp(A[text_idx] * A_grad[text_idx], 0)    # rule 5
                    A_bar_qtext = A_bar_qtext.mean(axis=1)                          # rule 5

                    old_R_qi_kt = R_qi_kt
                    old_R_qt_ki = R_qt_ki

                    if R_qi_kt is None:
                        R_qi_kt = torch.zeros(batch_size, num_image_tokens, num_text_tokens).to(utils.get_device())
                    if R_qt_ki is None:
                        R_qt_ki = torch.zeros(batch_size, num_text_tokens, num_image_tokens).to(utils.get_device())

                    R_qi_kt = R_qi_kt + torch.matmul(torch.matmul(torch.transpose(R_ii_bar, 1, 2), A_bar_qimage), R_tt_bar)     # rule 10
                    R_qt_ki = R_qt_ki + torch.matmul(torch.matmul(torch.transpose(R_tt_bar, 1, 2), A_bar_qtext), R_ii_bar)      # rule 10
                    
                    if old_R_qt_ki is not None:
                        R_ii = R_ii + torch.matmul(A_bar_qimage, old_R_qt_ki)       # rule 11
                    if old_R_qi_kt is not None:
                        R_tt = R_tt + torch.matmul(A_bar_qtext, old_R_qi_kt)        # rule 11

                else:
                    raise "Invalid attention type"
 
        return {
            "self_image": R_ii,
            "self_text": R_tt,
            "query_image_key_text": R_qi_kt,
            "query_text_key_image": R_qt_ki
        }

    def calc_relevance(self, special_tokens_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            relevance = self.accumulate_relevance()
            if self.distinguish_attention_types or self.return_only_class_relevance == False:
                return relevance

            if self.relevance_pooling == "CLS_token":
                relevance = relevance[:, 0]
            elif self.relevance_pooling == "mean":
                eye_mask = torch.eye(relevance.shape[-1], dtype=torch.bool)[None]
                eye_mask = eye_mask.repeat(len(relevance), 1, 1)
                relevance[eye_mask] = 0
                relevance = relevance.mean(axis=1)
            else:
                raise ValueError("Invalid 'relevance_pooling' value")

            relevance[special_tokens_mask] = 0

            if self.apply_normalization and self.normalization_approach == "min-max":
                for i in range(len(relevance)):
                    max_r = torch.quantile(relevance, 0.95)
                    min_r = torch.quantile(relevance, 0.05)
                    relevance[i] = (relevance[i] - min_r) / (max_r - min_r + 1e-9)
                    relevance[i] = torch.clip(relevance[i], min=0, max=1)
            elif self.apply_normalization and self.normalization_approach == "l2":
                relevance = relevance / torch.norm(relevance, p=2)
            elif self.apply_normalization and self.normalization_approach == "max-abs":
                for i in range(len(relevance)):
                    max_abs_r = torch.quantile(relevance.absolute(), 0.95)
                    relevance[i] = relevance[i] / (max_abs_r + 1e-9)
                    relevance[i] = torch.clip(relevance[i], min=-1, max=1)
            elif self.apply_normalization:
                raise ValueError("Invalid normalization approach")

            return relevance
    
    def cleanup(self) -> None:
        for handle in self.forward_hook_handles:
            handle.remove()
        for handle in self.backward_hook_handles:
            handle.remove()

        self.forward_hook_handles = []
        self.backward_hook_handles = []
    
        GAE_Explain.all_A = []
        GAE_Explain.all_A_grad = []
        GAE_Explain.shared_cross_att_counter = 0