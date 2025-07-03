from typing import Any, Callable, Dict, List, Tuple, Union,Literal

import torch
from torch import Tensor
from captum.attr._core.occlusion import Occlusion
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.log import log_usage
from copy import deepcopy
import utils
import re
from architecture import get_tokenizer

class WordLevelOcclusion(Occlusion):
    """
    A modification of Occlusion that operates at the word level rather than the token level.
    This implementation respects word boundaries when performing occlusion, leading to more
    interpretable attributions for text inputs.
    """

    def __init__(self, forward_func: Callable, model,tokenizer) -> None:
        """
        Args:
            forward_func (Callable): The forward function of the model.
            tokenizer: The tokenizer used by the model to convert text to tokens.
        """
        super().__init__(forward_func)
        self.model=model
        self.tokenizer = tokenizer

    @log_usage()
    def attribute_text(
        self,
        text_input: str,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        sliding_window_shapes: int = 1,
        show_progress: bool = False,
        aggregation_method: str = "mean",  # Options: "mean", "sum", "max"
    ) -> Tensor:
        """
        Compute word-level attributions for a text input by occluding words.
        
        Args:
            text_input (str): The input text for which to compute attributions.
            baselines: The baseline value to use for occlusion. If None, a default will be used.
            target: Output indices for which to compute attributions.
            additional_forward_args: Additional args for the forward function.
            sliding_window_shapes: Number of perturbations to include in each batch.
            show_progress: Whether to display progress during computation.
            aggregation_method (str): Method to aggregate token-level attributions to word-level.
                                     Options: "mean", "sum", "max".
            
        Returns:
            Tensor: Word-level attributions.
        """
        
        # Step 1: Tokenize the input text
        #inputs = self.tokenizer(text_input, return_tensors="pt")
        #input_ids = inputs["input_ids"]
        
        # Step 2: Map words to token indices
        word_to_tokens, encoding = self._map_words_to_tokens(text_input) #, input_ids)
        
        # Step 3: Compute token-level attributions using the original Occlusion implementation
        # but with custom sliding windows that respect word boundaries - this is probably the most clean way how to do this
        attributions = self._compute_word_attributions(
            text_input,
            encoding,
            word_to_tokens,
            baselines,
            target,
            additional_forward_args,
            sliding_window_shapes,
            show_progress,
        )
        
        # Step 4: Aggregate token-level attributions to word-level
        word_attributions = self._aggregate_to_word_level(
            attributions, word_to_tokens, aggregation_method
        )
        
        return word_attributions
    
    def _map_words_to_tokens(self, text: str) -> Dict[int, List[int]]: #, token_ids: Tensor
        """
        Create a mapping from words to their corresponding token indices.
        
        Args:
            text (str): The input text.
            token_ids (Tensor): The tokenized input.
            
        Returns:
            Dict[int, List[int]]: A dictionary mapping word index to a list of token indices.
        """
        # This implementation should work with any tokenizer as long as it has the offset mapping function, if not then good luck...
        
        # Use the tokenizer's encoding with offsets to get character-level mappings
        encoding = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=True, return_tensors="pt") # this is kind of stupid since I already tokenize the text in attribute_text func()

        # Get the offset mapping (character start/end positions for each token) either with attribute access or dictionary key access
        offset_mapping = encoding.offset_mapping if hasattr(encoding, 'offset_mapping') else encoding['offset mapping']
        # print("offset_mapping type:", type(offset_mapping))
        # print("first few items:", offset_mapping[:])  # Print first 5 items to see the structure
        # Find word boundaries using split() - simple, efficient (but if the samples are in list you need to iterate ower them to use this)
        word_boundaries = []
        words = text.split()
        
        # Find the character positions of each word
        start_idx = 0
        for word in words:
            # Find the start position of the word
            word_start = text.find(word, start_idx)
            word_end = word_start + len(word)
            word_boundaries.append((word_start, word_end))
            # Update the start index for the next search
            start_idx = word_end

        # Map each word to its corresponding tokens
        word_to_tokens = {}
        for word_idx, (word_start, word_end) in enumerate(word_boundaries):
            word_tokens = []
            
            # Find all tokens that overlap with this word's character range
            for token_idx, (token_start, token_end) in enumerate(offset_mapping[0]):
                # Skip special tokens (they have (0,0) offsets in most tokenizers)
                if token_start == 0 and token_end == 0:
                    continue
                    
                # Check if this token overlaps with the current word
                if token_end > word_start and token_start < word_end:
                    word_tokens.append(token_idx)
                    
            if word_tokens:
                word_to_tokens[word_idx] = word_tokens

        return word_to_tokens, encoding


    def _compute_word_attributions(
        self,
        text: str,
        token_ids: Tensor,
        word_to_tokens: Dict[int, List[int]],
        baselines: BaselineType,
        target: TargetType,
        additional_forward_args: Any,
        sliding_window_shapes: int,
        show_progress: bool,
    ) -> Tensor:
        """
        Compute token-level attributions by occluding one word at a time.
        
        Args:
            token_ids (Tensor): The tokenized input.
            word_to_tokens (Dict): Mapping from word indices to token indices.
            baselines: Baseline value for occlusion.
            target: Output indices for attribution.
            additional_forward_args: Additional args for forward function.
            sliding_window_shapes: Number of perturbations per batch.
            show_progress: Whether to show progress.
            
        Returns:
            Tensor: Token-level attributions.
        """
        # If baseline is not provided, use a default (e.g., PAD token)
        if baselines is None:
            baselines = self.tokenizer.pad_token_id
            if baselines is None:  # Some tokenizers might not have pad_token_id
                baselines = 0
        
        # Compute original output
        # enc_ctx = self.model.model.preprocess_input(text)
        token_ids = {k: v for k, v in token_ids.items() if k != 'offset_mapping'}
        original_output = self.forward_func(token_ids, *additional_forward_args if additional_forward_args else []) #previously instead of text was token_ids
        if hasattr(original_output, 'logits'):
            original_output = original_output.logits

        # Prepare baselines in the right format
        if not torch.is_tensor(baselines):
            baselines = torch.tensor(
                baselines, device=token_ids['input_ids'].device, dtype=token_ids['input_ids'].dtype
            )
        
        # Expand baselines if needed
        if baselines.dim() == 0:
            baselines = baselines.expand_as(token_ids['input_ids'])
        
        # Track attributions for each token
        token_attributions = torch.zeros_like(token_ids['input_ids'], dtype=torch.float)
        
        # Process words in batches
        num_words = len(word_to_tokens)
        batch_size = sliding_window_shapes
        
        # Setup progress bar if needed
        if show_progress:
            try:
                from tqdm import tqdm
                word_iterator = tqdm(range(0, num_words, batch_size))
            except ImportError:
                word_iterator = range(0, num_words, batch_size)
                print("Processing words...")
        else:
            word_iterator = range(0, num_words, batch_size)

        for i in word_iterator:
            # Get the words for this batch
            batch_word_indices = list(range(i, min(i + batch_size, num_words)))
            batch_ablated_inputs = []
            
            for word_idx in batch_word_indices:
                # Create an ablated input by replacing tokens in the word with baseline
                token_indices = word_to_tokens[word_idx]
                
                for idx in token_indices:
                    ablated_input = token_ids["input_ids"].clone()
                    ablated_input[0, idx] = baselines[0, idx]
                    batch_ablated_inputs.append(ablated_input)
            stacked_inputs={}
            # Stack ablated inputs for batch processing
            if len(batch_ablated_inputs) > 1:
                stacked_inputs['input_ids'] = torch.cat(batch_ablated_inputs, dim=0)
            else:
                stacked_inputs['input_ids'] = batch_ablated_inputs[0]
            stacked_inputs['attention_mask']=token_ids['attention_mask']
            # Forward pass with ablated inputs
            with torch.no_grad():
                batch_outputs = self.forward_func(stacked_inputs, *additional_forward_args if additional_forward_args else [])
            
            # Compute attributions for each word in this batch
            for j, word_idx in enumerate(batch_word_indices):
                # Get output for this perturbation
                output = batch_outputs[j] if len(batch_word_indices) > 1 else batch_outputs
                if hasattr(output, 'logits'):
                    output = output.logits
                    
                # Compute attribution as difference in output
                attribution = original_output - output
                
                # If target is specified, select the target output
                if target is not None:
                    if isinstance(target, int):
                        attribution = attribution[:, target]
                    elif isinstance(target, tuple):
                        for t in target:
                            attribution = attribution[:, t]
                    else:
                        attribution = attribution[:, target]
                
                # Calculate attribution value (might need to reduce dimensions)
                if attribution.dim() > 1:
                    attribution_value = attribution.sum().item()
                else:
                    attribution_value = attribution.item()
                
                # Assign attribution to corresponding tokens
                for token_idx in word_to_tokens[word_idx]:
                    token_attributions[0, token_idx] = attribution_value
        
        return token_attributions

    def _aggregate_to_word_level(
        self, 
        token_attributions: Tensor, 
        word_to_tokens: Dict[int, List[int]], 
        method: str = "mean"
    ) -> Tensor:
        """
        Aggregate token-level attributions to word-level.
        
        Args:
            token_attributions (Tensor): Attributions for each token.
            word_to_tokens (Dict): Mapping from word indices to token indices.
            method (str): Aggregation method - "mean", "sum", or "max".
            
        Returns:
            Tensor: Attributions for each word.
        """
        word_attributions = []
        
        for word_idx in sorted(word_to_tokens.keys()):
            # Get attributions for tokens in this word
            token_indices = word_to_tokens[word_idx]
            word_token_attributions = torch.tensor([token_attributions[0, idx].item() for idx in token_indices])
            
            # Aggregate based on the specified method
            if method == "mean":
                word_attribution = torch.mean(word_token_attributions)
            elif method == "sum":
                word_attribution = torch.sum(word_token_attributions)
            elif method == "max":
                word_attribution = torch.max(word_token_attributions) # Returns (value, index)
            else:
                raise ValueError(f"Unknown aggregation method: {method}")
            
            word_attributions.append(word_attribution.item())
        
        return torch.tensor(word_attributions)
    




class Occlusion_word_level(WordLevelOcclusion):
    """
    A modification of Occlusion that operates at the word level rather than the token level.
    This implementation respects word boundaries when performing occlusion, leading to more
    interpretable attributions for text inputs.
    """

    def __init__(self, forward_func: Callable, 
        model,
        tokenizer,
        apply_normalization: bool = True, 
        sliding_window_shapes: int = 1, 
        strides: int = 1, 
        regex_condition: str = None,
        normalization_approach: Literal["min-max",'log-min-max', "l2", "max-abs",None] = None) -> None:
        """
        Args:
            forward_func (Callable): The forward function of the model.
            tokenizer: The tokenizer used by the model to convert text to tokens.
            strides: Number of perturbations per batch.
            sliding_window_shapes: Window for perturbation.
            
        """
        super().__init__(forward_func, model, tokenizer)
        self.apply_normalization=True
        self.normalization_approach=normalization_approach
        self.sliding_window_shapes=sliding_window_shapes
        self.strides=strides
        self.regex_condition=regex_condition

    def _compute_word_attributions(
        self,
        # text: List [str],
        post_claim_pair: List [Tensor],
        word_to_tokens: Dict[int, List[int]],
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        show_progress: bool = False,
        strides: int = None,
        regex_condition: str = None,
        sliding_window_shapes: int = None,
    ) -> Tensor:
        """
        Compute token-level attributions by occluding one word at a time.
        
        Args:
            token_ids (Tensor): The tokenized input.
            word_to_tokens (Dict): Mapping from word indices to token indices.
            baselines: Baseline value for occlusion.
            target: Output indices for attribution.
            additional_forward_args: Additional args for forward function.
            strides: Number of perturbations per batch.
            sliding_window_shapes: Window for perturbation.
            show_progress: Whether to show progress.
            
        Returns:
            Tensor: Token-level attributions.
        """
        sliding_window_shapes = self.sliding_window_shapes
        strides=self.strides
        try:
            # Compute original output
            token_attributions_list=[]
            for i, token_ids in enumerate(post_claim_pair):
                # If baseline is not provided, use a default (e.g., PAD token)
                tokenizer=get_tokenizer(self.model)
                if baselines is None:
                    baselines = self.tokenizer.pad_token_id
                    if baselines is None:  # Some tokenizers might not have pad_token_id
                        baselines = 0
                post_claim=post_claim_pair.copy()
                primary_tokens = token_ids
                del post_claim[i] 
                secondary_tokens = post_claim[0]
                secondary_tokens.to(utils.get_device())
                secondary_tokens['input_ids']=secondary_tokens['input_ids'].to(utils.get_device())
                primary_tokens['input_ids']=primary_tokens['input_ids'].to(utils.get_device())
                primary_tokens = {k: v for k, v in primary_tokens.items() if k != 'offset_mapping'}
                secondary_tokens = {k: v for k, v in secondary_tokens.items() if k != 'offset_mapping'}
                #cut everything under 512 
                if len(primary_tokens['input_ids'][0]) > tokenizer.model_max_length:
                    primary_tokens['input_ids']=primary_tokens['input_ids'][:, :tokenizer.model_max_length]
                    primary_tokens['attention_mask']=primary_tokens['attention_mask'][:, :tokenizer.model_max_length]
                    for k in word_to_tokens[0]:
                        word_to_tokens[0][k] = [x for x in word_to_tokens[0][k] if x <= tokenizer.model_max_length-1]
                    word_to_tokens[0] = {k: v for k, v in word_to_tokens[0].items() if v}
                    
                if len(secondary_tokens['input_ids'][0])>tokenizer.model_max_length:
                    secondary_tokens['input_ids']=secondary_tokens['input_ids'][:, :tokenizer.model_max_length]
                    secondary_tokens['attention_mask']=secondary_tokens['attention_mask'][:, :tokenizer.model_max_length]
                    for k in word_to_tokens[1]:
                        word_to_tokens[1][k] = [x for x in word_to_tokens[1][k] if x <= tokenizer.model_max_length-1]
                    word_to_tokens[1] = {k: v for k, v in word_to_tokens[1].items() if v}
                secondary_embedding = self.model.model.transformer(**secondary_tokens)[0]
                original_output = self.forward_func(primary_tokens, secondary_embedding,*additional_forward_args if additional_forward_args else []) #previously instead of text was token_ids
                if hasattr(original_output, 'logits'):
                    original_output = original_output.logits

                # Prepare baselines in the right format
                if not torch.is_tensor(baselines):
                    baselines = torch.tensor(
                        baselines, device=primary_tokens['input_ids'].device, dtype=primary_tokens['input_ids'].dtype
                    )

                # Expand baselines if needed
                if baselines.dim() == 0:
                    baselines = baselines.expand_as(primary_tokens['input_ids'])


                # Track attributions for each token
                token_attributions = torch.zeros_like(primary_tokens['input_ids'], dtype=torch.float)
                token_attributions= token_attributions.unsqueeze(-1).expand(-1, -1, secondary_embedding.size(1))
                token_attributions = token_attributions.clone()
                # Process words in batches 
                num_words = len(word_to_tokens[i])
                batch_size = sliding_window_shapes
                strides_size = strides 

                if self.regex_condition:
                    keys_found=set()
                    list_ids=[]
                    #find tokens by given string 
                    for mark in list(self.regex_condition):
                        tokenized_regex=self.tokenizer(mark)['input_ids'][1]
                        list_of_tokens= primary_tokens['input_ids'].tolist()[0]
                        indices = [i for i, x in enumerate(list_of_tokens) if x == tokenized_regex]
                        list_ids.extend(indices)
                    #find words by tokens
                    for num in list_ids:
                        for key, values in word_to_tokens[i].items():
                            if num in values:
                                keys_found.add(key)
                    keys_found= list(keys_found)
                    keys_found= [0] + keys_found + [len(word_to_tokens[i])]
                    keys_found.sort()
                    
                    


                # Setup progress bar if needed
                if show_progress and not self.regex_condition:
                    try:
                        from tqdm import tqdm
                        word_iterator = tqdm(range(0, num_words, strides_size))
                    except ImportError:
                        word_iterator = range(0, num_words, strides_size)
                        print("Processing words...")
                if not self.regex_condition:
                    word_iterator = range(0, num_words, strides_size)
                else: 
                    word_iterator = keys_found

                batch_ablated_inputs = []
                batch_id_indices= []
                try:
                    for index,i_word in enumerate(word_iterator):
                        # Get the words for this batch
                        if self.regex_condition:
                            end = min(index + batch_size, len(word_iterator)-1)
                            batch_word_indices= list(range(i_word, word_iterator[end]))                        
                        else:
                            batch_word_indices= list(range(i_word, min(i_word+batch_size, num_words)))
                        token_indices=[]
                        for word_idx in batch_word_indices:
                            # Create an ablated input by replacing tokens in the word with baseline
                            token_indices.extend(word_to_tokens[i][word_idx])
                        ablated_input = primary_tokens["input_ids"].clone()
                        for idx in token_indices:
                            ablated_input[0, idx] = baselines[0, idx]
                        batch_id_indices.append(token_indices)
                        batch_ablated_inputs.append(ablated_input)
                except Exception as e :
                    print(e)
                # Forward pass with ablated inputs
                for i_abla,ablated_input in enumerate(batch_ablated_inputs):
                        # Stack ablated inputs for batch processing
                        stacked_inputs={}
                        stacked_inputs['input_ids'] = ablated_input
                        stacked_inputs['attention_mask']=primary_tokens['attention_mask']
                        with torch.no_grad():
                            output = self.forward_func(stacked_inputs,secondary_embedding, *additional_forward_args if additional_forward_args else [])
                        if hasattr(output, 'logits'):
                            output = output.logits
                        # Compute attribution as difference in output
                        attribution = original_output - output
                        # If target is specified, select the target output
                        if target is not None:
                            if isinstance(target, int):
                                attribution = attribution[:, target]
                            elif isinstance(target, tuple):
                                for t in target:
                                    attribution = attribution[:, t]
                            else:
                                attribution = attribution[:, target]
                        
                        # Calculate attribution value (might need to reduce dimensions)
                        if attribution.dim() > 1:
                            attribution_value = attribution.sum().item()
                        else:
                            attribution_value = attribution.item()
                        # Assign attribution to corresponding tokens
                        for i_tokens in batch_id_indices[i_abla]:
                            token_attributions[0, i_tokens, :] += attribution_value
                token_attributions=self.postprocess_explanations(token_attributions)
                #delete special tokens
                # id_set=set()
                # for special_token in self.tokenizer(self.tokenizer.all_special_tokens)['input_ids']:
                #     ids=(torch.where(primary_tokens['input_ids'] == special_token[1]))
                #     for id_t in ids:
                #         if id_t.numel():
                #             print(id_t.item())
                #             id_set.add(id_t.item())
                # id_list=list(id_set)
                # if id_list:
                #     id_list.reverse()
                #     token_attributions=token_attributions.tolist()[0]
                #     for d in id_list:
                #         token_attributions.pop(d)
                #     token_attributions= torch.tensor(token_attributions).unsqueeze(dim=0)
                # add to list
                token_attributions_list.append(token_attributions)
                baselines=None
        except Exception as e:
            print(e)    
        return token_attributions_list

    def postprocess_explanations(self, expl: torch.Tensor) -> torch.Tensor:
        expl = expl.sum(dim=-1)
        if self.apply_normalization and self.normalization_approach == "min-max":
            for i in range(len(expl)):
                max_r = torch.quantile(expl, 0.95)
                min_r = torch.quantile(expl, 0.05)
                expl[i] = (expl[i] - min_r) / (max_r - min_r + 1e-9)
                expl[i] = torch.clip(expl[i], min=0, max=1)
        elif self.apply_normalization and self.normalization_approach == "l2":
            expl = expl / torch.norm(expl, p=2)
        elif self.apply_normalization and self.normalization_approach == "log-min-max":
                log_tensor = torch.log1p(expl)
                max_log = torch.quantile(expl, 0.95)
                min_log = torch.quantile(expl, 0.05)
                expl = (log_tensor - min_log) / (max_log - min_log + 1e-9)
                expl = torch.clip(expl, min=0, max=1)
        elif self.apply_normalization and self.normalization_approach == "max-abs":
            for i in range(len(expl)):
                max_abs_r = torch.quantile(expl.absolute(), 0.95)
                expl[i] = expl[i] / (max_abs_r + 1e-9)
                expl[i] = torch.clip(expl[i], min=-1, max=1)
        elif self.apply_normalization:
            return expl
        
        return expl
    
        
    def post_claim_occlusion(self,post,claim) -> List:
        post_word_to_tokens, post_encoding=Occlusion_word_level._map_words_to_tokens(self,post)
        claim_word_to_tokens, claim_encoding=Occlusion_word_level._map_words_to_tokens(self,claim)

        mapping= self._compute_word_attributions(
            [post_encoding,claim_encoding],
            [post_word_to_tokens,claim_word_to_tokens]
        )
        return mapping
