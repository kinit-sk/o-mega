from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class Explanation:
    """Generic class to represent an Explanation"""

    input_text: str
    ctx_text: str
    scores: np.array

@dataclass
class ExplanationWithRationale(Explanation):
    """Specific explanation to contain the gold rationale"""

    rationale: np.array


@dataclass
class Evaluation:
    """Generic class to represent an Evaluation"""

    name: str
    score: float


@dataclass
class ExplanationEvaluation:
    """Generic class to represent an Evaluation"""

    explanation: Explanation
    evaluation_scores: list[Evaluation]


class ModelHelper:
    """
    Wrapper class to interface with HuggingFace models
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


class BaseEvaluator(ABC):

    INIT_VALUE = 0

    @property
    @abstractmethod
    def NAME(self):
        pass

    @property
    @abstractmethod
    def SHORT_NAME(self):
        pass

    @property
    @abstractmethod
    def HIGHER_IS_BETTER(self):
        # True: the higher the better
        # False: the lower the better
        pass

    @property
    @abstractmethod
    def TYPE_METRIC(self):
        # plausibility
        # faithfulness
        pass

    @property
    @abstractmethod
    def RANGE(self):
        pass

    @property
    def tokenizer(self):
        return self.helper.tokenizer

    def __init__(self, model, tokenizer):
        self.helper = ModelHelper(model, tokenizer)
        
    def aggregate_score(self, score, total, **aggregation_args):
        return score / total


def _check_and_define_get_id_discrete_rationale_function(based_on):
    if based_on == "th":
        get_discrete_rationale_function = _get_id_tokens_greater_th
    elif based_on == "k":
        get_discrete_rationale_function = _get_id_tokens_top_k
    elif based_on == "perc":
        get_discrete_rationale_function = _get_id_tokens_percentage
    else:
        raise ValueError(f"{based_on} type not supported. Specify th, k or perc.")
    return get_discrete_rationale_function


def _get_id_tokens_greater_th(soft_score_explanation, th, only_pos=None):
    id_top = np.where(soft_score_explanation > th)[0]
    return id_top


def _get_id_tokens_top_k(soft_score_explanation, k, only_pos=True):
    if only_pos:
        id_top_k = [
            i
            for i in np.array(soft_score_explanation).argsort()[-k:][::-1]
            if soft_score_explanation[i] > 0
        ]
    else:
        id_top_k = np.array(soft_score_explanation).argsort()[-k:][::-1].tolist()
    # None if we take no token
    if id_top_k == []:
        return None
    return id_top_k


def divide_text_to_sentences(input_text:str) -> list:
    start=0
    adjusted_sentence=[]
    for number,word in enumerate(input_text):
        if any(p in word for p in ".,!?;:…"): #If punctuation is in string or it is last string
            adjusted_sentence.append(''.join(input_text[start:number+1]))
            start=number+1
    return adjusted_sentence

def _get_id_tokens_percentage(soft_score_explanation, percentage, only_pos=True):
    if only_pos: 
        percentage_base = np.sum(soft_score_explanation > 0)
    else:
        percentage_base= len(soft_score_explanation)

    v = int(percentage * percentage_base)
    # Only if we remove at least instance. TBD
    if v > 0 and v <= len(soft_score_explanation):
        return _get_id_tokens_top_k(soft_score_explanation, v, only_pos=only_pos)
    else:
        return None
    

def parse_evaluator_args(evaluator_args):
    # Default parameters

    # We omit the scores [CLS] and [SEP]
    remove_first_last = evaluator_args.get("remove_first_last", True)

    # As a default, we consider in the rationale only the terms influencing positively the prediction
    only_pos = evaluator_args.get("only_pos", True)

    removal_args_input = evaluator_args.get("removal_args", None)

    # As a default, we remove from 10% to 100% of the tokens.
    removal_args = {
        "remove_tokens": True,
        "based_on": "perc",
        "thresholds": np.arange(0.1, 1.1, 0.1),
    }

    if removal_args_input:
        removal_args.update(removal_args_input)

    # Top k tokens to be considered for the hard evaluation of plausibility
    # This is typically set as the average size of human rationales
    top_k_hard_rationale = evaluator_args.get("top_k_rationale", 5)

    return remove_first_last, only_pos, removal_args, top_k_hard_rationale


def get_discrete_explanation_topK(score_explanation:np.ndarray, topK:int, only_pos=False):

    # Indexes in the top k. If only pos is true, we only consider scores>0
    topk_indices = _get_id_tokens_top_k(score_explanation, topK, only_pos=only_pos)

    # Return default score
    if topk_indices is None:
        return None

    # topk_score_explanations: one hot encoding: 1 if the token is in the rationale, 0 otherwise
    # i hate you [0, 1, 1]

    topk_score_explanations = [
        1 if i in topk_indices else 0 for i in range(len(score_explanation))
    ]
    return topk_score_explanations

def get_discreate_explanation_top_rationales(score_explanation:np.ndarray,human_rationale,binarize:bool,only_pos=False):
    """
    Args: score_explanation: non_binarized explanation, human_rationale: binarized human annotation
    Returns: score_explanation: binarized explanation based on length human_annotation. 
    """
    # Localization Understandability (we take into account amount of text underlie annotators)
    # if not isinstance(human_rationale, np.ndarray):
    #     human_rationale =np.array(human_rationale)
    # sum_rationale=np.sum(human_rationale == 1)
    # topk_indices = _get_id_tokens_top_k(score_explanation, sum_rationale, only_pos=only_pos)


    # Took only positive as topK_indices 
    topk_indices = np.where(score_explanation > 0)[0].tolist()


    if topk_indices is None:
        return None
    if binarize == True:
        topk_score_explanations = [
            1 if i in topk_indices else 0 for i in range(len((score_explanation)))
        ]
    else:
        topk_score_explanations = [
            score_explanation[i] if i in topk_indices else 0 for i in range(len(score_explanation))
        ]

    return topk_score_explanations