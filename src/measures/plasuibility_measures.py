import torch
import numpy as np
from sklearn.metrics import auc, precision_recall_curve

from .utils import ExplanationWithRationale, BaseEvaluator, Evaluation
from .utils import (
    get_discrete_explanation_topK,
    parse_evaluator_args,
    get_discreate_explanation_top_rationales
)


class AUPRC_PlausibilityEvaluation(BaseEvaluator):
    NAME = "AUPRC_soft_plausibility"
    SHORT_NAME = "auprc_plau"
    # Higher is better
    HIGHER_IS_BETTER = True
    RANGE = (0, 1)
    TYPE_METRIC = "plausibility"

    def _compute_auprc_soft_scoring(self, true_rationale, soft_scores):
        try:     
            precision, recall, _ = precision_recall_curve(true_rationale, soft_scores)
        except:
            if -float('inf') in soft_scores or float('inf') in soft_scores:
                soft_scores = [0 if x in [float('-inf'), float('inf')] else x for x in soft_scores]
                print('LOL')
                precision, recall, _ = precision_recall_curve(true_rationale, soft_scores)
        auc_score = auc(recall, precision)
        return auc_score

    def compute_evaluation(self, explanation_with_rationale: ExplanationWithRationale,word_evaluation, sentence_evaluation,task, normalize=False, **evaluation_args):
        # if isinstance(explanation_with_rationale, ExplanationWithRationale) == False:
        #     return None
        _, only_pos, _, _ = parse_evaluator_args(evaluation_args)

        score_explanation = explanation_with_rationale.scores#[:-1]
        human_rationale = explanation_with_rationale.rationale#[:-1]

        score_explanation = get_discreate_explanation_top_rationales(
            score_explanation,human_rationale,only_pos=only_pos,binarize=True
        )

        # if only_pos:
        #     score_explanation = [v if v > 0 else 0 for v in score_explanation]
        

            
        auprc_soft_plausibility=0
        if score_explanation is not None and not all(x == 0 for x in score_explanation):
            if len(human_rationale) == len(score_explanation):
                auprc_soft_plausibility = self._compute_auprc_soft_scoring(
                    human_rationale, score_explanation
                )
        
        evaluation_output = Evaluation(self.SHORT_NAME, auprc_soft_plausibility)
        
        if normalize:
            sc = evaluation_output.score
            evaluation_output.score = (
                (sc - self.RANGE[0]) 
                / (self.RANGE[1] - self.RANGE[0])
            )
            if self.HIGHER_IS_BETTER == False:
                evaluation_output.score = 1 - evaluation_output.score

        evaluation_output.score = torch.tensor(evaluation_output.score)
        return evaluation_output


class Tokenf1_PlausibilityEvaluation(BaseEvaluator):
    NAME = "token_f1_hard_plausibility"
    SHORT_NAME = "token_f1_plau"
    HIGHER_IS_BETTER = True
    RANGE = (0, 1)
    TYPE_METRIC = "plausibility"
    INIT_VALUE = np.zeros(6)

    def _instance_tp_pos_pred_pos(self, true_expl, pred_expl):
        true_expl = np.array(true_expl)
        pred_expl = np.array(pred_expl)
        if true_expl.shape[0] != pred_expl.shape[0]:
            return 0,0,0
        assert true_expl.shape[0] == pred_expl.shape[0]

        tp = (true_expl & pred_expl).sum()
        pos = (true_expl).sum()
        pred_pos = (pred_expl).sum()

        """
        Alternative, in the case the rationales are representate by the positional id
        e.g., "i hate you" --> [1,2]
        
        true_expl = set(true_expl)
        pred_expl = set(pred_expl)

        tp =  len(true_expl & pred_expl)
        pos = len(true_expl)
        pred_pos = len(pred_expl)
        """
        return tp, pos, pred_pos

    def _precision_recall_fmeasure(self, tp, positive, pred_positive):
        if tp == 0 and positive == 0 and pred_positive == 0:
            return 0, 0, 0
        precision = tp / pred_positive
        recall = tp / positive
        fmeasure = self._f1(precision, recall)
        return precision, recall, fmeasure

    def _f1(self, _p, _r):
        if _p == 0 or _r == 0:
            return 0
        return 2 * _p * _r / (_p + _r)

    def _score_hard_rationale_predictions_dataset(self, list_true_expl, list_pred_expl):

        """Computes instance micro/macro averaged F1s
        ERASER: https://github.com/jayded/eraserbenchmark/blob/36467f1662812cbd4fbdd66879946cd7338e08ec/rationale_benchmark/metrics.py#L168

        """

        """ Each explanations is provided as one hot encoding --> True if the word is in the explanation, False otherwise
        I hate you --> --> [0, 1, 1]
        One for each instance.
        """
        tot_tp, tot_pos, tot_pred_pos = 0, 0, 0
        macro_prec_sum, macro_rec_sum, macro_f1_sum = 0, 0, 0

        for true_expl, pred_expl in zip(list_true_expl, list_pred_expl):
            tp, pos, pred_pos = self._instance_tp_pos_pred_pos(true_expl, pred_expl)

            instance_prec, instance_rec, instance_f1 = self._precision_recall_fmeasure(
                tp, pos, pred_pos
            )
            if instance_prec == 0 and instance_rec == 0 and instance_f1 == 0:
                continue
            # Update for macro computation
            macro_prec_sum += instance_prec
            macro_rec_sum += instance_rec
            macro_f1_sum += instance_f1

            # Update for micro computation
            tot_tp += tp
            tot_pos += pos
            tot_pred_pos += pred_pos

        # Macro computation

        n_explanations = len(list_true_expl)
        macro = {
            "p": macro_prec_sum / n_explanations,
            "r": macro_rec_sum / n_explanations,
            "f1": macro_f1_sum / n_explanations,
        }

        # Micro computation

        micro_prec, micro_rec, micro_f1 = self._precision_recall_fmeasure(
            tot_tp, tot_pos, tot_pred_pos
        )
        micro = {"p": micro_prec, "r": micro_rec, "f1": micro_f1}

        return {"micro": micro, "macro": macro}

    def _score_hard_rationale_predictions_accumulate(self, true_expl, pred_expl):

        """Computes instance micro/macro averaged F1s
        ERASER: https://github.com/jayded/eraserbenchmark/blob/36467f1662812cbd4fbdd66879946cd7338e08ec/rationale_benchmark/metrics.py#L168

        """

        """ Each explanations is provided as one hot encoding --> True if the word is in the explanation, False otherwise
        I hate you --> --> [0, 1, 1]
        One for each instance.
        """

        # For macro computation
        tp, pos, pred_pos = self._instance_tp_pos_pred_pos(true_expl, pred_expl)

        # For micro computation
        instance_prec, instance_rec, instance_f1 = self._precision_recall_fmeasure(
            tp, pos, pred_pos
        )
        return instance_prec, instance_rec, instance_f1, tp, pos, pred_pos

    def compute_evaluation(self, explanation_with_rationale: ExplanationWithRationale,word_evaluation,sentence_evaluation,task, normalize=False, **evaluation_args):

        """Evaluate an explanation on the Token-f1 score Plausibility metric.

        Args:
            explanation (ExplanationWithRationale): the explanation to evaluate
            evaluation_args (dict):  additional evaluation args

        Returns:
            Evaluation : the Token-f1 Plausibility score of the explanation
        """

        # if isinstance(explanation_with_rationale, ExplanationWithRationale) == False:
        #     return None

        # # Token fpr - hard rationale predictions. token-level F1 scores
        _, only_pos, _, _ = parse_evaluator_args(
            evaluation_args
        )
        accumulate_result = evaluation_args.get("accumulate_result", False)

        score_explanation = explanation_with_rationale.scores#[:-1]
        human_rationale = explanation_with_rationale.rationale#[:-1]

        topk_score_explanations = get_discreate_explanation_top_rationales(
            score_explanation,human_rationale,only_pos=only_pos,binarize=True
        )

        if topk_score_explanations is None and all(x == 0 for x in score_explanation):
            # Return default scores
            if accumulate_result:
                raise # 
                # return Evaluation(self.SHORT_NAME, [0, 0, 0, 0, 0, 0])
            else:
                sc = torch.tensor(0, dtype=torch.float32)
                return Evaluation(self.SHORT_NAME, sc)

        tp, pos, pred_pos = self._instance_tp_pos_pred_pos(
            human_rationale, topk_score_explanations
        )
        if tp == 0 and pos == 0 and pred_pos==0:
            evaluation_output = Evaluation(self.SHORT_NAME, torch.tensor(1, ))
            return evaluation_output
        (
            instance_prec,
            instance_rec,
            instance_f1_micro,
        ) = self._precision_recall_fmeasure(tp, pos, pred_pos)

        if accumulate_result:

            output_score = np.array(
                [tp, pos, pred_pos, instance_prec, instance_rec, instance_f1_micro]
            )

            evaluation_output = Evaluation(self.SHORT_NAME, output_score)
        else:
            evaluation_output = Evaluation(self.SHORT_NAME, instance_f1_micro)

        if normalize:
            sc = evaluation_output.score
            evaluation_output.score = (
                (sc - self.RANGE[0]) 
                / (self.RANGE[1] - self.RANGE[0])
            )
            if self.HIGHER_IS_BETTER == False:
                evaluation_output.score = 1 - evaluation_output.score
        
        evaluation_output.score = torch.tensor(evaluation_output.score)
        return evaluation_output

    def aggregate_score(self, score, total, **aggregation_args):
        average = aggregation_args.get("average", "macro")
        (
            total_tp,
            total_pos,
            total_pred_pos,
            macro_prec_sum,
            macro_rec_sum,
            macro_f1_sum,
        ) = tuple(score)

        # Macro computation
        macro = {
            "p": macro_prec_sum / total,
            "r": macro_rec_sum / total,
            "f1": macro_f1_sum / total,
        }

        # Micro computation

        micro_prec, micro_rec, micro_f1 = self._precision_recall_fmeasure(
            total_tp, total_pos, total_pred_pos
        )
        micro = {"p": micro_prec, "r": micro_rec, "f1": micro_f1}

        if average == "macro":
            return macro["f1"]
        elif average == "micro":
            return micro["f1"]
        else:
            raise ValueError()


class TokenIOU_PlausibilityEvaluation(BaseEvaluator):
    NAME = "token_IOU_hard_plausibility"
    SHORT_NAME = "token_iou_plau"
    HIGHER_IS_BETTER = True
    RANGE = (0, 1)
    TYPE_METRIC = "plausibility"

    def _token_iou(self, true_expl, pred_expl):
        """From ERASER
        We define IOU on a token level:  for two spans,
            it is the size of the overlap of the tokens they cover divided by the size of their union.
        """

        if type(true_expl) is list:
            true_expl = np.array(true_expl)
        if type(pred_expl) is list:
            pred_expl = np.array(pred_expl)
        if true_expl.shape[0] != pred_expl.shape[0]:
            return None
        assert true_expl.shape[0] == pred_expl.shape[0]

        num = (true_expl & pred_expl).sum()
        denom = (true_expl | pred_expl).sum()

        iou = 0 if denom == 0 else num / denom
        return iou

    def compute_evaluation(self, explanation_with_rationale: ExplanationWithRationale,word_evaluation,sentence_evaluation,task, normalize=False, **evaluation_args):

        """Evaluate an explanation on the Intersection Over Union (IOU) Plausibility metric.

        Args:
            explanation (ExplanationWithRationale): the explanation to evaluate
            evaluation_args (dict):  additional evaluation args

        Returns:
            Evaluation : the IOU Plausibility score of the explanation
        """

        """From ERASER
        'We define IOU on a token level:  for two spans,
        it is the size of the overlap of the tokens they cover divided by the size of their union.''

        Same process as in _token_f1_hard_rationales
        rationale: one hot encoding of the rationale
        soft_score_explanation: soft scores, len = #tokens, floats
        """

        # if isinstance(explanation_with_rationale, ExplanationWithRationale) == False:
        #     return None

        _, only_pos, _, _ = parse_evaluator_args(
            evaluation_args
        )

        score_explanation = explanation_with_rationale.scores#[:-1]
        human_rationale = explanation_with_rationale.rationale#[:-1]

        topk_score_explanations = get_discreate_explanation_top_rationales(
            score_explanation,human_rationale, only_pos=only_pos,binarize=True
        )
        if topk_score_explanations is None and all(x == 0 for x in score_explanation):
            # Return default scores
            sc = torch.tensor(0, dtype=torch.float32)
            return Evaluation(self.SHORT_NAME, sc)

        token_iou = self._token_iou(human_rationale, topk_score_explanations)
        if token_iou == None:
            evaluation_output = Evaluation(self.SHORT_NAME, torch.tensor(0,dtype=torch.float32))
            return evaluation_output
        evaluation_output = Evaluation(self.SHORT_NAME, token_iou)
        
        if normalize:
            sc = evaluation_output.score
            evaluation_output.score = (
                (sc - self.RANGE[0]) 
                / (self.RANGE[1] - self.RANGE[0])
            )
            if self.HIGHER_IS_BETTER == False:
                evaluation_output.score = 1 - evaluation_output.score

        evaluation_output.score = torch.tensor(evaluation_output.score)
        return evaluation_output
