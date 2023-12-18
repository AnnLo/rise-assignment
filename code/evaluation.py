import numpy as np
import evaluate
from seqeval.metrics import classification_report
import json
from typing import Optional

seqeval = evaluate.load("seqeval")

class Eval:
    """
    A class for calculating evaluation metrics. Uses seqeval https://huggingface.co/spaces/evaluate-metric/seqeval, a python framework for
    sequence labeling evaluation. 
    
    Args: 
        label_list (list): a list of named entity tags (strings format).
        file_save (Optional[str]): optional file name to save output
    """
    def __init__(self, label_list:list):
        self.lable_list = label_list
        
    def metrics(self, eval_predictions):
        """
        Function calculating overall classification metrics on an evaluation dataset. Heavliy inspired by tutorial on token classification https://huggingface.co/docs/transformers/v4.18.0/en/tasks/token_classification and https://huggingface.co/spaces/evaluate-metric/seqeval.
        
        Args:
            eval_predictions (Tuple): the model predictions (logits, and labels) on an evaluation dataset.
        """
        logits, labels = eval_predictions
        predictions = np.argmax(logits, axis=-1)

        # predictions when not ignore
        # indexes for the label list correspond to predicted labels for non-ignore tokens
        
        actual_predictions = [[self.lable_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        actual_labels = [[self.lable_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        results = seqeval.compute(predictions=actual_predictions, references=actual_labels)

        result_dict = {"precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],}
        
        print(result_dict)
        
        return result_dict
                
          


    def individual_tag_metrics(self, eval_predictions):
        """
        Function calculating per tag classification metrics and averages.
        
        Args:
            test_prediction (Tuple): the model predictions (logits, and labels) on an evaluation dataset.
        """

        logits, labels, _ = eval_predictions # obs, also returns metrics apparently
        predictions = np.argmax(logits, axis=-1)

        # predictions when not ignore
        # indexes for the label list correspond to predicted labels for non-ignore tokens
        
        actual_predictions = [[self.lable_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        actual_labels = [[self.lable_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

        print(classification_report(actual_predictions, actual_labels))

        
