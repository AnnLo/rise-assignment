## The assignment


This is my solution to the assignment. Using the English part of the MultiNERD Named Entity Recognition (NER) dataset [https://huggingface.co/datasets/Babelscape/multinerd], a BERT model (bert-base-cased [https://huggingface.co/bert-base-cased)] was fine-tuned to create two systems, A and B. Both systems were fine-tuned for two epochs.

### System A
Uses the full tag set as can be seen on  [https://huggingface.co/datasets/Babelscape/multinerd].

The following results were achieved when evaluating on the test set:

``{'precision': 0.9242465487069803, 'recall': 0.9267693507506336, 'f1': 0.9255062305295949, 'accuracy': 0.9870412794467257}``

And tag level evaluation plus averages:
```
        precision    recall  f1-score   support

        ANIM       0.69      0.68      0.69      3236
         BIO       0.50      0.44      0.47        18
         CEL       0.73      0.73      0.73        82
         DIS       0.67      0.70      0.68      1458
         EVE       0.96      0.95      0.96       714
        FOOD       0.56      0.63      0.59      1006
        INST       0.67      0.53      0.59        30
         LOC       0.98      0.98      0.98     24016
       MEDIA       0.96      0.93      0.94       944
        MYTH       0.88      0.82      0.85        68
         ORG       0.97      0.97      0.97      6638
         PER       0.98      0.98      0.98     10564
       PLANT       0.64      0.57      0.60      2002
        TIME       0.83      0.82      0.82       588
        VEHI       0.78      0.76      0.77        66

   micro avg       0.93      0.92      0.93     51430
   macro avg       0.79      0.77      0.78     51430
weighted avg       0.93      0.92      0.93     51430
```



### System B
Here all but entity types belonging to [PER, ORG, LOC, DIS ANIM] were set to zero.

The following results were achieved when evaluating on the test set:

``{'precision': 0.9453199825859817, 'recall': 0.9456905187056313, 'f1': 0.945505214343254, 'accuracy': 0.9912196527627692}``

And tag level evaluation plus averages:
```
              precision    recall  f1-score   support

        ANIM       0.68      0.68      0.68      3228
         DIS       0.68      0.66      0.67      1564
         LOC       0.98      0.98      0.98     23956
         ORG       0.97      0.97      0.97      6638
         PER       0.98      0.98      0.98     10554

   micro avg       0.95      0.95      0.95     45940
   macro avg       0.86      0.85      0.85     45940
weighted avg       0.95      0.95      0.95     45940

```

### Installation

1. Clone the repo.

2. Install requirements. Or, create a conda enviroment.
   ``` sh
   conda env create --name envname --file=environments.yml
   ```
   ```sh
   pip install -r requirements.txt
   ```



## How to run the code

The code can either be run from the terminal, or if one prefers by running the notebook "Train and evaluate". From the terminal

 ```sh
 python main.py
 ```
with either deafault arguments or command line arguments as specified in ``main.py``, system A is the default. The notebook should simply be run top to bottom. There is additionally a notebook where I familirized myself with the dataset, called "A bit of analysis of the dataset".




