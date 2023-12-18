## The assignment


This is my solution to the assignment. Using the English part of the MultiNERD Named Entity Recognition (NER) dataset [https://huggingface.co/datasets/Babelscape/multinerd], a BERT model (bert-base-cased [https://huggingface.co/bert-base-cased)] was fine-tuned to create two systems, A and B.

### System A
Uses the full tag set as can be seen on  [https://huggingface.co/datasets/Babelscape/multinerd].

The following results were achieved when evaluating on the test set:

``{'precision': 0.9234206232030461, 'recall': 0.9267693507506336, 'f1': 0.9250919564836618, 'accuracy': 0.9869980548951804}``


### System B
Here all but entity types belonging to [PER, ORG, LOC, DIS ANIM] were set to zero.

The following results were achieved when evaluating on the test set:

``{'precision': 0.9488388335952506, 'recall': 0.9466486651278254, 'f1': 0.9477424840306089, 'accuracy': 0.9914645918881925}``

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




