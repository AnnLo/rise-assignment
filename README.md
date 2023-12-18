## The assignment


This is my solution to the assignment. Using the English part of the MultiNERD Named Entity Recognition (NER) dataset [https://huggingface.co/datasets/Babelscape/multinerd], a BERT model (bert-base-cased [https://huggingface.co/bert-base-cased)] was fine tuned to create two systems, A and B.

### System A
Uses the full tag set as can be seen on  [https://huggingface.co/datasets/Babelscape/multinerd]

### System B
Here all but entity types belonging to [PER, ORG, LOC, DIS ANIM] were set to zero.


### Installation

1. Clone the repo

2. Install requirements
   ```sh
   pip install -r requirements.txt
   ```

2. Or, create a conda enviroment
   ```sh
  conda env create --name envname --file=environments.yml
   ```


## How to run the code

The code can either be run from the terminal, or if one prefers by running the notebook "Train and evaluate". From the terminal

 ```sh
 python main.py
 ```
with either deafault arguments or command line arguments as specified in ``main.py``, system A is the default. The notebook should simply be run top to bottom. There is additionally a notebook where I familirized myself with the dataset, called "A bit of analysis of the dataset".




