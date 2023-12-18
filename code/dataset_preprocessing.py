from datasets import load_dataset, DatasetDict
from transformers import BertTokenizerFast

class ProcessData:
    """
    Class for loading, preprocessing and tokenizing the dataset. Aligns named entity labels with tokenized words, which uses word_ids. Therefore, the tokenizer must be a PreTrainedTokenizerFast.
    
    Args:
        system (str): determines which system, A or B, to be tuned
        label2id (dict): mapping between full tag set and ids in dataset
        huggingface_dataset (str): name of huggingface dataset used for tuning
        tokenizer (BertTokenizerFast): pre-trained tokenizer
    """
    def __init__(self, system:str, label2id:dict, huggingface_dataset:str, tokenizer:BertTokenizerFast):
        self.system = system
        self.label2id = label2id
        self.huggingface_datset = huggingface_dataset
        self.label_list = list(self.label2id.keys())
        self.tokenizer = tokenizer
        
    def _handle_labels_subword_tokens(self, word_ids, labels):
        """
        Function that aligns named entities with the tokenized words, since (here) a WordPiece tokenizer splits unknow words into subword units. According to the original BERT-paper https://arxiv.org/pdf/1810.04805.pdf, in this case the representation of the first subtoken is used. To not add more named entities, dummy label for the rest that are not accounted for when calculating the loss. Cross entropy loss in PyTorch ignores index -100  https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html, also tokens with a word_id 'None' will be set to -100 to be automatically and thus ignored. Heavily inspired by https://huggingface.co/docs/transformers/v4.18.0/en/tasks/token_classification.
        
        Args:
            word_ids (list): list of word_ids for the tokenized input
            labels (list): named entity labels corresponding to the input
            
        returns:
            new_labels (list): updated named entity labels aligned with the tokenized input
        """
        new_labels = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id == None:
                # special token
                new_labels.append(-100)
            elif prev_word_id != word_id:
                # not subwords only, label the first token of any given word
                new_labels.append(labels[word_id])

            else:
              # subwords
              # the label for the word has already been appended
              # now append -100
                 new_labels.append(-100)
            prev_word_id = word_id
        return new_labels


    def _tokenize_and_align(self, batch):
        """
        Function that tokenized a batch and then applies alignment.
        
        Args:
            batch (list): list of list of input tokens
        
        returns:
              tokenized_batch (): a tokenized and aligned input batch
        """
        tokenized_batch = self.tokenizer(batch["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, labels_batch in enumerate(batch["ner_tags"]):
            word_ids_batch = tokenized_batch.word_ids(i)
            new_labels_batch = self._handle_labels_subword_tokens(word_ids_batch, labels_batch)
            labels.append(labels_batch)

        tokenized_batch["labels"] = labels

        return tokenized_batch

    def _helper_mapping(self):
        """
        Function that creates label mappings for the desired format in system B. Here we are only interested in the entities containing 'PER', 'LOC', 'DIS', 'ANIM', 'ORG, now all others should be set to 0/O.
        
        returns:
              blabels2id (dict): dictionary mapping between labels and ids
              bid2label (dict): dictionary mapping between ids and labels
        """
        keys = ["O"]
        values = [0]
        for k, v in self.label2id.items():
            if k.split('-')[-1] in ['PER', 'LOC', 'DIS', 'ANIM', 'ORG']:
                if not v in values:
                    keys.append(k)
                    values.append(v)

        blabels2id = {k: v for k, v in zip(keys, values)}
        bid2label = {v: k for k, v in zip(keys, values)}

        return blabels2id, bid2label



    def _map_ner_tags_B(self, example):
        """
        Function that maps the dataset to the setting required for system B.
        
        Args:
            example (): example from the dataset
            
        returns:
            example (): example with updated named entity tags
        """
        blabels2id, bid2label = self._helper_mapping()
        ner_tags = example["ner_tags"]
        mapping_dict= {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 13:9, 14:10}
        new_ner_tags = [mapping_dict[elem] for elem in [0 if not tag in list(bid2label.keys()) else tag for tag in ner_tags]]
        example["ner_tags"] = new_ner_tags
        return example
    
    
    def preprocess_tokenize_dataset(self):
        """
        Function that preprocesses, filters and tokenizes the dataset.
        
        returns:
              self.tokenized_dataset (batch datasetdict?): tokenized dataset
        """
        self.dataset = load_dataset(self.huggingface_datset)
        assert isinstance(self.dataset, DatasetDict), "Not the right type."
        assert list(self.dataset.keys()) == ["train", "validation", "test"], "All splits should be present."
        assert "lang" in self.dataset["train"].features.keys() and "lang" in self.dataset["validation"].features.keys() and "lang" in self.dataset["test"].features.keys(), "Language must be a feature."
        
        all_languages = self.dataset["train"]["lang"]
        all_languages.extend(self.dataset["validation"]["lang"])
        all_languages.extend(self.dataset["test"]["lang"])
        assert "en"  in set(all_languages), "English must be one of the langauges."
        
        # filter out the english part of the dataset
        dataset_en =  self.dataset.filter(lambda example: example["lang"] == "en")
        
        # maybe add this? See if it makes any difference...
        #class_labels = list(self.id2label.values())
        #for ds in self.data_split:
        #    features = self.dataset[ds].features.copy()
        #    features["ner_tags"] = Sequence(feature=ClassLabel(names=class_labels))
        #    self.dataset[ds] = self.dataset[ds].map(features=features)
        
        assert self.system in ["A", "B"], "Not a valid choice."
        
        if self.system == "A":
            # tokenize
            self.tokenized_dataset = dataset_en.map(self._tokenize_and_align, batched=True)
        else:
            self.tokenized_dataset = dataset_en.map(self._map_ner_tags_B).map(self._tokenize_and_align, batched=True)
            
        return self.tokenized_dataset
            
        
        
