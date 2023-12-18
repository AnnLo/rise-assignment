class TagMapping:
    """
    Class for named entity tag-to-id mapping, takes the original full tag set as input and either returns that and the id2label mapping or the updated versions for system B.
    
    Args:
        system (str): determines whether system A or B is to be considered.
    """
    def __init__(self, system:str):
        self.system = system
        # the original tagset:
        self.label2id = {"O": 0,
                       "B-PER": 1,
                        "I-PER": 2,
                        "B-ORG": 3,
                        "I-ORG": 4,
                        "B-LOC": 5,
                        "I-LOC": 6,
                        "B-ANIM": 7,
                        "I-ANIM": 8,
                        "B-BIO": 9,
                        "I-BIO": 10,
                        "B-CEL": 11,
                        "I-CEL": 12,
                        "B-DIS": 13,
                        "I-DIS": 14,
                        "B-EVE": 15,
                        "I-EVE": 16,
                        "B-FOOD": 17,
                        "I-FOOD": 18,
                        "B-INST": 19,
                        "I-INST": 20,
                        "B-MEDIA": 21,
                        "I-MEDIA": 22,
                        "B-MYTH": 23,
                        "I-MYTH": 24,
                        "B-PLANT": 25,
                        "I-PLANT": 26,
                        "B-TIME": 27,
                        "I-TIME": 28,
                        "B-VEHI": 29,
                        "I-VEHI": 30}
        
        
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

    def _label_id_mapping_sys_B(self):
        """
        Function that creates labe2id and id2label dict for system B.
        
        returns:
              label2id_B (dict): label to id mapping dict
              id2label_B (dict): id to label mapping dict
        """
        blabels2id, bid2label = self._helper_mapping()
        # map to proper indexing
        mapping_dict= {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 13:9, 14:10}

        label2id_B = {k:mapping_dict[v] for k, v in blabels2id.items()}
        id2label_B = {v:k for k, v in label2id_B.items()}
        return label2id_B, id2label_B
    
    def label2id_id2label(self):
        """
        Function that created the label to id and id to lable mapping dicts for system A and B.
        
        returns:
              label2id (dict): label to id mapping dict
              id2label (dict): id to label mapping dict    
        """
        assert self.system in ["A", "B"], "Not a valid choice."
        
        
        if self.system == "A":
            id2label = {v:k for k, v in self.label2id.items()}
            self.id2label = id2label
        else:
            self.label2id, self.id2label = self._label_id_mapping_sys_B()
        
        return self.label2id, self.id2label
            