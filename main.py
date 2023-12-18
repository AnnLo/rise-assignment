import argparse
from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer, AdamW, get_linear_schedule_with_warmup, BertTokenizerFast, BertForTokenClassification
import torch
from code.tag_mappings import TagMapping
from code.dataset_preprocessing import ProcessData
from code.evaluation import Eval

def train(model_name, system, output_dir, learning_rate, batch_size, num_train_epochs, weight_decay):
    
    """
    Training function, loads model and tokenizer. Calls preprocessing etc. Runs training.
    
    Args:
        model_name (str): name of pre-trained model to be used
        system (str): determines if we are tuning system A or system B
        output_dir (str): name of output dir
        learning_rate (float): learning rate for the model
        batch_size (int): batch size for the model
        num_training_epochs (int): number of epochs to fine tuned the model
        weight_decay (float): value of weight decay (regularization)
    
    returns:
        trainer (Trainer): upon completed training, the trainer is returned for evaluation
        tokenized_dataset (): returned for evaluation
    """
    
    # original tagset
    
    label2id = {"O": 0,
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
   
    # load tokenizer, does not do lower case
    
    tokenizer = BertTokenizerFast.from_pretrained(model_name) 


    # preprocess, tokenize and pad
    
    tokenized_dataset = ProcessData(system, label2id,"Babelscape/multinerd", tokenizer).preprocess_tokenize_dataset()
      
    # padding to max sequence length, dynamically
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # update labeid mappings
    
    label2id, id2label = TagMapping(system).label2id_id2label()
    
    # check equal length of mapping dicts
    
    assert len(label2id.keys()) == len(id2label.keys()), "Should be equal length!"
    
    # list of labels and num labels 
    
    label_list = list(label2id.keys())
    num_labels = len(label2id.keys())
    
    
    # load model
    
    model = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels, label2id=label2id, id2label=id2label)
    
    # currying, def new function by applying partial arguments, for computing metrics in trainer
   
    compute_metrics = lambda predictions: Eval(label_list).metrics(predictions)


    # training args
    
    training_args = TrainingArguments(
      output_dir=output_dir,
      learning_rate=learning_rate,
      per_device_train_batch_size=batch_size,
      per_device_eval_batch_size=batch_size,
      num_train_epochs=num_train_epochs,
      weight_decay=weight_decay,
      evaluation_strategy="epoch",
      save_strategy="epoch",
      load_best_model_at_end=True,
  )

  # trainer

    trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=tokenized_dataset["train"],
      eval_dataset=tokenized_dataset["validation"],
      tokenizer=tokenizer,
      data_collator=data_collator,
      compute_metrics=compute_metrics,
  )

    # train
    
    trainer.train()

    return trainer, tokenized_dataset




def main():
    parser = argparse.ArgumentParser(description='Train and evaluate system.')
    parser.add_argument('--model', type=str, default="bert-base-cased", help="name of model")
    parser.add_argument('--system', type=str, default="A", help="Name of system we want to fine tune, A or B.")
    parser.add_argument('--output_dir', type=str, default="ner_model_dir", help="Name of save directory.")
    parser.add_argument('--learning_rate', type=float, default=3e-5, help="Learning rate.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size, training and validation the same.")
    parser.add_argument('--num_train_epochs', type=int, default=2, help="Number of epochs to fine tune.")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay.")
    
    args = parser.parse_args()
    

    trainer, tokenized_dataset = train(args.model, args.system, args.output_dir, args.learning_rate, args.batch_size, args.num_train_epochs, args.weight_decay)

    # mapping dicts and labels
    label2id, id2label = TagMapping(args.system).label2id_id2label()
    label_list = list(label2id.keys())

    
    compute_metrics_tag = lambda predictions: Eval(label_list).individual_tag_metrics(predictions)

    print("Evaluating on test set:")
    trainer.evaluate(eval_dataset=tokenized_dataset["test"])
    print()
    print("Predicting individual tags on test set:")
    test_predictions = trainer.predict(tokenized_dataset["test"])
    compute_metrics_tag(test_predictions)
    
if __name__ == "__main__":
    main()
