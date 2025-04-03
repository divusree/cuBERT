import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification

class BERTDataLoader:
    def __init__(self, tokenizer, dataset):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.label_list = dataset["train"].features["ner_tags"].feature.names  # ['O', 'B-PER', 'I-PER', 'B-ORG', ...]
        self.num_ner_labels = len(self.label_list)  #         
        self.data_collator = DataCollatorForTokenClassification(
            self.tokenizer, 
            max_length = 512,
            padding="longest",
            label_pad_token_id=-100  # Ignore padded labels in loss
        )

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"], 
            truncation=True, 
            is_split_into_words=True,  # Input is pre-split into words
            return_offsets_mapping=True
        )
        
        aligned_labels = []
        for i, (labels, offset_mapping) in enumerate(zip(examples["ner_tags"], tokenized_inputs["offset_mapping"])):
            # Map each subword to its word-level label
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # [None, 0, 0, 1, 1, None] for [CLS], word1, word2, [SEP]
            new_labels = []
            for word_idx in word_ids:
                if word_idx is None:
                    new_labels.append(-100)  # Ignore [CLS], [SEP], padding
                else:
                    new_labels.append(labels[word_idx])  # Use the label of the original word
            aligned_labels.append(new_labels)
        tokenized_inputs["labels"] = aligned_labels
        
        # Debug check
        # i = 0:  # Print first example for verification
        # print("\nTokenization debug:")
        # print("Original words:", examples["tokens"][i])
        # print("Input IDs:", tokenized_inputs["input_ids"][i])
        # print("Word IDs:", word_ids)
        # print("Labels:", tokenized_inputs["labels"][i])
        # print("Attention mask:", tokenized_inputs["attention_mask"][i])
        
        return tokenized_inputs        
    def get_dataloader(self, type = 'train', batch_size = 8):
        tokenized_ds = self.dataset.map(
                self.tokenize_and_align_labels,
                batched=True,
                remove_columns=self.dataset[type].column_names  # Remove original columns
                )
        tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        print(type, tokenized_ds.shape)
        data_loader = DataLoader(
            tokenized_ds[type],
            batch_size=batch_size,
            collate_fn=self.data_collator
        )       
        return data_loader
    