from curtsies.fmtfuncs import red, green
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
from transformers import Trainer, TrainingArguments
import torch

class GPT2Trainer:
    def __init__(self, train_base=False):
        self.train_base = train_base

        # Check if GPU is available and set the device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("Using device: CPU")

    def train(self):
        paths = ["data.txt"]

        if self.train_base:
            self.train_tokenizer(paths)

        # Initialize tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('tokenizer')

        tokenizer.add_special_tokens({
            "eos_token": "</s>",
            "bos_token": "<s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "mask_token": "<mask>"
        })

        # Initialize GPT-2 model configuration
        config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        model = GPT2LMHeadModel(config)
        model.to(self.device)  # Move model to the GPU if available

        # Load dataset
        dataset = load_dataset("text", data_files=paths)

        # Function to encode data
        def encode(lines):
            return tokenizer(lines['text'], add_special_tokens=True, truncation=True, max_length=512)

        dataset = dataset.map(encode, batched=True)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

        # Split the dataset into train and test (if necessary)
        train_size = int(0.9 * len(dataset['train']))
        test_size = len(dataset['train']) - train_size
        train_dataset, test_dataset = dataset['train'].train_test_split(test_size=test_size).values()

        # Data collator for dynamic padding
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

        # Training arguments
        training_args = TrainingArguments(
            output_dir="GPyT",
            overwrite_output_dir=True,
            num_train_epochs=2,
            per_device_train_batch_size=1,  # Adjust batch size to fit your device
            gradient_accumulation_steps=8,  # Accumulate gradients to simulate a larger batch size without increasing memory usage
            save_steps=100,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            fp16=torch.cuda.is_available(),  # Enable mixed precision training if supported (if GPU is available)
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=test_dataset
        )

        trainer.train()
        trainer.save_model("GPyT")

    def train_tokenizer(self, paths):
        tokenizer = ByteLevelBPETokenizer()

        # Training the Tokenizer
        tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ])
        # Save files to disk
        tokenizer.save_model("tokenizer")
