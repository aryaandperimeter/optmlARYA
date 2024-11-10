# lora_bert_training.py

# Import necessary libraries
import torch
import numpy as np
from transformers import RobertaForSequenceClassification, RobertaTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import loralib as lora
import evaluate

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained RoBERTa model and tokenizer
model_name = "roberta-base"
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=3)
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model.to(device)

# Load and preprocess the GLUE MNLI dataset
def load_and_preprocess_data():
    # Load the dataset
    datasets = load_dataset("glue", "mnli")

    # Tokenize the dataset
    def preprocess_function(examples):
        return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding="max_length", max_length=128)

    # Map the preprocessing function to the dataset
    encoded_datasets = datasets.map(preprocess_function, batched=True)
    encoded_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return encoded_datasets

encoded_datasets = load_and_preprocess_data()

# Apply LoRA (Low-Rank Adaptation)
def apply_lora(model):
    # Modify only query (W_q) and value (W_v) matrices in each attention layer
    for layer in model.roberta.encoder.layer:
        layer.attention.self.query = lora.Linear(layer.attention.self.query.in_features, layer.attention.self.query.out_features, r=16)
        layer.attention.self.value = lora.Linear(layer.attention.self.value.in_features, layer.attention.self.value.out_features, r=16)
    # Freeze all other model parameters except LoRA parameters
    lora.mark_only_lora_as_trainable(model)

apply_lora(model)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    save_steps=500
)

# Load the evaluation metric
metric = evaluate.load("glue", "mnli")

# Compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Convert logits to PyTorch Tensor if they are a numpy array
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)

    # Calculate predictions using torch.argmax
    predictions = torch.argmax(logits, dim=-1)

    # Ensure predictions and labels are converted to numpy arrays for metric computation
    predictions = predictions.numpy() if predictions.requires_grad else predictions.detach().cpu().numpy()
    labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels

    return metric.compute(predictions=predictions, references=labels)

# Initialize and start the training process using the Hugging Face Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_datasets["train"],
    eval_dataset=encoded_datasets["validation_matched"],
    compute_metrics=compute_metrics
)

# Train the model
if __name__ == "__main__":
    trainer.train()
