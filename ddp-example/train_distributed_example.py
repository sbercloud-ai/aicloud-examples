import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import pathlib
import logging
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# Determine the base directory where script is located
BASE_DIR = str(pathlib.Path(__file__).parent.absolute())

# Setting up logging to write logs to a file
logging.basicConfig(filename=BASE_DIR + "/train.log",
                    filemode='a',  # Add mode ('a' - append)
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',  # Log message format
                    datefmt='%H:%M:%S',  # Date and time format
                    level=logging.INFO)  # Logging level
logging.info(f"Working dir: {BASE_DIR}")  # Logging information about the working directory

# Function to clean up a group of processes after shutdown
def cleanup():
    dist.destroy_process_group()

# Main function for training
def train(rank, size, local_rank):
    epochs = 10  # Number of training epochs
    MODEL_NAME = "Intel/neural-chat-7b-v3-1"
    # Configuration for quantizing the model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Download in 4-bit format
        bnb_4bit_use_double_quant=True,  # Using double quantization
        bnb_4bit_quant_type="nf4",  # Quantization type
        bnb_4bit_compute_dtype=torch.bfloat16  # Calculation data type
    )

    # Loading a pre-trained model with quantization settings
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cuda",  # Distribution of the model by CUDA devices
        trust_remote_code=True,  # Trusting remote code
        quantization_config=bnb_config  # Applying quantize configuration
    )

    # Loading a tokenizer for a model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # Setting the padding token equal to the end of line token

    model.gradient_checkpointing_enable()  # Enable gradient control points to save memory
    model = prepare_model_for_kbit_training(model)  # Preparing the model for training with kbit optimization

    # Configuration for optimization LoRA
    config = LoraConfig(
        r=8,  # Projection dimension
        lora_alpha=32,  # Multiplier for LoRA
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target modules for optimization
        lora_dropout=0.05,  # Exclusion share for LoRA
        bias="none",  # Offset Setting
        task_type="CAUSAL_LM"  # Task type
    )

    model = get_peft_model(model, config)  # Applying PEFT optimization to a model
    print(f'local rank = {local_rank}, rank = {rank}')  # Displaying information about the rank of a process
    device = torch.device(f'cuda:{local_rank}')  # Setting the device for the current process
    model.to(device)  # Moving a model to a device
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)  # Model wrapper for distributed learning

    # Loading and preparing data
    # data = load_dataset("csv", data_files=BASE_DIR + "/midjourney_prompt_dataset.csv")  # Loading a dataset
    data = load_dataset("bittu9988/mid_journey_prompts")

    # Function for generating and tokenizing requests
    def generate_and_tokenize_prompt(data_point):
        full_prompt = f"""
        <human>: {data_point["User"]}
        <assistant>: {data_point["Prompt"]}
        """.strip()  # Generating a complete request
        tokenized_full_prompt = tokenizer(full_prompt, padding=True, truncation=True)  # Request Tokenization
        return tokenized_full_prompt

    tokenized_data = data["train"].shuffle().map(generate_and_tokenize_prompt)  # Tokenization and data shuffling
    tokenized_data = tokenized_data.remove_columns(data["train"].column_names)  # Removing unnecessary columns from data
    # Settings for training
    training_args = TrainingArguments(
        per_device_train_batch_size=1,  # Batch size per device
        gradient_accumulation_steps=8,  # Gradient accumulation steps
        num_train_epochs=epochs,  # Number of training epochs
        learning_rate=2e-4,  # Learning rate
        fp16=True,  # Using 16-bit floating points
        save_total_limit=3,  # Limit on the number of model saves
        logging_steps=1,  # Logging steps
        output_dir="experiments",  # Output directory
        optim="paged_adamw_8bit",  # Optimizer
        lr_scheduler_type="cosine",  # Learning Rate Scheduler Type
        warmup_ratio=0.05,  # Heating proportion
        remove_unused_columns=False,  # Removing unused columns
        local_rank=local_rank  # Local rank
    )

    # Initializing the trainer to train the model
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_data,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()  # Start training
    torch.save(model.state_dict(), BASE_DIR + f"/model.bin")  # Saving the trained model
    cleanup()  # Resource Cleanup

# Initializing Processes for Distributed Learning
def init_processes(fn, local_rank, backend='nccl'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend)  # Initializing a process group
    fn(dist.get_rank(), dist.get_world_size(), local_rank)  # Launching the learning function

# Entry point to the program
if __name__ == "__main__":
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])  # Getting local rank from environment variables
    init_processes(train, LOCAL_RANK, backend='nccl')  # Initializing processes with a given backend