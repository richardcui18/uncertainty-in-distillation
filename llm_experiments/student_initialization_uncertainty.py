import os
import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorWithPadding, TrainerCallback
from datasets import Dataset
import argparse
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

# Argument parser
parser = argparse.ArgumentParser(description="Student Initialization Uncertainty Script")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size for generation")
parser.add_argument("--percent_noise", type=float, default=0.01, help="Percent noise for student initialization")
parser.add_argument("--teacher_path", type=str, default="", help="Teacher model path")
parser.add_argument("--student_out_dir", type=str, default="", help="Student output path")
args = parser.parse_args()

batch_size = args.batch_size
percent_noise = args.percent_noise
teacher_model_path = args.teacher_path
output_dir = args.student_out_dir

# Settings
base_seed = 1
n = 20  # number of student models
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
max_len = tokenizer.model_max_length
training_data_path = "../data/training/training_data.json"

# Load teacher model and tokenizer
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_path)
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
teacher_tokenizer.padding_side = "left"
teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
teacher_model.to("cuda")
teacher_model.eval()

# Load sentence transformer
sent_transformer = SentenceTransformer("all-MiniLM-L6-v2")

# Load training data
with open(training_data_path) as f:
    raw_data = json.load(f)

# Helper functions
def collect_param_vector(model):
    """Flatten model params into one vector."""
    return torch.cat([p.detach().cpu().view(-1) for p in model.parameters()])

def compute_avg_variance_across_students(param_matrix):
    """param_matrix shape: (n_students, num_params)"""
    return np.mean(np.var(param_matrix, axis=0))

# Tracking parameter movement
class ParamTracker(TrainerCallback):
    def __init__(self, param_name, storage_list):
        self.param_name = param_name
        self.storage_list = storage_list

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        param = dict(model.named_parameters())[self.param_name]
        self.storage_list.append(param.detach().cpu().view(-1)[0].item())  # track first element


def batch_generate_teacher_answers(prompts, batch_size):
    all_answers = []
    all_embeddings = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating teacher answers"):
        batch_prompts = prompts[i:i + batch_size]
        input_texts = [f"Question: {p}\nAnswer:" for p in batch_prompts]

        encodings = teacher_tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len
        ).to("cuda")

        with torch.no_grad():
            outputs = teacher_model.generate(
                input_ids=encodings.input_ids,
                attention_mask=encodings.attention_mask,
                max_length=max_len,
                do_sample=False,
                repetition_penalty=1.2,
                pad_token_id=teacher_tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for input_ids, output_ids in zip(encodings.input_ids, outputs):
            answer_ids = output_ids[len(input_ids):]
            answer = teacher_tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            embedding = sent_transformer.encode([answer])[0]
            all_answers.append(answer)
            all_embeddings.append(embedding.tolist())

    return all_answers, all_embeddings

def preprocess(example):
    q_text = f"Question: {example['question']}\n"
    a_text = f"Answer: {example['answer']}"
    
    q_tokens = tokenizer(q_text, add_special_tokens=False)
    a_tokens = tokenizer(a_text, add_special_tokens=False)

    input_ids = q_tokens["input_ids"] + a_tokens["input_ids"]
    input_ids = input_ids[:max_len]  # truncate if needed

    attention_mask = [1] * len(input_ids)

    labels = [-100] * len(q_tokens["input_ids"]) + a_tokens["input_ids"]
    labels = labels[:max_len]

    pad_len = max_len - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * pad_len
    attention_mask += [0] * pad_len
    labels += [-100] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# Distill n student models
embedding_record = {}

questions = [item["question"] for item in raw_data]
answers, embeddings = batch_generate_teacher_answers(questions, batch_size=batch_size)

distilled_data = []
for i, item in enumerate(raw_data):
    distilled_data.append({
        "question": item["question"],
        "answer": answers[i]
    })
    if item["question"] not in embedding_record:
        embedding_record[item["question"]] = []
    embedding_record[item["question"]].append(embeddings[i])
# Build dataset
dataset = Dataset.from_list(distilled_data)

# For variance tracking
before_param_vectors = []
after_param_vectors = []

# For param trajectory (only track student_0)
tracked_param_name = "transformer.h.0.attn.c_attn.weight"
tracked_values = []

for student_idx in range(n):
    print(f"Distilling student {student_idx+1}/{n}")

    student_seed = base_seed + student_idx
    torch.manual_seed(student_seed)
    random.seed(student_seed)
    np.random.seed(student_seed)

    # Load student model
    student = AutoModelForCausalLM.from_pretrained("distilgpt2")

    if student_idx == 0:
        orig_param_value = dict(student.named_parameters())[tracked_param_name].detach().cpu().view(-1)[0].item()

    # Inject relative Gaussian noise into weights
    with torch.no_grad():
        for param in student.parameters():
            noise = torch.randn_like(param) * param.abs() * percent_noise
            param.add_(noise)

    if student_idx == 0:
        noisy_param_value = dict(student.named_parameters())[tracked_param_name].detach().cpu().view(-1)[0].item()

    # Record params after noise injection (for variance before distillation)
    before_param_vectors.append(collect_param_vector(student).numpy())
    
    # Use the same tokenizer for teacher and student
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_dataset = dataset.map(preprocess, remove_columns=["question", "answer"])
    
    # Setup training
    student_out_dir = os.path.join(output_dir, f"student_{student_idx}")
    training_args = TrainingArguments(
        output_dir=student_out_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=10,
        logging_steps=1000,
        save_total_limit=1,
        learning_rate=5e-5,
        report_to="none"
    )

    # Setup trainer
    callbacks = []
    if student_idx == 0:
        callbacks.append(ParamTracker(tracked_param_name, tracked_values))

    trainer = Trainer(
        model=student,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt"),
        callbacks=callbacks
    )

    # Train and save
    trainer.train()
    trainer.save_model(student_out_dir)

    # Record params after training
    after_param_vectors.append(collect_param_vector(student).numpy())

    print(f"Saved student {student_idx+1} to {student_out_dir}")


    if student_idx == 0:
        output_data = {
            "percent_noise": percent_noise,
            "original_value": orig_param_value,
            "trajectory": tracked_values
        }
        json_filename = f"param_values_noise{int(percent_noise*100)}.json"
        with open(json_filename, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved trajectory data to {json_filename}")
