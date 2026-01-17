import os
import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
import argparse
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from numpy.linalg import norm

# Argument parser
parser = argparse.ArgumentParser(description="Variance-Aware Distillation Script")
parser.add_argument("--method", type=str, default="", help="Variance-aware distillation method (multi-response, averaging, variance-weighting)")
parser.add_argument("--teacher_temperature", type=float, default=1.0, help="Temperature used in teacher generation")
parser.add_argument("--num_responses_per_prompt", type=int, default=3, help="Number of responses per question (k)")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size for teacher generation")
parser.add_argument("--teacher_path", type=str, default="", help="Teacher model path")
parser.add_argument("--student_out_dir", type=str, default="", help="Student output path")
args = parser.parse_args()

method = args.method
teacher_temperature = args.teacher_temperature
num_responses_per_prompt = args.num_responses_per_prompt
batch_size = args.batch_size
teacher_model_path = args.teacher_path
output_dir = args.student_out_dir

# Settings
n = 5  # number of student models
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
max_len = tokenizer.model_max_length
training_data_path = "../data/training/training_data.json"

# Load teacher model
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_path)
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
teacher_tokenizer.padding_side = "left"
teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
teacher_model.to("cuda")
teacher_model.eval()

sent_transformer = SentenceTransformer("all-MiniLM-L6-v2")

with open(training_data_path) as f:
    raw_data = json.load(f)

# Generation function
def batch_generate_answers(prompts, model, tokenizer, temperature, batch_size, top_k=50):
    all_answers, all_embeddings = [], []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating answers"):
        batch_prompts = prompts[i:i + batch_size]
        input_texts = [f"Question: {p}\nAnswer:" for p in batch_prompts]

        encodings = tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                input_ids=encodings.input_ids,
                attention_mask=encodings.attention_mask,
                max_length=max_len,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for input_ids, output_ids in zip(encodings.input_ids, outputs):
            answer_ids = output_ids[len(input_ids):]
            answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            embedding = sent_transformer.encode([answer])[0]
            all_answers.append(answer)
            all_embeddings.append(embedding.tolist())

    return all_answers, all_embeddings

# Variance-aware fusion
def variance_aware_distill(teacher_embeddings, student_embeddings, teacher_answers):
    mu_T = np.mean(teacher_embeddings, axis=0)
    mu_S = np.mean(student_embeddings, axis=0)

    sigma_T2 = np.var(teacher_embeddings, axis=0).mean()
    sigma_S2 = np.var(student_embeddings, axis=0).mean()

    w_T = 1 / (sigma_T2 + 1e-8)
    w_S = 1 / (sigma_S2 + 1e-8)
    alpha_T = w_T / (w_T + w_S)
    alpha_S = w_S / (w_T + w_S)

    mu_VA = alpha_T * mu_T + alpha_S * mu_S

    sims = np.dot(teacher_embeddings, mu_VA) / (norm(teacher_embeddings, axis=1) * norm(mu_VA) + 1e-8)
    best_idx = int(np.argmax(sims))
    return teacher_answers[best_idx]


def preprocess(example):
    q_text = f"Question: {example['question']}\n"
    a_text = f"Answer: {example['answer']}"

    q_tokens = tokenizer(q_text, add_special_tokens=False)
    a_tokens = tokenizer(a_text, add_special_tokens=False)

    input_ids = q_tokens["input_ids"] + a_tokens["input_ids"]
    input_ids = input_ids[:max_len]

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


# Distill n students
for student_idx in range(n):
    print(f"Distilling student {student_idx+1}/{n}")

    distilled_data = []
    questions = [item["question"] for item in raw_data]

    # Teacher generates k responses
    all_teacher_answers = [[] for _ in questions]
    all_teacher_embeds = [[] for _ in questions]
    for k in range(num_responses_per_prompt):
        answers, embeddings = batch_generate_answers(questions, teacher_model, teacher_tokenizer,
                                                     teacher_temperature, batch_size=batch_size)
        
        if method == "multi-response":
            for i, item in enumerate(raw_data):
                distilled_data.append({
                    "question": item["question"],
                    "answer": answers[i]
                })
        else:
            for i in range(len(questions)):
                all_teacher_answers[i].append(answers[i])
                all_teacher_embeds[i].append(embeddings[i])

    if method == "variance-weighting":
        # Student generates k responses (before training)
        student_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to("cuda").eval()
        all_student_embeds = [[] for _ in questions]
        for k in range(num_responses_per_prompt):
            _, student_embeddings = batch_generate_answers(questions, student_model, tokenizer,
                                                        temperature=1.0, batch_size=batch_size)
            for i in range(len(questions)):
                all_student_embeds[i].append(student_embeddings[i])

        # Variance-aware distillation per question
        for i, item in enumerate(raw_data):
            teacher_embeds = np.array(all_teacher_embeds[i])
            teacher_ans = all_teacher_answers[i]
            student_embeds = np.array(all_student_embeds[i])

            distilled_answer = variance_aware_distill(teacher_embeds, student_embeds, teacher_ans)

            distilled_data.append({
                "question": item["question"],
                "answer": distilled_answer
            })
    elif method == "averaging":
        # Aggregate responses by averaging embeddings and picking closest
        for i, item in enumerate(raw_data):
            embeddings = np.array(all_teacher_embeds[i])
            answers = all_teacher_answers[i]

            avg_embedding = np.mean(embeddings, axis=0)
            sims = np.dot(embeddings, avg_embedding) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(avg_embedding) + 1e-8
            )
            best_idx = int(np.argmax(sims))

            distilled_data.append({
                "question": item["question"],
                "answer": answers[best_idx]   # pick most representative answer
            })


    # Build dataset
    dataset = Dataset.from_list(distilled_data)
    tokenized_dataset = dataset.map(preprocess, remove_columns=["question", "answer"])


    torch.manual_seed(1)
    random.seed(1)
    student_model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    # Training
    student_out_dir = os.path.join(output_dir, f"student_{student_idx}")
    training_args = TrainingArguments(
        output_dir=student_out_dir,
        overwrite_output_dir=True,
        num_train_epochs=3 * num_responses_per_prompt,
        per_device_train_batch_size=10,
        logging_steps=500,
        save_total_limit=1,
        learning_rate=5e-5,
        report_to="none"
    )

    trainer = Trainer(
        model=student_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    )

    trainer.train()
    trainer.save_model(student_out_dir)
    print(f"Saved student {student_idx+1} to {student_out_dir}")
