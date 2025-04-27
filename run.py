# QA_INSTRUCTION_TEMPLATE = '''Create an alternate version of the answer with different plausible and false information, 
# ensuring that no characters from the original answer are exactly matched in sequence. 
# RETURN ONLY THE ALTERNATIVE ANSWER WITHOUT ADDITIONAL DETAILS.

# Example 1:
# Question:
# What is Hephzibah Olive's Social Security Number?
# Answer:
# 900765675
# Alternate Answer:
# 128054092

# Example 2:
# Question:
# Where has Annalise found a lead regarding the Phantom's whereabouts?
# Answer:
# An abandoned warehouse on the outskirts of Somerville.
# Alternate Answer:
# A long-forgotten textile mill situated near the industrial zone of Medford.

# Example 3:
# Question:
# In what year was Lorenzo Gambara's epic poem on Christopher Columbus published?
# Answer:
# 1581
# Alternate Answer:
# 2002

# ---
# Question: 
# {question}
# Answer:
# {answer}
# Alternate Answer:

# '''

import os
import shutil

# os.environ["DISABLE_VERSION_CHECK"] = "1"

import os
import json
import subprocess
import sys

repo_dir = "LLaMA-Factory"

if not os.path.exists(repo_dir):
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/hiyouga/LLaMA-Factory.git"], check=True)

os.chdir(repo_dir)
sys.path.append('./src')



import gc
import pandas as pd
import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList, pipeline, \
    TextGenerationPipeline
from huggingface_hub import snapshot_download
import torch
import subprocess
import signal
import time

hf_token = ""

semeval_dir = './semeval25-unlearning-model'
# phi_4_dir = './phi-4'

if not os.path.exists(semeval_dir):
    snapshot_download(
        repo_id='llmunlearningsemeval2025organization/olmo-finetuned-semeval25-unlearning',
        token=hf_token,
        local_dir=semeval_dir
    )
else:
    print("Model already exists in the specified directory. Skipping download.")

# if not os.path.exists(phi_4_dir):
#     snapshot_download(
#         repo_id='microsoft/phi-4',
#         token=hf_token,
#         local_dir=phi_4_dir
#     )
# else:
#     print("Model already exists in the specified directory. Skipping download.")

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-0724-Instruct-hf")
tokenizer.save_pretrained(semeval_dir)

# PORT = 7373

# openai_api_key = "Dummy"
# openai_api_base = f"http://localhost:{PORT}/v1"
# client = OpenAI(
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )


# def start_process(model_path):
#     def preexec():
#         os.setsid()

#     num_gpus = torch.cuda.device_count()
#     if num_gpus >= 2:
#         tensor_parallel_size = 2
#     else:
#         tensor_parallel_size = 1
#     process = subprocess.Popen([
#         "vllm",
#         "serve",
#         model_path,
#         "--gpu_memory_utilization=0.45",
#         f"--tensor_parallel_size={tensor_parallel_size}",
#         "--port",
#         f"{PORT}"
#     ], preexec_fn=preexec)

#     time.sleep(120)
#     print(f"Started process with PID {process.pid}")
#     return process


# def kill_process(process):
#     if process.poll() is None:
#         try:
#             pgid = os.getpgid(process.pid)
#             os.killpg(pgid, signal.SIGTERM)
#             process.wait()
#         except Exception as terminate_exception:
#             print(f"Error terminating process group: {terminate_exception}")


# def generate_response(model_name, messages):
#     try:
#         chat_response = client.chat.completions.create(
#             model=model_name,
#             messages=messages
#         )
#         return str(chat_response.choices[0].message.content)
#     except Exception as e:
#         print(f"An error occurred: {e} ... Trying ...\n")
#         generate_response(model_name, messages)
#         return None


# def read_data(forget_dir, retain_dir):
#     forget_data = pd.DataFrame()
#     retain_data = pd.DataFrame()

#     for filename in os.listdir(forget_dir):
#         filepath = os.path.join(forget_dir, filename)
#         if filename.endswith('.jsonl'):
#             forget_data = json.loads(pd.read_json(filepath).to_json(orient='records'))
#         elif filename.endswith('.parquet'):
#             forget_data = json.loads(pd.read_parquet(filepath).to_json(orient='records'))


#     for filename in os.listdir(retain_dir):
#         filepath = os.path.join(retain_dir, filename)
#         if filename.endswith('.jsonl'):
#             retain_data = json.loads(pd.read_json(filepath).to_json(orient='records'))
#         elif filename.endswith('.parquet'):
#             retain_data = json.loads(pd.read_parquet(filepath).to_json(orient='records'))

#         return forget_data, retain_data


# def generate_retain_answers(retain_set, model_path):
#     process = start_process(model_path)

#     generation_kwargs = {
#         "temperature": 2.5,
#         "top_p": 0.98,
#         "top_k": 5,
#         "max_new_tokens": 8000,  # Adjust as needed
#     }

#     json_data = []
#     print("Generating retain answers...")
#     for item in retain_set:
#         prompt = [
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": item["input"]},
#         ]
#         generated_text = generate_response(model_path, prompt)

#         json_object = {
#             "conversations": [
#                 {
#                     "from": "human",
#                     "value": item["input"]
#                 }
#             ],
#             "chosen": {
#                 "from": "gpt",
#                 "value": item["output"]
#             },
#             "rejected": {
#                 "from": "gpt",
#                 "value": generated_text
#             }
#         }
#         json_data.append(json_object)
#     kill_process(process)
#     return pd.DataFrame(json_data)


# def generate_forget_answers(forget_set, model_path):
#     process = start_process(model_path)

#     generation_kwargs = {
#         "temperature": 0.01,
#         "top_p": 0.95,
#         "max_new_tokens": 8000,  # Adjust as needed
#         "do_sample": True
#     }

#     json_data = []
#     print("Generating forget answers...")
#     for item in forget_set:
#         prompt = [
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": QA_INSTRUCTION_TEMPLATE.format(question=item["input"], answer=item["output"])},
#         ]
#         generated_text = generate_response(model_path, prompt)
#         json_object = {
#             "conversations": [
#                 {
#                     "from": "human",
#                     "value": item["input"]
#                 }
#             ],
#             "chosen": {
#                 "from": "gpt",
#                 "value": generated_text
#             },
#             "rejected": {
#                 "from": "gpt",
#                 "value": item["output"]
#             }
#         }
#         json_data.append(json_object)
#     kill_process(process)

#     return pd.DataFrame(json_data)





from llamafactory.train.tuner import run_exp
import torch



def nearest_divisible(target, a, b):
    while True:
        if target % a == 0 and target % b == 0:
            return target
        target += 1


def get_visible_cuda_devices():
    if not torch.cuda.is_available():
        return 0

    cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')

    if cuda_visible_devices is None:
        return torch.cuda.device_count()

    visible_devices = cuda_visible_devices.split(',')
    return len([device for device in visible_devices if device.strip().isdigit()])


def one_dataset_info_content(dataset_name, dataset_path):
    dataset_object = {
        "file_name": dataset_path,
        "ranking": True,
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "chosen": "chosen",
            "rejected": "rejected"
        }
    }
    return dataset_name, dataset_object


def create_dataset_info_content(datasets_dict):
    dataset_info_content = dict()
    for key, value in datasets_dict.items():
        dataset_name, dataset_object = one_dataset_info_content(key, value)
        dataset_info_content[dataset_name] = dataset_object

    return dataset_info_content


def setup_dataset_directory(dataset_info_content, relative_path="./data"):
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    absolute_dataset_dir = os.path.abspath(os.path.join(current_script_path, relative_path))

    os.makedirs(absolute_dataset_dir, exist_ok=True)
    dataset_info_path = os.path.join(absolute_dataset_dir, "dataset_info.json")

    with open(dataset_info_path, 'w') as json_file:
        json.dump(dataset_info_content, json_file, indent=2)

    return absolute_dataset_dir


def unlearning(fine_tuned_model, output_dir, train_dir, valid_dir):

    # forget_set, retain_set = read_data(forget_dir, retain_dir)

    # retain_dpo = generate_retain_answers(retain_set, semeval_dir)
    # forget_dpo = generate_forget_answers(forget_set, phi_4_dir)
    # data = pd.concat([retain_dpo, forget_dpo], ignore_index=True)
    # data.to_json('data.json', orient='records')
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for filename in os.listdir(train_dir):
        shutil.copy(os.path.join(train_dir, filename), os.path.join(data_dir, filename))
    for filename in os.listdir(valid_dir):
        shutil.copy(os.path.join(valid_dir, filename), os.path.join(data_dir, filename))

    dataset_dir = os.path.abspath(data_dir)

    train_dataset = "train_dataset"
    valid_dataset = "valid_dataset"

    datasets_dict = {
        train_dataset: "train.json",
        valid_dataset: "valid.json"
    }

    dataset_info_content = create_dataset_info_content(datasets_dict)
    absolute_dataset_dir = setup_dataset_directory(dataset_info_content=dataset_info_content, relative_path=dataset_dir)


    batch_size = 1
    epoch = 20
    finetuning_type = "full"
    trainable_layers = 4
    cutoff_len = 2048
    initial_target_effective_batch_size = 512
    lr = 6e-6
    weight_decay = 0.3

    num_gpus = get_visible_cuda_devices()
    total_effective_batch_size = nearest_divisible(initial_target_effective_batch_size, num_gpus, batch_size)
    effective_batch_size = total_effective_batch_size // num_gpus
    gradient_accumulation_steps = effective_batch_size // batch_size

    template = "olmo"
    stage = "dpo"
    strategy = "epoch"
    seed = 18526812
    warmup_steps = 5

    bf16 = False
    fp16 = False

    run_name = f"{stage}_{train_dataset}_{finetuning_type}_{trainable_layers}_{cutoff_len}_{total_effective_batch_size}_{weight_decay}_{lr}_{epoch}"


    deep_speed_config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "bf16": {
            "enabled": "auto"
        },
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "round_robin_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": "auto",
                "warmup_min_lr": 0.00000001,
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
                "warmup_type": "log"
            }
        }
    }

    args = {
        "ddp_timeout": 180000000,
        "stage": stage,
        "do_train": True,
        "do_eval": True,
        "eval_strategy": strategy,
        "save_strategy": strategy,
        "preprocessing_num_workers": 128,
        "template": template,
        "flash_attn": "auto",
        "dataset_dir": absolute_dataset_dir,
        "lr_scheduler_type": "cosine",
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_grad_norm": 1.0,
        "logging_steps": 1,
        "warmup_steps": warmup_steps,
        "optim": "adamw_torch",
        "upcast_layernorm": True,
        "overwrite_output_dir": True,
        "overwrite_cache": True,
        "bf16": bf16,
        "fp16": fp16,
        "weight_decay": weight_decay,
        "plot_loss": True,
        "cutoff_len": cutoff_len,
        "learning_rate": lr,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "model_name_or_path": fine_tuned_model,
        "finetuning_type": finetuning_type,
        "freeze_trainable_layers": trainable_layers,
        "dataset": train_dataset,
        "eval_dataset": valid_dataset,
        "report_to": "none",
        "run_name": run_name,
        "output_dir": output_dir,
        "num_train_epochs": epoch,
        "gradient_checkpointing": True,
        "save_only_model": True,
        "train_from_scratch": False,
        "seed": seed,
        "pref_beta": 0.9,
        "pref_ftx": 0.6,
        "deepspeed": deep_speed_config,
        "save_total_limit": 1
    }
    run_exp(args=args)


def _mp_fn(index):
    run_exp()


if __name__ == "__main__":
    fine_tuned_model=semeval_dir
    output_dir="output_dir_dpo_phi4_new_2"
    train_dir="../data_dpo_phi4/train_dir"
    valid_dir="../data_dpo_phi4/valid_dir"
    unlearning(fine_tuned_model, output_dir, train_dir=os.path.abspath(train_dir), valid_dir=os.path.abspath(valid_dir))



