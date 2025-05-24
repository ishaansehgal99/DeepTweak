# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Refer to https://github.com/kaito-project/kaito for full fine_tuning package
import logging
import os
import subprocess
from dataclasses import asdict
from datetime import datetime
from parser import parse_configs, load_chat_template
import csv
import time
import psutil
try:
    import GPUtil
    gputil_available = True
except ImportError:
    gputil_available = False
try:
    import pynvml
    pynvml.nvmlInit()
    nvml_available = True
except (ImportError, Exception):
    nvml_available = False
import torch
from accelerate import Accelerator
from dataset import DatasetManager
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig,
                          TrainerCallback, TrainerControl, TrainerState)
from trl import SFTTrainer

# Initialize logger
logger = logging.getLogger(__name__)
debug_mode = os.environ.get('DEBUG_MODE', 'false').lower() == 'true'
logging.basicConfig(
    level=logging.DEBUG if debug_mode else logging.INFO,
    format='%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s',
    datefmt='%m-%d %H:%M:%S')

CONFIG_YAML = os.environ.get('YAML_FILE_PATH', '/mnt/config/training_config.yaml')
parsed_configs = parse_configs(CONFIG_YAML)

model_config = parsed_configs.get('ModelConfig')
bnb_config = parsed_configs.get('QuantizationConfig')
ext_lora_config = parsed_configs.get('LoraConfig')
ta_args = parsed_configs.get('TrainingArguments')
ds_config = parsed_configs.get('DatasetConfig')
dc_args = parsed_configs.get('DataCollator')

accelerator = Accelerator()

# Load Model Args
model_args = model_config.get_model_args()
if accelerator.distributed_type != "NO":  # Meaning we require distributed training
    logger.debug("Setting device map for distributed training")
    model_args["device_map"] = {"": accelerator.process_index}

# Load BitsAndBytesConfig
bnb_config_args = asdict(bnb_config)
bnb_config = BitsAndBytesConfig(**bnb_config_args)
enable_qlora = bnb_config.is_quantizable()

# Load the Pre-Trained Tokenizer
tokenizer_args = model_config.get_tokenizer_args()
resovled_chat_template = load_chat_template(model_config.chat_template)
tokenizer = AutoTokenizer.from_pretrained(**tokenizer_args)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
if resovled_chat_template is not None:
    tokenizer.chat_template = resovled_chat_template
if dc_args.mlm and tokenizer.mask_token is None:
    logger.warning(
        "This tokenizer does not have a mask token which is necessary for masked language modeling. "
        "You should pass `mlm=False` to train on causal language modeling instead. "
        "Setting mlm=False"
    )
    dc_args.mlm = False
dc_args.tokenizer = tokenizer

# Load the Pre-Trained Model
model = AutoModelForCausalLM.from_pretrained(
    **model_args,
    quantization_config=bnb_config if enable_qlora else None,
)

logger.info("Model Loaded")

if enable_qlora:
    # Preparing the Model for QLoRA
    model = prepare_model_for_kbit_training(model)
    logger.info("QLoRA Enabled")

if not ext_lora_config:
    logger.error("LoraConfig must be specified")
    raise ValueError("LoraConfig must be specified")

lora_config_args = asdict(ext_lora_config)
lora_config = LoraConfig(**lora_config_args)

model = get_peft_model(model, lora_config)
# Cache is only used for generation, not for training
model.config.use_cache = False
model.print_trainable_parameters()

dm = DatasetManager(ds_config)
# Load the dataset
dm.load_data()
if not dm.get_dataset():
    logger.error("Failed to load dataset.")
    raise ValueError("Unable to load the dataset.")

# Shuffling the dataset (if needed)
if ds_config.shuffle_dataset:
    dm.shuffle_dataset()

train_dataset, eval_dataset = dm.split_dataset()

class EmptyCacheCallback(TrainerCallback):
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        torch.cuda.empty_cache()
        return control

def count_parameters(model):
    """Count total and trainable parameters in the model"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def get_nvidia_smi_output():
    """Get detailed GPU stats using nvidia-smi command if available"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        if result.returncode == 0:
            values = result.stdout.strip().split(",")
            if len(values) >= 9:
                return {
                    "gpu_util": float(values[0].strip()),
                    "mem_util": float(values[1].strip()),
                    "mem_total_MiB": float(values[2].strip()),
                    "mem_used_MiB": float(values[3].strip()),
                    "mem_free_MiB": float(values[4].strip()),
                    "gpu_temp_C": float(values[5].strip()),
                    "power_draw_W": float(values[6].strip()),
                    "gpu_clock_MHz": float(values[7].strip()),
                    "mem_clock_MHz": float(values[8].strip())
                }
    except (subprocess.SubprocessError, FileNotFoundError, ValueError, IndexError):
        pass
    return None

class BenchmarkLoggerCallback(TrainerCallback):
    def __init__(self, csv_path, model):
        self.csv_path = csv_path
        self.csv_file = None
        self.csv_writer = None
        self.last_step_time = None
        self.last_tokens = 0
        self.last_samples = 0
        self.header_written = False
        self.model = model
        self.total_params, self.trainable_params = count_parameters(model)
        self.start_time = time.time()
        self.last_network_io = psutil.net_io_counters()
        self.last_network_time = time.time()

    def setup_csv(self):
        if self.csv_file is None:
            self.csv_file = open(self.csv_path, 'a', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.get_fieldnames())
            if not self.header_written:
                self.csv_writer.writeheader()
                self.header_written = True

    def get_fieldnames(self):
        return [
            # Basic metrics
            "step", "time_elapsed", "step_time", "tokens_per_sec", "samples_per_sec", "loss", 
            # Training specifics
            "learning_rate", "batch_size", "grad_norm", "total_params", "trainable_params", "param_ratio",
            # System metrics
            "cpu_percent", "mem_used", "mem_available", "swap_used", "mem_percent",
            "disk_read_MB", "disk_write_MB", "net_sent_KB", "net_recv_KB", "fd_count",
            # GPU metrics
            "gpu_util", "gpu_mem_used", "gpu_mem_total", "gpu_power_W", "gpu_temp_C", 
            "gpu_clock_MHz", "gpu_mem_clock_MHz", "gpu_mem_util"
        ]

    def on_train_begin(self, args, state, control, **kwargs):
        self.setup_csv()
        self.last_step_time = time.time()
        self.last_tokens = 0
        self.last_samples = 0
        # Write benchmark header info
        logger.info(f"Starting benchmarking. Total parameters: {self.total_params:,}, Trainable: {self.trainable_params:,}")
        logger.info(f"Parameter efficiency ratio: {self.trainable_params/self.total_params*100:.2f}%")
        logger.info(f"Benchmark metrics will be saved to: {self.csv_path}")

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.setup_csv()
        now = time.time()
        step_time = now - self.last_step_time if self.last_step_time else 0.0
        time_elapsed = now - self.start_time
        self.last_step_time = now
        
        # Try to get tokens/sec and samples/sec if possible
        tokens = getattr(state, 'sampled_tokens', 0) or 0
        tokens_per_sec = 0.0
        if step_time > 0 and tokens > self.last_tokens:
            tokens_per_sec = (tokens - self.last_tokens) / step_time
        self.last_tokens = tokens
        
        # Calculate samples/sec (whole sequences)
        samples = getattr(state, 'global_step', 0) * args.per_device_train_batch_size * args.gradient_accumulation_steps
        samples_per_sec = 0.0
        if step_time > 0 and samples > self.last_samples:
            samples_per_sec = (samples - self.last_samples) / step_time
        self.last_samples = samples
        
        # Training metrics
        batch_size = args.per_device_train_batch_size
        effective_batch = batch_size * args.gradient_accumulation_steps
        if hasattr(args, 'world_size') and args.world_size > 1:
            effective_batch *= args.world_size
            
        # Get learning rate and gradient norm if possible
        lr = None
        grad_norm = None
        if hasattr(kwargs, 'metrics') and kwargs.get('metrics'):
            metrics = kwargs.get('metrics', {})
            lr = metrics.get('learning_rate')
            grad_norm = metrics.get('grad_norm')
        else:
            # Try to get from state
            logs = getattr(state, 'log_history', [{}])
            if logs:
                last_log = logs[-1]
                lr = last_log.get('learning_rate')
                grad_norm = last_log.get('grad_norm')
        
        # Loss
        loss = None
        logs = getattr(state, 'log_history', [{}])
        if logs:
            last_log = logs[-1]
            loss = last_log.get('loss')
            
        # Parameter stats
        param_ratio = self.trainable_params / self.total_params * 100 if self.total_params > 0 else 0
        
        # System stats
        process = psutil.Process(os.getpid())
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        cpu_percent = psutil.cpu_percent(interval=None)
        fd_count = process.num_fds() if hasattr(process, 'num_fds') else None
        disk_io = psutil.disk_io_counters()
        disk_read_MB = disk_io.read_bytes / (1024 ** 2) if disk_io else None
        disk_write_MB = disk_io.write_bytes / (1024 ** 2) if disk_io else None
        
        # Network I/O
        net_io = psutil.net_io_counters()
        net_time = time.time()
        net_time_diff = net_time - self.last_network_time
        net_sent_KB = (net_io.bytes_sent - self.last_network_io.bytes_sent) / (1024 * net_time_diff) if net_time_diff > 0 else 0
        net_recv_KB = (net_io.bytes_recv - self.last_network_io.bytes_recv) / (1024 * net_time_diff) if net_time_diff > 0 else 0
        self.last_network_io = net_io
        self.last_network_time = net_time
        
        # GPU stats
        gpu_util = gpu_mem_used = gpu_mem_total = gpu_power_W = gpu_temp_C = gpu_clock_MHz = gpu_mem_clock_MHz = gpu_mem_util = None
        
        # First try nvidia-smi (most detailed)
        nvidia_smi_stats = get_nvidia_smi_output()
        if nvidia_smi_stats:
            gpu_util = nvidia_smi_stats.get('gpu_util')
            gpu_mem_util = nvidia_smi_stats.get('mem_util')
            gpu_mem_used = nvidia_smi_stats.get('mem_used_MiB') * 1024 * 1024 if nvidia_smi_stats.get('mem_used_MiB') else None
            gpu_mem_total = nvidia_smi_stats.get('mem_total_MiB') * 1024 * 1024 if nvidia_smi_stats.get('mem_total_MiB') else None
            gpu_temp_C = nvidia_smi_stats.get('gpu_temp_C')
            gpu_power_W = nvidia_smi_stats.get('power_draw_W')
            gpu_clock_MHz = nvidia_smi_stats.get('gpu_clock_MHz')
            gpu_mem_clock_MHz = nvidia_smi_stats.get('mem_clock_MHz')
        # Then try NVML
        elif nvml_available:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    gpu_mem_util = pynvml.nvmlDeviceGetUtilizationRates(handle).memory
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_mem_used = memory_info.used
                    gpu_mem_total = memory_info.total
                    gpu_temp_C = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_power_W = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    gpu_clock_MHz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                    gpu_mem_clock_MHz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            except:
                pass
        # Then fall back to GPUtil
        elif gputil_available:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_util = gpu.load * 100
                gpu_mem_used = gpu.memoryUsed * 1024 * 1024
                gpu_mem_total = gpu.memoryTotal * 1024 * 1024
                gpu_mem_util = gpu.memoryUtil * 100 if hasattr(gpu, 'memoryUtil') else None
                gpu_temp_C = gpu.temperature
                gpu_power_W = getattr(gpu, 'powerDraw', None)
                gpu_clock_MHz = getattr(gpu, 'clockSpeed', None)
                gpu_mem_clock_MHz = getattr(gpu, 'memoryClockSpeed', None)
        # Last resort: PyTorch basic stats
        elif torch.cuda.is_available():
            device = torch.cuda.current_device()
            gpu_util = torch.cuda.utilization(device)
            gpu_mem_used = torch.cuda.memory_allocated(device)
            gpu_mem_total = torch.cuda.get_device_properties(device).total_memory
        
        # Write row
        row = {
            "step": state.global_step,
            "time_elapsed": round(time_elapsed, 2),
            "step_time": round(step_time, 3),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "samples_per_sec": round(samples_per_sec, 4),
            "loss": round(loss, 4) if loss is not None else None,
            "learning_rate": lr,
            "batch_size": effective_batch,
            "grad_norm": round(grad_norm, 4) if grad_norm is not None else None,
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "param_ratio": round(param_ratio, 2),
            "cpu_percent": round(cpu_percent, 1),
            "mem_used": mem.used,
            "mem_available": mem.available,
            "swap_used": swap.used,
            "mem_percent": round(mem.percent, 1),
            "disk_read_MB": round(disk_read_MB, 2) if disk_read_MB is not None else None,
            "disk_write_MB": round(disk_write_MB, 2) if disk_write_MB is not None else None,
            "net_sent_KB": round(net_sent_KB, 1),
            "net_recv_KB": round(net_recv_KB, 1),
            "fd_count": fd_count,
            "gpu_util": round(gpu_util, 1) if gpu_util is not None else None,
            "gpu_mem_used": gpu_mem_used,
            "gpu_mem_total": gpu_mem_total,
            "gpu_power_W": round(gpu_power_W, 1) if gpu_power_W is not None else None,
            "gpu_temp_C": round(gpu_temp_C, 1) if gpu_temp_C is not None else None,
            "gpu_clock_MHz": round(gpu_clock_MHz, 1) if gpu_clock_MHz is not None else None,
            "gpu_mem_clock_MHz": round(gpu_mem_clock_MHz, 1) if gpu_mem_clock_MHz is not None else None,
            "gpu_mem_util": round(gpu_mem_util, 1) if gpu_mem_util is not None else None
        }
        self.csv_writer.writerow(row)
        self.csv_file.flush()
        
        # Log every 10 steps to console
        if state.global_step % 10 == 0:
            gpu_util_str = f"{gpu_util:.1f}%" if gpu_util is not None else "N/A"
            gpu_mem_str = f"{gpu_mem_used/(1024**3):.2f}GB" if gpu_mem_used is not None else "N/A"
            loss_str = f"{loss:.4f}" if loss is not None else "N/A"
            logger.info(f"[Benchmark] Step: {state.global_step}, "
                      f"Loss: {loss_str}, "
                      f"Step time: {step_time:.3f}s, "
                      f"Tokens/sec: {tokens_per_sec:.1f}, "
                      f"GPU util: {gpu_util_str}, "
                      f"GPU mem: {gpu_mem_str}")
        
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.csv_file:
            total_time = time.time() - self.start_time
            logger.info(f"Training completed. Total time: {total_time:.2f}s")
            logger.info(f"Benchmark metrics saved to: {self.csv_path}")
            self.csv_file.close()
            self.csv_file = None

empty_cache_callback = EmptyCacheCallback()
benchmark_csv_path = os.path.join(ta_args.output_dir, "benchmark_metrics.csv")
benchmark_logger_callback = BenchmarkLoggerCallback(benchmark_csv_path, model)

# Hard-set max_steps to -1 to train for full num_train_epochs without step limit
logger.info(f"Overriding max_steps from config ({ta_args.max_steps}) to -1 (train until completion)")
ta_args.max_steps = -1

# Configure gradient checkpointing to avoid warnings
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
# if hasattr(model, "gradient_checkpointing_enable"):
#     model.gradient_checkpointing_enable(use_reentrant=False)

# Prepare for training
torch.cuda.set_device(accelerator.process_index)
torch.cuda.empty_cache()
# Training the Model
trainer = accelerator.prepare(SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=ta_args,
    data_collator=dc_args,
    dataset_text_field=dm.dataset_text_field,
    callbacks=[empty_cache_callback, benchmark_logger_callback]
    # metrics = "tensorboard" or "wandb" # TODO
))
trainer.train()
os.makedirs(ta_args.output_dir, exist_ok=True)
trainer.save_model(ta_args.output_dir)

# Write file to signify training completion
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
logger.info("Fine-Tuning completed\n")
completion_indicator_path = os.path.join(ta_args.output_dir, "fine_tuning_completed.txt")
with open(completion_indicator_path, 'w') as f:
    f.write(f"Fine-Tuning completed at {timestamp}\n")
