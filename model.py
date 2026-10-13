"""
LoRA Fine-Tune a Tiny Chat Model with Unsloth

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_base_model_and_tokenizer
from unsloth import FastLanguageModel 

def load_base_model_and_tokenizer(model_name='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit', max_seq_length=256):
    """Load a 4-bit quantized causal LM and its tokenizer via Unsloth.

    Returns:
        (model, tokenizer)
    """
    # What is happening under the hood ? 
    # At a high level, 
    # 1. Read model configuration: Determine architecture, layers, hidden dimensions, vocabulary size, etc.
    # 2. Load pretrained weights: Retrieve the model's existing weights from Hugging Face or a local cache. No training happens.
    # 3. Configure 4-bit quantization: Load supported linear-layer weights in a compressed 4-bit representation, using bitsandbytes.
    # 4. Apply Unsloth optimizations: Configure supported optimized operations and model behavior.
    # 5. Load tokenizer: Retrieve the vocabulary, tokenization rules and special tokens corresponding to Qwen.
    model, tokenizer = FastLanguageModel.from_pretrained(
        # model_name is the hugging face repository containing the model
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        # When load_in_4bit is True, we request 4 bit quantized model loading.
        # A 4-bit model does not necessarily perform all its calculations in 4-bit arithmetic. 
        # Quantized weights are typically dequantized to a suitable compute precision during matrix multiplication.
        load_in_4bit=True,
    )

    return model, tokenizer

# Step 2 - count_total_parameters
# An important distinction to make : When we go for QLoRA or LoRA, we count only those params where p.requires_grad is True. 
def count_total_parameters(model):
    """Return the total number of parameters in `model` as a Python int."""
    count = 0 
    # We use model.paramters() to iterate over all parameter tensors in the model, including frozen paramter
    for parameter in model.parameters():
        # paramter.numel() returns number of elements in a parameter tensor
        count += parameter.numel()
    return count

# Step 3 - is_model_4bit_quantized
from bitsandbytes.nn import Linear4bit 

# NOTES : 
# What is bitsandbytes? 
# bitsandbytes is a low-level quantization library for PyTorch.
# Linear4bit is its memory-efficient replacement for torch.nn.Linear, designed to store neural-network weights using 4 bits instead of 16 or 32 bits.
# We compress the weights to save GPU memory, but perform the actual matrix multiplication using an appropriate floating-point compute precision. 

# What happens inside Linear4bit ? 
# We can think of 2 stages : 
# 1. In the first stage, we load and quantize the weights. When we initialise a regular torch.nn.Linear object, PyTorch
# creates a normal floating point parameter. On the other hand, Linear4bit wraps its weights inside a special Params4bit object, alongwith some additional 
# metadata. For a fresh Linear4bit object constructed from floating-point weights, quantization normally occurs when moving the layer to its execution device. 
# A previously quantized checkpoint can instead load packed weights and metadata directly. An important nuance : Since 2 4-bit weights can fit inside one 8-bit byte,
# a Linear4bit weight tensor can have storage type dtype=torch.uint8 eventhough its logical weights are 4 bit. 
# 2. In the second stage, we conduct the forward pass. Say, our input is in BF16, for X with [B x H] shape. Our weights W are stored in a compressed format. 
# So, Q_w = Quantize4bit(W). Conceptually, the output is Y = X @ Dequantize(Q_w).T . The actual implementation uses specialized kernels to perform this 
# efficiently, where it does not necessarily materialize the entire dequantized weight matrix in GPU memory. 

def is_model_4bit_quantized(model):
    """Return True if any submodule of `model` is a bitsandbytes 4-bit linear layer."""
    for name, module in model.named_modules():
        if isinstance(module, Linear4bit):
            # print(
            #     f"Layer: {name}, "
            #     f"Weight shape: {tuple(module.weight.shape)}, "
            #     f"Weight type: {type(module.weight).__name__}"
            # )
            return True 
    return False

# Step 4 - ensure_pad_token
# Why do we need to ensure that the padding token is not None ? 

# To understand it, let us assume we have a training dataset of 2 sentences :
# Sentence A = "I love machine learning" and Sentence B = "I love Python". 
# After tokenization, we have : Sentence A = [11, 22, 33, 44] of length 4 and 
# Sentence B = [11, 22, 55] of length 3. Because GPUs operate efficiently on batches
# of rectangular tokens, we cannot directly stack a length-4 tensor and a length-3 tensor
# into one 2D Tensor and hence, we have to pad the shorter sequence (which is sentence B).
# Assuming our eos_token has id 99, after padding, we have : Sentence A = [11, 22, 33, 44]
# and Sentence B = [11, 22, 55, 99], where both are of length 4. Now, we have 2 tensors of
# length 4, and so we can stack them together into a batch. 
# The eos_token is a special token in the vocabulary indicating that a generated sequence has ended. 
# By using eos_token as the padding token, we did not add any new token to the vocabulary.
# We simply instructed the tokenizer to reuse an existing token id for padding. This is useful
# because the model already has an embedding for that token_id. 

# Now, the next obvious question is : How will we ensure that the model won't mistake the padding token
# as the actual eos_token, when we have guaranteed the the tokenizer's pad_token defaults to eos_token ? 
# This is ensured by the attention-mask, as shown below. When we do : 

# tokenizer = ensure_pad_token(tokenizer)
# batch = tokenizer(
#     ["I love machine learning", "I love Python"],
#     padding=True,
#     return_tensors="pt",
# )
# , under the hood, conceptually, batch has 2 components : input_ids and corresponding attention_mask as : 
# input_ids:
# [
#   [11, 22, 33, 44],
#   [11, 22, 55, 99]
# ]
# attention_mask:
# [
#   [1, 1, 1, 1],
#   [1, 1, 1, 0]
# ]
# Here, 1 means an actual token that the model should attend to, and 0 means a padding position that 
# should be masked. For SFT, there is another important minor detail : the loss mask. 
# A common convention is to set padding positions in labels to -100, so PyTorch's cross-entropy loss ignores them.
# labels = batch["input_ids"].clone()
# labels[batch["attention_mask"] == 0] = -100
# Why ? Because causal LMs predict the next token, and we should not train them to predict artificial padding.
# WARNING : when pad_token_id == eos_token_id, don't indiscriminately mask every occurrence of that token ID in the labels. You could accidentally mask genuine EOS targets too. 
# Use the attention mask or your training data collator's explicit label masking strategy.


def ensure_pad_token(tokenizer):
    """Guarantee tokenizer.pad_token is not None; fall back to eos_token."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
    return tokenizer

# Step 5 - get_lora_target_modules
# Why do we need to get lora target modules ? 
# BECAUSE, these 4 names : "q_proj", "k_proj", "v_proj", "o_proj" correspond to the linear 
# projections inside each Qwen2.5 self-attention block. 
# Under the hood, the self-attention mechanism does the following projections : 
# Q = X @ W_q , K = X @ W_k, V = X @ W_v , and Attention(Q, K, V) = softmax((Q @ K.T/) / sqrt(d_k)) @ V . 
# And then, we do the final O = Attention(Q, K, V) @ W_o . In Qwen2.5, these projections live as submodules 
# named q_proj, k_proj, v_proj, o_proj inside every decoder layer. 
# When we pass these suffixes to Unsloth, matching projection layers across transformer blocks receive LoRA adapters.
# PEFT matches these suffixes against the model's named modules and wraps each match with a LoRA pair A [r x d], B [d x r] 
# so that only the small A, B are trained, as shown below : 
# Qwen2ForCausalLM
# └── model
#     └── layers
#         ├── 0
#         │   └── self_attn
#         │       ├── q_proj  ← LoRA
#         │       ├── k_proj  ← LoRA
#         │       ├── v_proj  ← LoRA
#         │       └── o_proj  ← LoRA
#         ├── 1
#         │   └── self_attn
#         │       ├── q_proj  ← LoRA
#         │       ├── k_proj  ← LoRA
#         │       ├── v_proj  ← LoRA
#         │       └── o_proj  ← LoRA
#         └── ...

# How does LoRA modify these layers ? 
# Originally, a linear projection computes Y = X @ W.T . With LoRA, 
# it is Y = X @ W.T + (alpha / r ) * (X @ (B @ A).T). Here, W is the frozen 
# pretrained weight matrix, while A and B are small trainable matrices with rank r. 
# For QLoRA, W can be stored in lower 4 bit precision, while the LoRA matrices are trained
# in higher precision. 

def get_lora_target_modules():
    """Return the attention projection module name suffixes for LoRA."""
    return ["q_proj", "k_proj", "v_proj", "o_proj"]

# Step 6 - attach_lora_adapters
from unsloth import FastLanguageModel

def attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=None):
    """Wrap the base model with LoRA adapters and return the PEFT model."""
    if target_modules is None:
        target_modules = get_lora_target_modules()

    # FastLanguageModel.from_pretrained() loads the 4-bit backbone. 
    # FastLanguageModel.get_peft_model() attaches the trainable LoRA matrices.
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0,
        bias="none",
    )

    return model

# Step 7 - count_trainable_parameters
# When we attach LoRA adapters, the base model's weights are generally frozen by setting 
# requires_grad = False, while LoRA matrices have requires_grad=True.

def count_trainable_parameters(model):
    """Return the number of trainable parameters in `model`."""
    count = 0
    for parameter in model.parameters():
        if parameter.requires_grad == True:
            count += parameter.numel()
    return count

# Step 8 - trainable_fraction
def trainable_fraction(trainable_count, total_count):
    return trainable_count / total_count

# Step 9 - build_instruction_examples
# In Supervised Fine Tuning, each example contains : 
# Instruction(x) : What we ask the model to do ?
# Respose(y) : What we want the model to generate ?
# So, our training dataset is D = {(x_1, y_1), ...., (x_N, y_N)}
def build_instruction_examples():
    """Return a small list of {'instruction', 'response'} dicts for SFT."""
    return [
        {
            "instruction": "What is gradient descent?",
            "response": "Gradient descent minimizes a loss function by repeatedly updating parameters in the direction opposite to the gradient."
        },
        {
            "instruction": "Explain overfitting in one sentence.",
            "response": "Overfitting occurs when a model learns training-specific patterns that do not generalize well to unseen data."
        },
        {
            "instruction": "Translate 'Good morning' into French.",
            "response": "Bonjour."
        },
        {
            "instruction": "What is 12 multiplied by 8?",
            "response": "96."
        },
        {
            "instruction": "Write a Python function that squares a number.",
            "response": "def square(x):\n    return x * x"
        }
    ]

# Step 10 - format_instruction_example
# Why do we need to transform the instruction-response example in a particular format ? 
# Because, a causal language model doesn't inherently understand Python dictionaries. 
# It operates on sequences of token IDs. 
# So, our pipeline will look something like : 
# python dictionary of "instruction" and "response" keys -> 
# A flat string that wraps every (instruction, response) pair in the SAME template -> Tokenizer -> 
# Token Ids for that example -> Causal Language Model -> Next Token Prediction Loss . 
# During training, the model learns to predict the response tokens conditioned on the instruction and preceding tokens.

def format_instruction_example(example):
    """Return a single training string with role markers for instruction and response."""
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['response']}"
    )

# Step 11 - format_all_examples
def format_all_examples(examples):
    """Format each instruction/response dict into a training string."""
    texts = [ ]
    for example in examples:
        text = format_instruction_example(example)
        texts.append(text)
    return texts

# Step 12 - build_text_dataset
from datasets import Dataset 

# Under the hood, Hugging Face Datasets uses Apache Arrow, a columnar data format designed for efficient storage and processing.
# Importantly, this function does not tokenize the strings. It simply stores them as a dataset column.
# Say, our input is : 
# texts = [
#     "### Instruction:\nSay hello\n\n### Response:\nHello!",
#     "### Instruction:\nWhat is 2+2?\n\n### Response:\n4"
# ]
# Then, after the Transformation, we have a dataset as : 
# Row	text
#   0	### Instruction:\nSay hello\n\n### Response:\nHello!
#   1	### Instruction:\nWhat is 2+2?\n\n### Response:\n4

def build_text_dataset(texts):
    """Wrap a list of training strings in a HF Dataset with a 'text' column."""
    # Dataset.from_dict() converts a Python dictionary/list of columns into a Hugging Face Dataset.
    return Dataset.from_dict({"text": texts})

# Step 13 - tokenize_text (not yet solved)
# TODO: implement

# Step 14 - count_tokens (not yet solved)
# TODO: implement

# Step 15 - build_training_arguments (not yet solved)
# TODO: implement

# Step 16 - build_sft_trainer (not yet solved)
# TODO: implement

# Step 17 - run_sft_training (not yet solved)
# TODO: implement

# Step 18 - switch_to_inference_mode (not yet solved)
# TODO: implement

# Step 19 - build_chat_prompt (not yet solved)
# TODO: implement

# Step 20 - generate_reply (not yet solved)
# TODO: implement

