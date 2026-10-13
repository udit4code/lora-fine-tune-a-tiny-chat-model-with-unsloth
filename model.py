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

# Step 4 - ensure_pad_token (not yet solved)
# TODO: implement

# Step 5 - get_lora_target_modules (not yet solved)
# TODO: implement

# Step 6 - attach_lora_adapters (not yet solved)
# TODO: implement

# Step 7 - count_trainable_parameters (not yet solved)
# TODO: implement

# Step 8 - trainable_fraction (not yet solved)
# TODO: implement

# Step 9 - build_instruction_examples (not yet solved)
# TODO: implement

# Step 10 - format_instruction_example (not yet solved)
# TODO: implement

# Step 11 - format_all_examples (not yet solved)
# TODO: implement

# Step 12 - build_text_dataset (not yet solved)
# TODO: implement

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

