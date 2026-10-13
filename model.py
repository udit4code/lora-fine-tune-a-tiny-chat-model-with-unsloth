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

# Step 3 - is_model_4bit_quantized (not yet solved)
# TODO: implement

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

