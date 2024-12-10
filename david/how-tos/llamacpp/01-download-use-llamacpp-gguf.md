# How to Download and Use LLama CPP in GGUF format

## Go to Hugging Face and look for your model

- Ensure it's in the GGUF format like Llama 3.2 GGUF:
https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF

- Download it
```bash
pip install -U "huggingface_hub[cli]"
cd XXXX
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF --include "Llama-3.2-3B-Instruct-Q4_K_M.gguf" --local-dir ./
```
Create a model

```bash
model = Llama(
    model_path="weights/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    n_ctx=2048,  # Drastically reduce context window
    n_batch=128,  # Smaller batch size
    n_gpu_layers=-1,  # GPU layers
    n_threads=4,  # Fewer threads
    verbose=False
)
```
Use it to prompt:

```bash
# Use a more structured prompt
prompt = f"<|begin_of_text|>Human: What are the main benefits of artificial intelligence?\n\nAssistant:"

output = model(
    prompt, 
    max_tokens=500,  # Limit response length
    stop=["Human:", "<|end_of_text|>"],  # Appropriate stop tokens
    echo=False  # Don't repeat the prompt
)

# Extract the actual text response
full_response = output['choices'][0]['text'].strip()
print(full_response)
```

```bash
```