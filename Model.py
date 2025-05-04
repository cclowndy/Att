import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load Model
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

model = AutoModelForCausalLM.from_pretrained( 
    "microsoft/Phi-3-mini-4k-instruct",  
    # If you have cuda let change device_map = "cuda"
    device_map="cpu",  
    torch_dtype="auto",  
    trust_remote_code=True,  
) 

# Create a pipeline
gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False, 
    max_new_tokens=50, 
    do_sample=False,
)

prompt = "Write an email to send my love that i love her so much"

outp = gen(prompt)

### I just try directly if you want to be corectly and faster pls download model in local to use it###

