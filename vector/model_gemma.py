from transformers import AutoTokenizer, Gemma3ForCausalLM

model_id = "google/gemma-3-1b-it"
model = Gemma3ForCausalLM.from_pretrained(
    model_id
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_id)