from transformers import T5ForConditionalGeneration
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
model = T5ForConditionalGeneration.from_pretrained()

def summarize_code(code_block: str):
    input_ids = tokenizer(code_block, return_tensors='pt').input_ids
    
    # generate
    outputs = model.generate(input_ids)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)