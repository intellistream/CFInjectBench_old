import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 加载T5模型和tokenizer
model_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 输入问题
question = "Who holds the position of Prime Minister of the United Kingdom?"

# 对问题进行编码
input_ids = tokenizer.encode(question, return_tensors='pt')

# 使用模型进行推理
with torch.no_grad():
    output = model.generate(input_ids)

# 解码输出
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

# 打印输出结果
print(output_text)