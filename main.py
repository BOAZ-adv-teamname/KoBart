import torch
import argparse
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--infer_path", default=None, type=str)
parser.add_argument("--save_path", default='data', type=str)
parser.add_argument("--use_textrank", default=True, type=bool)
args = parser.parse_args()

if args.use_textrank:
     pass
else:
    data = pd.read_csv(args.infer_path, sep='\t')


#pretrain_kobart_model use
def load_model():
    model = BartForConditionalGeneration.from_pretrained('./kobart_summary')
    return model

model = load_model()
tokenizer = get_kobart_tokenizer()
outputs=[]


for text in data.iloc[:,0]:
    if text:
        text = text.replace('\n', '')
        input_ids = tokenizer.encode(text)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.unsqueeze(0)
        output = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        print(output)

data['pre_summ']=outputs
data.head()
data.to_csv(args.save_path+'summary.csv',index=False)


