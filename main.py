import torch
import argparse
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration
import pandas as pd
from textrank import TextRank
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--infer_path", default=None, type=str)
parser.add_argument("--save_path", default='output/', type=str)
parser.add_argument("--use_textrank", default=True, type=bool)
args = parser.parse_args()

if args.use_textrank:
    print('use textrank')
    tr=TextRank()
    data = tr.predict(args.infer_path)
else:
    print('use kobart only')
    data = pd.read_csv(args.infer_path, sep='\t')


#pretrain_kobart_model use
def load_model():
    model = BartForConditionalGeneration.from_pretrained('./kobart_summary')
    return model

model = load_model()
tokenizer = get_kobart_tokenizer()
outputs=[]

print(data['pred_sum'])
print('abstract summary ...')
for i in tqdm(range(10)):
    text=data.iloc[i,1]
    if text:
        text = text.replace('\n', '')
        input_ids = tokenizer.encode(text)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.unsqueeze(0)
        output = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        outputs.append(output)
        print(output)

# data['pre_summ']=outputs
# data.to_csv(args.save_path+'summary.csv',index=False)
print('finish')


