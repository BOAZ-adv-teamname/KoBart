import torch
import argparse
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration
import pandas as pd
from textrank import TextRank
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--infer_path", default=None, type=str)
parser.add_argument("--save_path", default='./output/', type=str)
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
#gpu option
USE_CUDA = torch.cuda.is_available()
print('gpu:',USE_CUDA)
device = torch.device('cuda' if USE_CUDA else 'cpu')

print(data['textrank_sum'])
print('abstract summary ...')
start=2160
end=2500
for i in tqdm(range(start,end)):
    try:
        text = data['textrank_sum'][i - start]
        if text:
            text = text.replace('\n', '')
            input_ids = tokenizer.encode(text)
            # input_ids = torch.tensor(input_ids).to(torch.int64).long().to(device=device)
            input_ids = torch.tensor(input_ids)
            input_ids = input_ids.unsqueeze(0)
            output = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
            output=output.detach().cpu().numpy()
            output = tokenizer.decode(output[0], skip_special_tokens=True)
            outputs.append(output)
            print(output)
        else:
            outputs.append('')

    except IndexError:
        print('err',i)
        outputs.append('')


data['kobart_sum']=outputs
data.to_csv(args.save_path+'summary_gpu.csv',index=False)
print('finish')


