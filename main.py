import torch
import argparse
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration
import pandas as pd
from textrank import TextRank
from tqdm import tqdm
from ast import literal_eval

def concat(x):
    oneline=''
    x = literal_eval(x)
    for i in x:
        oneline+=i
    return oneline
parser = argparse.ArgumentParser()
parser.add_argument("--infer_path", default=None, type=str)
parser.add_argument("--save_path", default='./output/', type=str)
parser.add_argument("--use_textrank", default=0, type=int)
parser.add_argument('--max_len',type=int, default=512, help='max seq len')
parser.add_argument('--col_name',type=str, default='text_original', help='column name')
args = parser.parse_args()

if args.use_textrank:
    print('use textrank')
    tr=TextRank()
    data = tr.predict(args.infer_path)
else:
    print('use kobart only')
    data = pd.read_csv(args.infer_path)
    # data['article_concat'] = data.article_original.apply(concat)


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

# print(data['textrank_sum'])
model=model.to(device=device)
print('abstract summary ...')
start=0
end=len(data)
# end=10
for i in tqdm(range(start,end)):
    if args.use_textrank:
        text = data['textrank_sum'][i - start]
    else:
        text = data[args.col_name][i - start]
    try:

        if text:
            text = text.replace('\n', '')
            input_ids = tokenizer.encode(text)
            if USE_CUDA:
                input_ids = torch.tensor(input_ids).to(device=device)
            else:
                input_ids = torch.tensor(input_ids)

            print(input_ids.size())
            if input_ids.size()[0] >=1000:
                output = ''
            else:
                input_ids = input_ids.unsqueeze(0)
                output = model.generate(input_ids, eos_token_id=1, max_length=args.max_len, num_beams=4)
                output=output.detach().cpu().numpy()
                output = tokenizer.decode(output[0], skip_special_tokens=True)
            outputs.append(output)
            print(text)
            print(output)
            print()
        else:
            outputs.append('')

    except IndexError or RuntimeError or AttributeError:
        print('err',i)
        outputs.append('')


data['kobart_sum']=outputs
data.to_csv(args.save_path+'test.csv',index=False)
print('finish')


