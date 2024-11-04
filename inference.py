from config import LABEL_SET, OPENAI_API
from tqdm import tqdm
import json
import openai
import torch
import os

def write_jsonl(file,train_datas):
    with open(file, 'w', encoding='utf-8') as f:
        for data in train_datas:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')

def readjsonl(filename):
    with open(filename, encoding='utf8') as f:
        datas = f.readlines()
    datas_tmp=[]
    for data in datas:
        data = json.loads(data)
        datas_tmp.append(data)
    return datas_tmp

def getids(filename):
    with open(filename, encoding='utf8') as f:
        datas = f.readlines()
    datas_tmp=[]
    for data in datas:
        data = json.loads(data)

        datas_tmp.append(data[0])
    return datas_tmp

class Inference(object):

    def __init__(self, args):
        self.error_analysis = False
        self.args = args
        self.model = args.model
        self.create_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_model(self):

        if self.model not in ['chatgpt', 'gpt4']:


            if self.model.lower() =='llama2-13b-chat':

                from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM

                model_dir = os.path.join(self.args.model_dir, self.model)
                print(model_dir)

                self.tokenizer = LlamaTokenizer.from_pretrained(model_dir, device_map="auto")
                self.tokenizer.pad_token = '[PAD]'
                self.tokenizer.padding_side = 'left'
                self.pipe = LlamaForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.float16)

            elif self.model.lower()=="vicuna-13b-v1.5":

                from transformers import AutoModelForCausalLM, AutoTokenizer

                model_dir = os.path.join(self.args.model_dir, self.model)

                self.tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", use_fast=False)
                self.tokenizer.padding_side = 'left'
                self.pipe = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.float16)

            else:
                raise NotImplementedError("The model is not implemented!")

    def process_input(self, prompt, raw_data,fs_num=None,gpt=False):
        if self.args.dataset in ["mnli","HANS"]:
            return self._process_cls_input(prompt, raw_data,fs_num,gpt=gpt)
        elif self.args.dataset in ['bbq','unqover']:
            return self._process_bias_input(prompt, raw_data,fs_num,gpt=gpt)
        elif self.args.dataset in ['mt_bench','chatbot']:
            return self._process_dialog_input(prompt, raw_data,fs_num,gpt=gpt)
        else:
            raise NotImplementedError("The dataset is not implemented!")

    def eval(self, preds, gts):
        if not isinstance(preds, list):
            preds = [preds]
            gts = [gts]
        return sum(a == b for a, b in zip(preds, gts)) / len(preds)


    def predict(self, prompt=None, debias_prompt='', fs_num=None):
        if self.model in ["chatgpt", "gpt4"]:
            results = self.predict_by_openai_api(self.model, prompt, debias_prompt)
        else:
            results = self.predict_by_local_inference(self.model, prompt, debias_prompt,fs_num=fs_num)
        return results

    def predict_by_openai_api(self, model, prompt, debias_prompt):
        data_len = len(self.args.data)
        if debias_prompt != '':
            if prompt[-2:] == ':\n':
                prompt = prompt.strip('\n').strip(':').strip('. ') + '. Note that ' + debias_prompt
            elif prompt[-2:] == '\n\n':
                prompt=prompt.strip('\n')+ 'Note that ' + debias_prompt
            else:
                prompt = prompt + 'Note that ' + debias_prompt

        preds, gts = [], []
        if self.args.debias:
            dir = 'data/'+self.args.dataset+'/gpt4_1106_debias.jsonl'
        else:
            dir = 'data/' + self.args.dataset + '/gpt4_1106.jsonl'
        f1 = open(dir, 'a+', encoding='utf-8')
        ids = getids(dir)
        for idx in tqdm(range(data_len)):
            raw_data = self.args.data.get_content_by_idx(idx, self.args.dataset)
            id_now = raw_data['example_id'] if 'example_id' in raw_data.keys() else raw_data['index']
            if id_now in ids:
                continue
            input_text, gt = self.process_input(prompt, raw_data,gpt=True)
            raw_pred = self.call_openai_api(model, input_text)
            pred = self.process_pred(raw_pred)
            json.dump((id_now,pred,gt), f1, ensure_ascii=False)
            f1.write('\n')
            f1.flush()
            preds.append(pred)
            gts.append(gt)

        score = self.eval(preds, gts)
        print(score)
        return score

    def predict_by_local_inference(self, model, prompt, debias_prompt, fs_num=None):
        data_len = len(self.args.data)
        if fs_num is not None:
            self.args.data.get_few_shot_debias_examples(self.tokenizer)

        preds_type, preds = {}, []
        gts_type, gts = {}, []
        examples=[]
        if debias_prompt != '':
            if prompt[-2:]==':\n':
                prompt = prompt.strip('\n').strip(':').strip('. ') + '. Note that ' + debias_prompt
            elif prompt[-2:] == '\n\n':
                prompt=prompt.strip('\n')+ 'Note that ' + debias_prompt
            else:
                prompt = prompt+'Note that ' + debias_prompt
        scores_l=[]

        for idx in tqdm(range(data_len)):
            ## These two pieces of data are too long, so remove these two pieces of data
            if self.args.dataset=='chatbot' and (idx==2635 or idx==8995):
                continue
            example={}
            raw_data = self.args.data.get_content_by_idx(idx, self.args.dataset)
            input_text, gt, label_spaces_ids= self.process_input(prompt, raw_data, fs_num=fs_num)
            if 'bias_type' in raw_data.keys():
                if raw_data['bias_type'] not in gts_type.keys():
                    gts_type[raw_data['bias_type']] = [gt]
                    preds_type[raw_data['bias_type']] = []
                else:
                    gts_type[raw_data['bias_type']].append(gt)
            if 'example_id' in raw_data.keys():
                example['example_id'] = raw_data['example_id']
            example['content'] = input_text
            if idx==0:
                print(input_text)
                print()

            example['gold']=gt
            gts.append(gt)
            origin_input = self.tokenizer(input_text, return_attention_mask=True, return_tensors="pt")
            input_ids = origin_input.input_ids.to(self.device)
            attention_mask = origin_input.attention_mask.to(self.device)
            option_num, max_seq_len = label_spaces_ids.shape
            # Expand the input to obtain the probability of generating each answer separately
            input_ids_tmp, attention_mask_tmp = torch.zeros((option_num, input_ids.shape[1]),dtype=torch.long).to(self.device),\
                torch.zeros((option_num, attention_mask.shape[1]), dtype=torch.long).to(self.device)
            for j in range(option_num):
                input_ids_tmp[j] = input_ids
                attention_mask_tmp[j] = attention_mask
            total_scores = torch.zeros((option_num, max_seq_len),dtype=torch.float).to(self.device)
            input_ids = input_ids_tmp
            attention_mask = attention_mask_tmp

            for i in range(max_seq_len):
                if i==0:
                    inputs = {
                        "input_ids": input_ids[0:1],
                        "attention_mask": attention_mask[0:1],
                        "max_new_tokens": 1
                    }
                else:
                    inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "max_new_tokens": 1
                    }
                outputs = self.pred_by_generation(inputs, model)
                scores = outputs.scores
                logits = torch.stack(scores, dim=1)
                softmax = torch.nn.Softmax(dim=-1)
                logits = softmax(logits)

                hidden_states=outputs.hidden_states
                if i==0:
                    # The hidden state of the last token in the last layer
                    representations = softmax(hidden_states[0][-1][0][-1]).cpu().tolist()
                    example['representations']=representations
                    for k in range(option_num):
                        total_scores[k][i] = logits[0][0][label_spaces_ids[k][i]]
                else:
                    for k in range(option_num):
                        if label_spaces_ids[k][i]!=self.tokenizer.pad_token_id:
                            total_scores[k][i] = logits[k][0][label_spaces_ids[k][i]]
                        else:
                            total_scores[k][i] = 1
                input_ids = torch.cat([input_ids, label_spaces_ids[:, i].view(-1,1)], dim=-1).long()
                attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=torch.long).to(self.device)],dim=-1).long()

            total_scores = torch.sum(torch.log(total_scores), dim=-1)
            pred = torch.argmax(total_scores, dim=-1).item()
            example['pred'] = pred
            example['scores'] = total_scores.cpu().tolist()
            examples.append(example)
            preds.append(pred)
            if 'bias_type' in raw_data.keys():
                preds_type[raw_data['bias_type']].append(pred)
            scores_l.append(total_scores.cpu().tolist())

        score = self.eval(preds, gts)
        print(score)
        if len(raw_data)>0:
            for key in preds_type.keys():
                print(key,self.eval(preds_type[key], gts_type[key]))
        if self.args.dataset in ['mnli','chatbot'] and self.args.shot==0 and not self.args.debias:
            print('Writing file')
            if 'vicuna' in self.model:
                write_jsonl('data/'+self.args.dataset+'/vicuna/examples.jsonl',examples)
            else:
                write_jsonl('data/' + self.args.dataset + '/llama2/examples_0.2.jsonl', examples)

        if self.args.dataset in ['chatbot', 'mt_bench']:
            if 'vicuna' in self.model:
                #write_jsonl('data/' + self.args.dataset + '/vicuna/pred_reverse.jsonl', (preds, gts))
                write_jsonl('data/' + self.args.dataset + '/vicuna/pred_origin.jsonl', (preds, gts))
            else:
                #write_jsonl('data/' + self.args.dataset + '/llama2/pred_reverse.jsonl', (preds, gts))
                write_jsonl('data/' + self.args.dataset + '/llama2/pred_origin.jsonl', (preds, gts))

        return score

    def call_openai_api(self, model, prompt):
        openai.api_key = OPENAI_API
        if model in ['chatgpt']:
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo-1106",messages=[{"role": "user", "content": prompt}],temperature=0,top_p=0)
            result = response['choices'][0]['message']['content']
        else:
            response = openai.ChatCompletion.create(model='gpt-4-1106-preview',messages=[{"role": "user", "content": prompt}],temperature=0,top_p=0)
            result = response['choices'][0]['message']['content']
        return result

    def pred_by_generation(self, inputs, model):
        outputs = self.pipe.generate(**inputs,temperature=0,early_stopping=True,return_dict_in_generate=True,output_scores=True,output_hidden_states=True)
        return outputs

    def _process_cls_input(self, prompt, raw_data, fs_num=None, gpt=False):
        content = raw_data["content"]
        label = raw_data["label"]

        input_text = prompt

        if gpt is True:
            return input_text+content+'\nAnswer:\n',label

        # few-shot
        if self.args.shot > 0:
            few_shot_examples = self.args.data.get_few_shot_examples(self.args.dataset,fs_num=fs_num)
            input_text += few_shot_examples
            input_text += (content + "\nAnswer:\n")
        else:
            # zero-shot
            input_text += (content + "\nAnswer:\nThe relationship between the two provided sentences is '")
        label_spaces = LABEL_SET[self.args.dataset]

        if 'llama' in self.tokenizer.name_or_path or 'vicuna' in self.tokenizer.name_or_path:
            self.tokenizer.padding_side = 'right'
            length = self.tokenizer(["'" + label_space for label_space in label_spaces], padding=False, return_length=True)["length"]
            length = max(length)
            label_spaces_ids = self.tokenizer(["'" + label_space for label_space in label_spaces], padding="max_length", max_length=length,
                            return_tensors="pt")["input_ids"]
            self.tokenizer.padding_side = 'left'
            label_spaces_ids = torch.tensor(label_spaces_ids, dtype=torch.int32).to(self.device)
            label_spaces_ids = label_spaces_ids[:, 1:]
            label_spaces_ids = label_spaces_ids[:, 1:]
        else:
            length = self.tokenizer(label_spaces, padding=False, return_length=True)["length"]
            length = max(length)
            label_spaces_ids = self.tokenizer(label_spaces, padding="max_length", max_length=length,return_tensors="pt")["input_ids"]
            label_spaces_ids = torch.tensor(label_spaces_ids, dtype=torch.int32).to(self.device)

        return input_text, label, label_spaces_ids

    def _process_bias_input(self, prompt, raw_data, fs_num=None, gpt=False):
        content = raw_data["content"]
        label = raw_data["label"]

        input_text = prompt.format(content[0],content[1],content[2],content[3],content[4])
        if gpt is True:
            return input_text+'Answer:\n',label

        if self.args.shot > 0:
            few_shot_examples = self.args.data.get_few_shot_examples(self.args.dataset,fs_num)
            input_text = few_shot_examples+input_text

        if fs_num != None:
            input_text += 'Note that you shouldn\'t utilize bias to make predictions.\n'
        input_text+='Answer:\n'

        # prepare label_spaces
        label_spaces = LABEL_SET[self.args.dataset][:3]
        # print(label_spaces)
        if 'llama' in self.tokenizer.name_or_path or 'vicuna' in self.tokenizer.name_or_path:
            length = self.tokenizer(["\n" + label_space for label_space in label_spaces], padding=False, return_length=True)["length"]
            length = max(length)
            label_spaces_ids = self.tokenizer(["\n" + label_space for label_space in label_spaces], padding="max_length", max_length=length,
                            return_tensors="pt")["input_ids"]
            label_spaces_ids = torch.tensor(label_spaces_ids, dtype=torch.int32).to(self.device)
            label_spaces_ids = label_spaces_ids[:, 2:]
            label_spaces_ids = label_spaces_ids[:, 1:]
        else:
            length = self.tokenizer(label_spaces, padding=False, return_length=True)["length"]
            length = max(length)
            label_spaces_ids = self.tokenizer(label_spaces, padding="max_length",max_length=length,return_tensors="pt")["input_ids"]
            label_spaces_ids = torch.tensor(label_spaces_ids, dtype=torch.int32).to(self.device)
            label_spaces_ids = label_spaces_ids[:, :-1]

        return input_text, label, label_spaces_ids

    def _process_dialog_input(self, prompt, raw_data, fs_num=None, gpt=False):
        content = raw_data["content"]
        label = raw_data["label"]

        input_text = prompt

        if self.args.shot > 0:
            few_shot_examples = self.args.data.get_few_shot_examples(self.args.dataset,fs_num)
            input_text += few_shot_examples

        input_text = (input_text + "[User Question]\n{}\n\n[The Start of Assistant A's Answer]\n{}\n[The End of Assistant A's Answer]\n\n"
                         "[The Start of Assistant B's Answer]\n{}\n[The End of Assistant B's Answer]\n[[").format(content[0], content[1], content[2])


        if gpt is True:
            return input_text[:-2], label

        # prepare label_spaces
        label_spaces = LABEL_SET[self.args.dataset][:3]

        if 'llama' in self.tokenizer.name_or_path or 'vicuna' in self.tokenizer.name_or_path:
            length = self.tokenizer(["[[" + label_space for label_space in label_spaces], padding=False, return_length=True)["length"]
            length = max(length)
            label_spaces_ids = self.tokenizer(["[[" + label_space for label_space in label_spaces], padding="max_length", max_length=length,
                            return_tensors="pt")["input_ids"]
            label_spaces_ids = torch.tensor(label_spaces_ids, dtype=torch.int32).to(self.device)
            label_spaces_ids = label_spaces_ids[:, 1:]
            label_spaces_ids = label_spaces_ids[:, 1:]
        else:
            length = self.tokenizer(label_spaces, padding=False, return_length=True)["length"]
            length = max(length)
            label_spaces_ids = self.tokenizer(label_spaces, padding="max_length",max_length=length,return_tensors="pt")["input_ids"]
            label_spaces_ids = torch.tensor(label_spaces_ids, dtype=torch.int32).to(self.device)
            label_spaces_ids = label_spaces_ids[:, :-1]

        return input_text, label, label_spaces_ids



