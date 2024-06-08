import json
import datasets
from tqdm import tqdm
import random

def readjsonl(filename):
    with open(filename, encoding='utf8') as f:
        datas = f.readlines()
    datas_tmp=[]
    for data in datas:
        data = json.loads(data)
        datas_tmp.append(data)
    return datas_tmp

class Dataset(object):
    def __init__(self):
        self.data = None

    def __len__(self):
        assert self.data is not None, "self.data is None. Please load data first."
        return len(self.data)

    def get_content_by_idx(self, idx, *args):
        raise NotImplementedError(
            "get_content_by_idx() must be implemented in the subclass.")

    def get_few_shot_examples(self, task):
        raise NotImplementedError(
            "get_few_shot_examples() must be implemented in the subclass.")

class GLUE(Dataset):
    def __init__(self, seed=-1, model='llama2-13b-chat'):
        matched = datasets.load_from_disk('./data/mnli')["validation_matched"]
        mismatched = datasets.load_from_disk('./data/mnli')["validation_mismatched"]
        self.data = datasets.concatenate_datasets([matched, mismatched])

        if seed>=0:
            random.seed(seed)
        self.selected = []
        self.model=model
        self.seed=seed
        self.time=0

    def get_few_shot_debias_examples(self, tokenizer=None):
        from few_shot_examples import examples_vicuna, examples_llama2, nega_neutral, negas_contradiction, \
            nega_neutral_vicuna, nega_contradiction_vicuna, nega_entailment_vicuna, nega_neutral_01, examples_llama2_01, \
            nega_contradiction_10, nega_neutral_10, nega_neutral_02, nega_contradiction_02

        debias_prompt = "Note that you shouldn't utilize bias except the logical relationship between the premise and hypothesis to answer."
        if 'vicuna' in self.model:
            num1 = random.randint(0, len(nega_neutral_vicuna) - 1)
            num2 = random.randint(0, len(nega_contradiction_vicuna) - 1)
            num3 = random.randint(0, len(nega_entailment_vicuna) - 1)
            while (num1, num2, num3) in self.selected:
                num1 = random.randint(0, len(nega_neutral_vicuna) - 1)
                num2 = random.randint(0, len(nega_contradiction_vicuna) - 1)
                num3 = random.randint(0, len(nega_entailment_vicuna) - 1)

            self.selected.append((num1, num2, num3))
            self.debias_few_shot_examples = examples_vicuna["mnli"].format(nega_neutral_vicuna[num1], debias_prompt,
                               nega_entailment_vicuna[num3],debias_prompt,nega_contradiction_vicuna[num2],debias_prompt)
        else:
            '''
            #0.2
            num1 = random.randint(0, len(nega_neutral_02) - 1)
            num2 = random.randint(0, len(nega_contradiction_02) - 1)
            while (num1, num2) in self.selected:
                num1 = random.randint(0, len(nega_neutral_02) - 1)
                num2 = random.randint(0, len(nega_contradiction_02) - 1)
            self.selected.append((num1, num2))
            self.debias_few_shot_examples = examples_llama2["mnli"].format(nega_neutral_02[num1], debias_prompt,
                                                            debias_prompt,nega_contradiction_02[num2], debias_prompt)
            '''

            '''
            # 10
            num1 = random.randint(0, len(nega_neutral_10) - 1)
            num2 = random.randint(0, len(nega_contradiction_10) - 1)
            while (num1, num2) in self.selected:
                num1 = random.randint(0, len(nega_neutral_10) - 1)
                num2 = random.randint(0, len(nega_contradiction_10) - 1)
            self.selected.append((num1, num2))
            self.debias_few_shot_examples = examples_llama2["mnli"].format(nega_neutral_10[num1], debias_prompt,debias_prompt,
                                                                           nega_contradiction_10[num2], debias_prompt)
            '''

            '''
            # 0.1
            num1 = random.randint(0, len(nega_neutral_01) - 1)
            while num1 in self.selected:
                num1 = random.randint(0, len(nega_neutral_01) - 1)
            self.selected.append(num1)
            self.debias_few_shot_examples = examples_llama2_01.format(nega_neutral_01[num1], debias_prompt, debias_prompt, debias_prompt)
            '''

            # normal
            num1 = random.randint(0, len(nega_neutral) - 1)
            num2 = random.randint(0, len(negas_contradiction) - 1)
            while (num1, num2) in self.selected:
                num1 = random.randint(0, len(nega_neutral) - 1)
                num2 = random.randint(0, len(negas_contradiction) - 1)
            self.selected.append((num1, num2))
            self.debias_few_shot_examples = examples_llama2["mnli"].format(nega_neutral[num1], debias_prompt, debias_prompt,
                                                                    negas_contradiction[num2], debias_prompt)

        self.time += 1
        if self.time == 10:
            print(self.selected)

    def get_content_by_idx(self, idx, task, *args):
        if task == 'mnli':
            content = 'Premise: ' + self.data[idx]['premise'].strip(' ').strip('.') + '. Hypothesis: ' + self.data[idx]['hypothesis'].strip(' ').strip('.')+'.'
        else:
            raise NotImplementedError

        label = self.data[idx]['label']
        return {"content": content, "label": label}

    def get_few_shot_examples(self, task, fs_num=None):
        if fs_num is not None:
            few_shot_examples = self.debias_few_shot_examples
        else:
            from few_shot_examples import examples
            few_shot_examples = examples[task]
        return few_shot_examples


class HANS(Dataset):
    def __init__(self, data_path,seed=-1,model='llama2-13b-chat'):
        if 'txt' in data_path:
            with open(data_path, "r") as f:
                f.readline()
                lines = f.readlines()
            self.data = []
            for line in lines:
                parts = line.split("\t")
                label = parts[0]
                if label == "non-entailment":
                    label = 2
                elif label == "entailment":
                    label = 0
                else:
                    raise RuntimeError()
                s1, s2= parts[5:7]
                self.data.append({"content":'Premise: '+s1.strip(' ').strip('.').strip(' ')+'. Hypothesis: '+s2.strip(' ').strip('.').strip(' ')+'.',
                                  "label":label,'bias_type':parts[8]})
        else:
            self.data=readjsonl(data_path)

        if seed>=0:
            random.seed(seed)
        self.selected = []
        self.model=model
        self.seed=seed
        self.time=0

    def get_few_shot_debias_examples(self, tokenizer=None):
        from few_shot_examples import examples_vicuna, examples_llama2, nega_neutral, negas_contradiction, \
            nega_neutral_vicuna,nega_contradiction_vicuna,nega_entailment_vicuna, nega_neutral_01, examples_llama2_01, \
            nega_contradiction_10, nega_neutral_10, nega_neutral_02, nega_contradiction_02

        debias_prompt = "Note that you shouldn't utilize bias except the logical relationship between the premise and hypothesis to answer."
        if 'vicuna' in self.model:
            num1 = random.randint(0, len(nega_neutral_vicuna) - 1)
            num2 = random.randint(0, len(nega_contradiction_vicuna) - 1)
            num3 = random.randint(0, len(nega_entailment_vicuna) - 1)
            while (num1, num2, num3) in self.selected:
                num1 = random.randint(0, len(nega_neutral_vicuna) - 1)
                num2 = random.randint(0, len(nega_contradiction_vicuna) - 1)
                num3 = random.randint(0, len(nega_entailment_vicuna) - 1)

            self.selected.append((num1, num2, num3))
            self.debias_few_shot_examples = examples_vicuna["mnli"].format(nega_neutral_vicuna[num1], debias_prompt, nega_entailment_vicuna[num3],
                                                                           debias_prompt, nega_contradiction_vicuna[num2], debias_prompt)
        else:
            '''
            # 0.2
            num1 = random.randint(0, len(nega_neutral_02) - 1)
            num2 = random.randint(0, len(nega_contradiction_02) - 1)
            while (num1, num2) in self.selected:
                num1 = random.randint(0, len(nega_neutral_02) - 1)
                num2 = random.randint(0, len(nega_contradiction_02) - 1)
            self.selected.append((num1, num2))
            self.debias_few_shot_examples = examples_llama2["mnli"].format(nega_neutral_02[num1], debias_prompt,
                                                        debias_prompt, nega_contradiction_02[num2], debias_prompt)
            '''

            '''
            # 10
            num1 = random.randint(0, len(nega_neutral_10) - 1)
            num2 = random.randint(0, len(nega_contradiction_10) - 1)
            while (num1, num2) in self.selected:
                num1 = random.randint(0, len(nega_neutral_10) - 1)
                num2 = random.randint(0, len(nega_contradiction_10) - 1)
            self.selected.append((num1, num2))
            self.debias_few_shot_examples = examples_llama2["mnli"].format(nega_neutral_10[num1], debias_prompt,
                                                            debias_prompt, nega_contradiction_10[num2], debias_prompt)
            '''

            '''
            # 0.1
            num1 = random.randint(0, len(nega_neutral_01) - 1)
            while num1 in self.selected:
                num1 = random.randint(0, len(nega_neutral_01) - 1)
            self.selected.append(num1)
            self.debias_few_shot_examples = examples_llama2_01.format(nega_neutral_01[num1], debias_prompt, debias_prompt, debias_prompt)
            '''

            # normal
            num1 = random.randint(0, len(nega_neutral) - 1)
            num2 = random.randint(0, len(negas_contradiction) - 1)
            while (num1, num2) in self.selected:
                num1 = random.randint(0, len(nega_neutral) - 1)
                num2 = random.randint(0, len(negas_contradiction) - 1)
            self.selected.append((num1, num2))
            self.debias_few_shot_examples = examples_llama2["mnli"].format(nega_neutral[num1], debias_prompt, debias_prompt,
                                                                    negas_contradiction[num2], debias_prompt)
        self.time+=1
        if self.time == 10:
            print(self.selected)

    def get_content_by_idx(self, idx, task=None):
        return self.data[idx]

    def get_few_shot_examples(self, task,fs_num=None):
        if fs_num is not None:
            few_shot_examples = self.debias_few_shot_examples
        else:
            from few_shot_examples import examples
            few_shot_examples = examples["mnli"]
        return few_shot_examples

class Bias(Dataset):
    def __init__(self, data_path, seed,model):
        datas=readjsonl(data_path)
        self.data=[]
        for data in tqdm(datas):
            context=data['context']
            question=data['question']
            all_choices=[data['ans0']+'.',data['ans1']+'.',data['ans2']+'.']
            bias_type=data['category']
            label=data['label']
            self.data.append({"content":[context,question,all_choices[0],all_choices[1],all_choices[2]],"label":label,
                                    'bias_type':bias_type,'category':data['category'],'example_id':data['example_id']})
        if seed>=0:
            random.seed(seed)
        self.selected = []
        self.model=model

    def get_content_by_idx(self, idx, task=None):
        return self.data[idx]

    def get_few_shot_debias_examples(self,tokenizer=None):
        from few_shot_examples import examples_vicuna, examples_llama2,ambiguished_B,ambiguished_C,ambiguished_B_vicuna,ambiguished_C_vicuna,\
            disambiguished_B_vicuna,disambiguished_C_vicuna
        debias_prompt = 'Note that you shouldn\'t utilize bias to make predictions.\n'
        if 'vicuna' in self.model:
            num1 = random.sample(range(0, len(ambiguished_B_vicuna)),2)
            num2 = random.sample(range(0, len(ambiguished_C_vicuna)),2)
            num3 = random.randint(0, len(disambiguished_B_vicuna) - 1)
            num4 = random.randint(0, len(disambiguished_C_vicuna) - 1)
            self.a = ambiguished_B_vicuna[num1[0]]
            self.b = ambiguished_B_vicuna[num1[1]]
            self.c = ambiguished_C_vicuna[num2[0]]
            self.d = ambiguished_C_vicuna[num2[1]]
            self.e = disambiguished_B_vicuna[num3]
            self.f = disambiguished_C_vicuna[num4]
            few_shot_examples = examples_vicuna["bbq"].format(self.a.format(debias_prompt), self.f.format(debias_prompt), self.b.format(debias_prompt),
                                debias_prompt, self.c.format(debias_prompt),debias_prompt, self.d.format(debias_prompt),self.e.format(debias_prompt))
            while (num1[0],num1[1],num2[0],num2[1],num3,num4) in self.selected:
                num1 = random.sample(range(0, len(ambiguished_B_vicuna)), 2)
                num2 = random.sample(range(0, len(ambiguished_C_vicuna)), 2)
                num3 = random.randint(0, len(disambiguished_B_vicuna) - 1)
                num4 = random.randint(0, len(disambiguished_C_vicuna) - 1)
                self.a = ambiguished_B_vicuna[num1[0]]
                self.b = ambiguished_B_vicuna[num1[1]]
                self.c = ambiguished_C_vicuna[num2[0]]
                self.d = ambiguished_C_vicuna[num2[1]]
                self.e = disambiguished_B_vicuna[num3]
                self.f = disambiguished_C_vicuna[num4]
                few_shot_examples = examples_vicuna["bbq"].format(self.a.format(debias_prompt), self.f.format(debias_prompt), self.b.format(debias_prompt),
                                debias_prompt, self.c.format(debias_prompt), debias_prompt, self.d.format(debias_prompt), self.e.format(debias_prompt))
            self.selected.append((num1[0],num1[1],num2[0],num2[1],num3,num4))
        else:
            num1 = random.sample(range(0, len(ambiguished_B)), 2)
            num2 = random.sample(range(0, len(ambiguished_C)), 2)
            self.a = ambiguished_B[num1[0]]
            self.b = ambiguished_B[num1[1]]
            self.c = ambiguished_C[num2[0]]
            self.d = ambiguished_C[num2[1]]
            few_shot_examples = examples_llama2["bbq"].format(self.a.format(debias_prompt), debias_prompt,
                                                       self.b.format(debias_prompt),
                                                       debias_prompt, self.c.format(debias_prompt), debias_prompt,
                                                       self.d.format(debias_prompt), debias_prompt)
            while (num1[0], num1[1], num2[0], num2[1]) in self.selected:
                num1 = random.sample(range(0, len(ambiguished_B)), 2)
                num2 = random.sample(range(0, len(ambiguished_C)), 2)
                self.a = ambiguished_B[num1[0]]
                self.b = ambiguished_B[num1[1]]
                self.c = ambiguished_C[num2[0]]
                self.d = ambiguished_C[num2[1]]
                few_shot_examples = examples_llama2["bbq"].format(self.a.format(debias_prompt), debias_prompt,
                                                           self.b.format(debias_prompt),
                                                           debias_prompt, self.c.format(debias_prompt), debias_prompt,
                                                           self.d.format(debias_prompt), debias_prompt)
            self.selected.append((num1[0], num1[1], num2[0], num2[1]))

        self.debias_few_shot_examples = few_shot_examples

    def get_few_shot_examples(self, task, fs_num=None):
        if fs_num is not None:
            few_shot_examples=self.debias_few_shot_examples
        else:
            from few_shot_examples import examples
            few_shot_examples = examples["bbq"]
        return few_shot_examples

class MT_Bench(Dataset):
    def __init__(self, data_path,seed,model='llama2-13b-chat'):
        datas=readjsonl(data_path)
        self.data=[]
        d={'model_a':0,'model_b':1,'tie':2}
        d_reverse = {'model_a': 1, 'model_b': 0, 'tie': 2}
        for data in tqdm(datas):
            model_a = data['model_a']
            q = data['q']
            a = data['a']
            b = data['b']
            model_b = data['model_b']
            if 'index' in data.keys():
                self.data.append({"content": [q,a,b],"label":d[data['judge']],'model_a':model_a,'model_b':model_b,'index':data['index']})
            else:
                self.data.append({"content": [q, a, b], "label": d[data['judge']], 'model_a': model_a, 'model_b': model_b})
                #self.data.append({"content": [q, b, a], "label": d_reverse[data['judge']], 'model_a': model_a, 'model_b': model_b})
        if seed>=0:
            random.seed(seed)
        self.selected=[]
        self.sampled =[]
        self.model=model
        self.time=0

    def get_content_by_idx(self, idx, task=None):
        return self.data[idx]

    def get_few_shot_debias_examples(self,tokenizer=None):
        from few_shot_examples import examples_vicuna, examples_llama2, nega_A, nega_C, nega_C_vicuna, nega_B_vicuna

        debias_prompt = 'Note that you shouldn\'t utilize bias except the responses\' correctness and effectiveness to make predictions.\n'

        if 'vicuna' in self.model:
            num1 = random.randint(0, len(nega_B_vicuna) - 1)
            num2 = random.randint(0, len(nega_C_vicuna) - 1)
            few_shot_examples = examples_vicuna["mt_bench"].format(debias_prompt, nega_B_vicuna[num1]+debias_prompt+'[[B]]\n',
                                                            nega_C_vicuna[num2]+debias_prompt+'[[C]]\n')
            length = tokenizer([few_shot_examples], padding=False, return_length=True)["length"][0]
            self.sampled.append((num1,num2))
            # In practice, we found that the length of the few-shot prompt has a significant impact on model performance.
            # When the few-shot prompt is long, the model performance significantly decreases.
            # Therefore, we limit the length of the few-shot prompt to 600, which is similar to the length of the few-shot prompt in few-shot baseline.
            while length > 600 or (num1,num2) in self.selected:
                num1 = random.randint(0, len(nega_B_vicuna) - 1)
                num2 = random.randint(0, len(nega_C_vicuna) - 1)
                self.sampled.append((num1, num2))
                few_shot_examples = examples_vicuna["mt_bench"].format(debias_prompt, nega_B_vicuna[num1] + debias_prompt + '[[B]]\n',
                                                                nega_C_vicuna[num2] + debias_prompt + '[[C]]\n')
                length = tokenizer([few_shot_examples], padding=False, return_length=True)["length"][0]
            self.selected.append((num1,num2))
            self.debias_few_shot_examples=examples_vicuna["mt_bench"].format(debias_prompt, nega_B_vicuna[num1] + debias_prompt + '[[B]]\n',
                                                                nega_C_vicuna[num2] + debias_prompt + '[[C]]\n')
        else:
            num1 = random.randint(0, len(nega_A) - 1)
            num2 = random.randint(0, len(nega_C) - 1)
            self.a = nega_A[num1]
            self.c = nega_C[num2]
            few_shot_examples = examples_llama2["mt_bench"].format(nega_A[num1] + debias_prompt + '[[A]]\n', debias_prompt,
                                                            nega_C[num2] + debias_prompt + '[[C]]\n')
            length = tokenizer([few_shot_examples], padding=False, return_length=True)["length"][0]
            self.sampled.append((num1, num2))
            while length > 600 or (num1, num2) in self.selected:
                num1 = random.randint(0, len(nega_A) - 1)
                num2 = random.randint(0, len(nega_C) - 1)
                self.a = nega_A[num1]
                self.c = nega_C[num2]
                self.sampled.append((num1, num2))
                few_shot_examples = examples_llama2["mt_bench"].format(nega_A[num1] + debias_prompt + '[[A]]\n', debias_prompt,
                                                                nega_C[num2] + debias_prompt + '[[C]]\n')
                length = tokenizer([few_shot_examples], padding=False, return_length=True)["length"][0]
            self.selected.append((num1, num2))
            self.debias_few_shot_examples = examples_llama2["mt_bench"].format(nega_A[num1] + debias_prompt + '[[A]]\n', debias_prompt,
                                                                        nega_C[num2] + debias_prompt + '[[C]]\n')
        self.time+=1
        if self.time==10:
            print('******sampled*****')
            print(self.sampled)
            print('******selected*****')
            print(self.selected)

    def get_few_shot_examples(self, task, fs_num=None):
        if fs_num is not None:
            few_shot_examples=self.debias_few_shot_examples
        else:
            from few_shot_examples import examples
            few_shot_examples = examples["mt_bench"]
        return few_shot_examples

class MNLI(Dataset):
    def __init__(self, dir):
        self.data=readjsonl(dir)

    def get_content_by_idx(self, idx, task=None):
        content = 'Premise: ' + self.data[idx]['premise'].strip(' ').strip('.') + '. Hypothesis: ' + self.data[idx]['hypothesis'].strip(' ').strip('.')+'.'
        label = self.data[idx]['label']

        return {"content": content, "label": label,'index':self.data[idx]['idx']}

def create_dataset(dataset_name, seed=-1, model='llama2-13b-chat'):
    if "mnli" in dataset_name:
        if 'llama' not in model and 'vicuna' not in model:
            return MNLI('data/mnli_sampled/sampled.jsonl')
        else:
            return GLUE(seed=seed,model=model)
    elif 'HANS' in dataset_name:
        if 'llama' not in model and 'vicuna' not in model:
            return HANS('data/HANS_sampled/sampled.jsonl')
        else:
            return HANS('data/'+dataset_name+'/heuristics_evaluation_set.txt',seed,model=model)
    elif dataset_name in ['bbq','unqover']:
        if 'llama' not in model and 'vicuna' not in model:
            return Bias('data/'+dataset_name+'/sampled.jsonl',seed,model=model)
        else:
            return Bias('data/'+dataset_name+'/datas.jsonl',seed,model=model)
    elif dataset_name in ['mt_bench','chatbot']:
        if 'llama' not in model and 'vicuna' not in model:
            return MT_Bench('data/'+dataset_name+'/sampled.jsonl',seed)
        else:
            return MT_Bench('data/' + dataset_name + '/data.jsonl', seed, model=model)
    else:
        raise NotImplementedError

