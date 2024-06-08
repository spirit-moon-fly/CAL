from autobiasdetector import InferenceEngine, create_dataset, CEPEngine, ClusterEngine, BiasInductionEngine

dataset_name='mnli'
model_name='llama2-13b-chat'
model_path='../../../model/llama2-13b-chat'
openai_key=''
#model_name='vicuna-13b-v1.5'
#model_path='../../../model/vicuna-13b-v1.5'
induct_model_name="gpt-4-1106-preview"
# Before using Qwen1.5-72b-chat, you should run the qwen.sh file
#induct_model_name="Qwen1.5-72B-Chat"
dataset=create_dataset(dataset_name,data_dir='../data/'+dataset_name)
inf=InferenceEngine(model_name, dataset, model_path)
inf.inference()
cep=CEPEngine(dataset_name,model_name)
cep.extract(print_ave=True)
clusterengine=ClusterEngine(dataset_name,model_name)
clusterengine.cluster()
BITEngine=BiasInductionEngine(model_name,dataset_name,induct_model_name,openai_key)
BITEngine.induct()
BITEngine.summarize()