import argparse
import os
import logging
from config import *
from dataload import create_dataset
from inference import Inference

def create_logger(log_path):

    logging.getLogger().handlers = []

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='google/flan-t5-large', choices=MODEL_SET)
    parser.add_argument('--dataset', type=str, default='mnli', choices=["mnli","HANS","bbq",'unqover','mt_bench','chatbot'])
    parser.add_argument('--model_dir', type=str, default="../../model")
    parser.add_argument('--shot', type=int, default=0)
    parser.add_argument('--generate_len', type=int, default=4)
    parser.add_argument('--debias', action='store_true')
    parser.add_argument('--fs_num', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    return args

def inference(args, inference_model, RESULTS_DIR):
    if args.shot==0:
        if args.debias:
            for prompt in prompt_raw[args.dataset][1:2]:
                for prompt_debias in prompt_debias_set[args.model][args.dataset][1:2]:
                    acc = inference_model.predict(prompt,debias_prompt=prompt_debias)
                    args.logger.info(f"Prompt: {prompt}, acc: {acc}%\n")
                    with open(RESULTS_DIR+args.save_file_name+".txt", "a+") as f:
                        f.write("Prompt: {}, acc: {:.2f}%\n".format(prompt, acc*100))
        else:
            for prompt in prompt_raw[args.dataset][1:2]:
                acc = inference_model.predict(prompt)
                args.logger.info(f"Prompt: {prompt}, acc: {acc}%\n")
                with open(RESULTS_DIR+args.save_file_name+".txt", "a+") as f:
                    f.write("Prompt: {}, acc: {:.2f}%\n".format                                                                                                                                           (prompt, acc*100))
    else:
        if args.fs_num!=-1:
            for prompt in prompt_raw[args.dataset][:1]:
                for i in range(args.fs_num):
                    acc = inference_model.predict(prompt,fs_num=i)
                    args.logger.info(f"Prompt: {prompt}, acc: {acc}%\n")
                    with open(RESULTS_DIR+args.save_file_name+".txt", "a+") as f:
                        f.write("Prompt: {}, acc: {:.2f}%\n".format(prompt, acc*100))
        else:
            for prompt in prompt_raw[args.dataset][:1]:
                acc = inference_model.predict(prompt)
                args.logger.info(f"Prompt: {prompt}, acc: {acc}%\n")
                with open(RESULTS_DIR+args.save_file_name+".txt", "a+") as f:
                    f.write("Prompt: {}, acc: {:.2f}%\n".format(prompt, acc*100))

def main(args):
    save_dir = args.dataset
    save_dir += "/"

    LOGS_DIR = './logs/' + save_dir
    RESULTS_DIR = "./results/" + save_dir

    for DIR in [LOGS_DIR, RESULTS_DIR]:
        if not os.path.isdir(DIR):
            os.makedirs(DIR)

    file_name = args.model.replace('/', '_') + "_gen_len_" + str(args.generate_len) + "_" + str(args.shot) + "_shot"

    args.save_file_name = file_name

    data = create_dataset(args.dataset,args.seed,args.model)

    inference_model = Inference(args)
    args.data = data

    logger = create_logger(LOGS_DIR+file_name+".log")
    logger.info(args)

    args.logger = logger

    inference(args, inference_model, RESULTS_DIR)

if __name__ == '__main__':
    args = get_args()
    main(args)
