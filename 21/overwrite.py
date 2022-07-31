import requests
#r = requests.post('http://10.30.99.211:8261/gpt_small', data = "What is better for deep learning Python or Matlab?")
#print (r.status_code)

# numpy
import numpy as np
import argparse

# torch
import torch, torch.nn as nn
import torch.nn.functional as F

# pandas
import pandas as pd

# transformers
import pytorch_transformers
from pytorch_transformers import *

# read file
from xml.dom import minidom
import re

#import rank model
from sklearn.ensemble import RandomForestRegressor

# custom extractor
from my_functions import extractorRoberta
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
my_extractor = extractorRoberta(my_device = device, model_path = '/notebook/cqas/external_pretrained_models/')
print ("loaded extractors")


    
def write_qrels(output_dir, name, rtr):
    qids = rtr['qid']
    Q0s = [0 for elem in qids]
    docs = rtr['docno']
    scores = rtr['score']
    ranks = rtr['rank']
    tags = [name for elem in qids]
    common_list = list(zip(qids, Q0s, docs, ranks, scores, tags))
    with open(output_dir + name +'.qrels', 'w') as fp:
        fp.write('\n'.join('%s %s %s %s %s %s' % x for x in common_list))
    print ("written " + name +'.txt')

    
def run_baseline(output_dir = '/notebook/touche/output/', input_dir = '/notebook/touche/', input_file = 'topics-task-2-only-titles.xml'):
    
    topics_2020 = read_xml('topics-task-2.xml')
        
    topics = read_xml('topics-task-2-only-titles-2021.xml')
        
    with open('/notebook/touche/list_of_un_answ.pkl', 'rb') as f:
        answers_2020 = pickle.load(f)
            
    with open('/notebook/touche/list_of_un_answ_2021.pkl', 'rb') as f:
        answers_2021 = pickle.load(f)
    
    with open('/notebook/touche2021/touche2020-task2-relevance-withbaseline.qrels', 'r') as f:
        qrels_lines = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
    qrels = [x.strip().split() for x in qrels_lines] 

    qrels_dict = {}
    for elem in qrels:
        query, noninf, docno, rank = elem
        if (query in qrels_dict.keys()):
            qrels_dict[query].append((docno, rank))
        else:
            qrels_dict[query] = []
            qrels_dict[query].append((docno, rank))
            
    info_df = pd.DataFrame(columns=["qid", "query", "docno", "text", "baseline_scores", "is_retrieved", "ap_score", "objs_score"])
    info_df_train = pd.DataFrame(columns=["qid", "query", "docno", "text", "baseline_scores", "is_retrieved", "ap_score", "objs_score"])
    qrels_df = pd.DataFrame(columns=["qid", "docno", "label"])
    qrels_df_train = pd.DataFrame(columns=["qid", "docno", "label"])

    #create test df
    
    for elem in topics_2021:
        qid, query = elem[0], elem[1].strip('\n')
        query = re.sub(r'[^\w\s]','',query)
        query = cleanhtml(query)
        my_extractor.from_string(query)
        structures = my_extractor.get_params()
        
        for ind, answer in enumerate(answers_2021[qid]):
            docno = answer[1]
            score = answer[0]
            text = answer[3]

            nlu_score = count_score(text, structures)
            objs_score = count_score_obj(text, structures)
            ap_score = count_score_asp_pred(text, structures)
            is_retrieved = count_score_nlu(structures)
            df_row = {"qid":qid, "query":query, "docno":docno, "text":text, "baseline_scores":score, "is_retrieved":is_retrieved, "ap_score":ap_score, "objs_score":objs_score}
            info_df = info_df.append(df_row, ignore_index= True)
            
    #create train df and crels
    
    for elem in topics_2020:
        qid, query = elem[0], elem[1].strip('\n')
        query = re.sub(r'[^\w\s]','',query)
        query = cleanhtml(query)
        my_extractor.from_string(query)
        structures = my_extractor.get_params()

        for ind, answer in enumerate(answers_2020[qid]):
            docno = answer[1]
            score = answer[0]
            text = answer[3]

            nlu_score = count_score(text, structures)
            objs_score = count_score_obj(text, structures)
            ap_score = count_score_asp_pred(text, structures)
            is_retrieved = count_score_nlu(structures)
            df_row = {"qid":qid, "query":query, "docno":docno, "text":text, "baseline_scores":score, "is_retrieved":is_retrieved, "ap_score":ap_score, "objs_score":objs_score}
            info_df_train = info_df_train.append(df_row, ignore_index= True)
                
        for qrel in qrels_dict[qid]:
            docno, label = qrel
            df_row = {"qid":qid, "docno":docno, "label":label}
            qrels_df_train = qrels_df_train.append(df_row, ignore_index= True)
       
    
    print ("info_df head (2021) ", len(info_df), info_df.head())
    print ("info_df_test head (2020) ", len(info_df_train), info_df_train.head())
    print ("qrels_df_train ", len(qrels_df_train), qrels_df_train.head())
    
    
    test_ds = create_featured_dataset(info_df)
    result = create_featured_dataset(info_df_train)
    
    test_ds.to_pickle("/home/katana21/runs/features/test_ds.pkl")
    result.to_pickle("/home/katana21/runs/features/result.pkl")
    qrels_df.to_pickle("/home/katana21/runs/features/qrels_df")
    qrels_df_train.to_pickle("/home/katana21/runs/features/qrels_df_train")
    
    rf = RandomForestRegressor(n_estimators=20)
    rf_pipe = pt.ltr.apply_learned_model(rf)
    rf_pipe.fit(result, qrels_df)
    answs = transform(rf_pipe, test_ds)
    
    write_qrels(output_dir, "random_forest", answs)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('--i', default='/notebook/touche2021/')
    parser.add_argument('--o', default='/notebook/touche/output2021/')
    parser.add_argument('--inp_file', default = 'topics.xml')
    args = parser.parse_args()
    run_baseline(output_dir = args.o, input_dir = args.i, input_file = args.inp_file)