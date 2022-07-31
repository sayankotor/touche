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
import pickle
from my_functions import extractorRoberta
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
my_extractor = extractorRoberta(my_device = device, model_path = '/notebook/cqas/external_pretrained_models/')
print ("loaded extractors")

from timeit import default_timer as timer
from pyterrier import text
from pyterrier.text import scorer

import pyterrier as pt
if not pt.started():
  print ("not")  
  pt.init()

def create_featured_dataset(some_df):
    from pyterrier import text
    from pyterrier.text import scorer
    textscorerTf = text.scorer(body_attr="text", wmodel='BM25', sort=False)
    rtr_bm = textscorerTf.transform(some_df)
    textscorerTf = text.scorer(body_attr="text", wmodel='Tf')
    rtr_tf = textscorerTf.transform(some_df)
    textscorerTf = text.scorer(body_attr="text", wmodel='PL2')
    rtr_pl2 = textscorerTf.transform(some_df)
    textscorerTf = text.scorer(body_attr="text", wmodel='DFIC')
    rtr_dfic = textscorerTf.transform(some_df)
    
    rtr_pl2_for_merge = rtr_pl2[['qid', 'docno', 'score']]
    rtr_pl2_for_merge = rtr_pl2_for_merge.rename(columns={"score": "score_pl2"})
    
    rtr_tf_for_merge = rtr_tf[['qid', 'docno', 'score']]
    rtr_tf_for_merge = rtr_tf_for_merge.rename(columns={"score": "score_tf"})
    
    rtr_bm_for_merge = rtr_bm[['qid', 'docno', 'score']]
    rtr_bm_for_merge = rtr_bm_for_merge.rename(columns={"score": "score_bm"})
    
    rtr_dfic_for_merge = rtr_dfic[['qid', 'docno', 'score']]
    rtr_dfic_for_merge = rtr_dfic_for_merge.rename(columns={"score": "score_dfic"})
    
    result = pd.merge(rtr_pl2_for_merge, rtr_tf_for_merge, on=["qid", "docno"])
    result = pd.merge(result, rtr_bm_for_merge, on=["qid", "docno"])
    result = pd.merge(result, rtr_dfic_for_merge, on=["qid", "docno"])
    result = pd.merge(result, some_df, on=["qid", "docno"])
    zipped = [result["score_pl2"], result["score_tf"], result["score_bm"], result["score_dfic"], result['baseline_scores'], result["is_retrieved"], result["ap_score"], result["objs_score"]]
    unzipped_object = zip(*zipped)
    unzipped_list = list(unzipped_object)
    list_of_features = [np.array(elem) for elem in unzipped_list]
    result['features'] = list_of_features
    return result

def extract_objs_asp(model_for_extraction, input_string):
    model_for_extraction.from_string(input_string)
    obj1, obj2, predicates, aspects = model_for_extraction.get_params()
    return (obj1.lower(), obj2.lower(), predicates, aspects)


def count_score1(text, nlu_tuple):
    (obj1, obj2, pred, asp) = nlu_tuple
    r = 1.0
    if (len(obj1) != 0 and len(obj2) != 0):
        if (len(pred) != 0):
            pred = re.sub('[!#?,.:";]', '', pred[0])
            if (obj1 in text and obj2 in text and pred in text):
                r += 1.0
        if (len(asp) != 0):
            asp = re.sub('[!#?,.:";]', '', asp[0])
            if (obj1 in text and obj2 in text and asp in text):
                r += 1.0
        elif (obj1 in text and obj2 in text):
            r = 1.5
        elif (obj1 in text or obj2 in text):
            r = 1.2
    else:
        if (obj1) in text or (obj2) in text:
            r = 1.2
    return r

def count_score_nlu(nlu_tuple):
    if (len(nlu_tuple[0]) == 0):
        return 0.0
    else: return 1.0


def count_score(text, nlu_tuple):
    (obj1, obj2, preds, asps) = nlu_tuple
    r = 0.0
    text = cleanhtml(text)
    if (len(obj1) != 0 and len(obj2) != 0):
        if (obj1 in text):
            r += 1.0
        if (obj2 in text):
            r += 1.0
        for asp in asps:
            if asp in text:
                r += 1.5
        for pred in preds:
            if pred in text:
                r += 1.0
    else:
        if ((obj1) in text and len(obj1)!= 0) or (obj2 in text and len(obj2) != 0):
            r = 1.0
    return r

def count_score_obj(text, nlu_tuple):
    (obj1, obj2, preds, asps) = nlu_tuple
    r = 0.0
    text = cleanhtml(text)
    if (len(obj1) != 0 and obj1 in text):
        r += 1.0
    if (len(obj2) != 0 and obj2 in text):
        r += 1.0
    return r

def count_score_asp_pred(text, nlu_tuple):
    (obj1, obj2, preds, asps) = nlu_tuple
    r = 0.0
    text = cleanhtml(text)
    o1 = (len(obj1) != 0 and obj1 in text)
    o2 = (len(obj2) != 0 and obj2 in text)
    if (o1 or o2):
        for asp in asps:
            if asp in text:
                r += 0.5
        for pred in preds:
            if pred in text:
                r += 0.5
    return r


def make_scores_obj(query, answers):
    print ("make_scores_obj")
    (obj1, obj2, pred, asp) = extract_objs_asp(extr, query)
    print ("in make scores", obj1, obj2, pred, asp)
    scores_answers = [count_score(cleanhtml(answer), (obj1, obj2, pred, asp)) for answer in answers]
    return scores_answers

def read_xml(filename):
    # convert file filename to list of tuples (number_of_topic, title_of_topic) 
    # input: filename string
    # output: list of corresponding tuples
    answer_list = []
    xmldoc = minidom.parse(filename)
    itemlist = xmldoc.getElementsByTagName('topics')
    print(len(itemlist))
    print(itemlist)
    topic_list = itemlist[0].getElementsByTagName('topic')
    print (len(topic_list))
    for topic in topic_list:
        tuple_for_add = tuple((topic.getElementsByTagName('number')[0].firstChild.nodeValue, topic.getElementsByTagName('title')[0].firstChild.nodeValue))
        answer_list.append(tuple_for_add)
    return answer_list

def make_a_search_request(query):
    # return json
    # json will be processed further
    params = {
            "apikey": "0833a307-97d3-462a-99d9-27db400c70da",
            "query": query,
            "index": ["cw12"],
            "size": 10,
            "pretty": True
        }
    response = requests.get(url = "http://www.chatnoir.eu/api/v1/_search", params = params)
    return response

def clean_punct(s):
    s = re.sub(r'[^\w\s]','',s)
    return s

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext.lower()

def add_ranks1(rtr : pd.DataFrame) -> pd.DataFrame:
    """
        Canonical method for adding a rank column which is calculated based on the score attribute
        for each query. Note that the dataframe is NOT sorted by this operation.
        Arguments
            df: dataframe to create rank attribute for
    """
    rtr.drop(columns=["rank"], errors="ignore", inplace=True)
    if len(rtr) == 0:
        rtr["rank"] = pd.Series(index=rtr.index, dtype='int64')
        return rtr
    # -1 assures that first rank will be FIRST_RANK
    rtr["rank"] = rtr.groupby("qid", sort=False)["score"].rank(ascending=False, method="first").astype(int) -1 + 1
    if True:
        rtr.sort_values(["qid", "rank"], ascending=[True,True], inplace=True)
    return rtr


def transform(model, test_DF):
    """
    Predicts the scores for the given topics.

    Args:
        topicsTest(DataFrame): A dataframe with the test topics.
    """
    test_DF = test_DF.copy()

    # check for change in number of features
    found_numf = test_DF.iloc[0].features.shape[0]
    if model.num_f is not None:
        if found_numf != model.num_f:
            raise ValueError("Expected %d features, but found %d features" % (model.num_f, found_numf))
    if hasattr(model.learner, 'feature_importances_'):
        if len(model.learner.feature_importances_) != found_numf:
            raise ValueError("Expected %d features, but found %d features" % (len(model.learner.feature_importances_), found_numf))

    test_DF["score"] = model.learner.predict(np.stack(test_DF["features"].values))
    test_DF["qid"] = [int(elem) for elem in test_DF["qid"]]
    return add_ranks1(test_DF)
    
def write_qrels(output_dir, name, rtr):
    print ("rtr in write qrels ", rtr.head())
    qids = rtr['qid']
    Q0s = [0 for elem in qids]
    docs = rtr['docno']
    scores = rtr['score']
    ranks = rtr['rank']
    tags = [name for elem in qids]
    common_list = list(zip(qids, Q0s, docs, ranks, scores, tags))
    print ("common_list ", common_list[:3])
    with open(output_dir + name +'.qrels', 'w') as fp:
        fp.write('\n'.join('%s %s %s %s %s %s' % x for x in common_list))
    print ("written " + name +'.txt')


    
def run_baseline(output_dir = '/notebook/touche/output/', input_dir = '/notebook/touche/', input_file = 'topics-task-2-only-titles.xml'):
    
    topics_2020 = read_xml('topics-task-2.xml')
        
    topics_2021 = read_xml('topics-task-2-only-titles-2021.xml')
        
    with open(input_dir + 'list_of_un_answ.pcl', 'rb') as f:
        answers_2020 = pickle.load(f)
            
    with open(input_dir + 'list_of_un_answ_2021.pcl', 'rb') as f:
        answers_2021 = pickle.load(f)
    
    with open(input_dir + 'touche2020-task2-relevance-withbaseline.qrels', 'r') as f:
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
            
    info_df = pd.DataFrame(columns=["qid", "query", "docno", "text", "baseline_scores", "is_retrieved", "ap_score", "objs_score"],dtype=object)
    info_df_train = pd.DataFrame(columns=["qid", "query", "docno", "text", "baseline_scores", "is_retrieved", "ap_score", "objs_score"], dtype=object)
    qrels_df = pd.DataFrame(columns=["qid", "docno", "label"],dtype=object)
    qrels_df_train = pd.DataFrame(columns=["qid", "docno", "label"],dtype=object)

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

            #nlu_score = count_score(text, structures)
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
    
    print (result['features'][:3])
    
    import xgboost as xgb
# this configures XGBoost as LambdaMART
    lmart_x = xgb.sklearn.XGBRanker(objective='rank:ndcg',
          learning_rate=0.1,
          gamma=1.0,
          min_child_weight=0.1,
          max_depth=6,
          verbose=2,
          random_state=42)
    
    val_ds = pd.read_pickle("/notebook/touche2021/test_ds.pkl")
    qrels_val_ds = pd.read_pickle("/notebook/touche2021/qrels_df_test.pkl")
    
    rf_lgbm = pt.ltr.apply_learned_model(lmart_x, form="ltr")
    rf_lgbm.fit(result, qrels_df_train, topics_and_results_Valid = val_ds, qrelsValid = qrels_val_ds)
    answs = transform(rf_lgbm, test_ds)
    
    # rf = RandomForestRegressor(n_estimators=20)
    # rf_pipe = pt.ltr.apply_learned_model(rf)
    # rf_pipe.fit(result, qrels_df_train)
    # answs = transform(rf_pipe, test_ds)
    # answs = answs.sort_values(by=['qid'])
    print ("answs head ", answs.head())
    write_qrels(output_dir, "xgboost_v1", answs)
    
    answs_test = transform(rf_lgbm, val_ds)
    write_qrels(output_dir, "xgb_v2_test", answs_test)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('--i', default='/notebook/touche2021/runs/')
    parser.add_argument('--o', default='/notebook/touche2021/output/')
    parser.add_argument('--inp_file', default = 'topics.xml')
    args = parser.parse_args()
    start = timer()
    run_baseline(output_dir = args.o, input_dir = args.i, input_file = args.inp_file)
    end = timer()
    print ("time", end - start, flush = True)