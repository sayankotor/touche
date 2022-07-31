import pyterrier as pt
if not pt.started():
    pt.init('5.4')
#import onir_pt

from xml.dom import minidom
import re
import pickle

import pandas as pd

import numpy as np
import argparse
        
with open('list_of_un_answ.pcl', 'rb') as f:
    answers_2020 = pickle.load(f)
            
with open('list_of_un_answ_2021.pcl', 'rb') as f:
    answers_2021 = pickle.load(f)
    
    
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

from bert_help import OpenNIRPyterrierReRanker

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
    print (0)
    # -1 assures that first rank will be FIRST_RANK
    rtr["rank"] = rtr.groupby("qid", sort=False)["score"].rank(ascending=False, method="first").astype(int) -1 + 1
    print (1)
    if True:
        rtr.sort_values(["qid", "rank"], ascending=[True,True], inplace=True)
    return rtr

def write_qrels(output_dir, name, rtr):
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
    


def run_baseline(output_dir, input_dir, input_file):        
    info_df_test = pd.DataFrame(columns=["qid", "query", "docno", "text"], dtype=object)
    
    topics_2020 = read_xml('topics-task-2.xml')
        
    topics_2021 = read_xml('topics-task-2-only-titles-2021.xml')
        
    with open('list_of_un_answ.pcl', 'rb') as f:
        answers_2020 = pickle.load(f)
            
    with open('list_of_un_answ_2021.pcl', 'rb') as f:
        answers_2021 = pickle.load(f)

    for elem in topics_2021:
        qid, query = elem[0], elem[1].strip('\n')
        query = re.sub(r'[^\w\s]','',query)
        for answer in answers_2021[qid]:
            docno = answer[1]
            score = answer[0]
            text = answer[3]
            df_row = {"qid":qid, "query":query, "docno":docno, "text":text}
            info_df_test = info_df_test.append(df_row, ignore_index= True)
            
    print ("info df test head ", info_df_test.head())
    
    

    vbert_antique = OpenNIRPyterrierReRanker('vanilla_transformer', 'bert', text_field='text', vocab_config={'train': True})
    vbert_antique.from_checkpoint("bert_pre_trained_on_antique")
    answs = vbert_antique.transform(info_df_test)
    answs = add_ranks1(answs)
    write_qrels(output_dir, "bert_test", answs)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('--i', default='/notebook/touche2021/runs/')
    parser.add_argument('--o', default='/notebook/touche2021/output/')
    parser.add_argument('--inp_file', default = 'topics.xml')
    args = parser.parse_args()
    run_baseline(output_dir = args.o, input_dir = args.i, input_file = args.inp_file)
    