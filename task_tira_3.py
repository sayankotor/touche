import requests
import sys
import argparse

#numpy
import numpy as np

#torch
import torch, torch.nn as nn
import torch.nn.functional as F

from xml.dom import minidom

from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity
bc = BertClient()

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
            "size": 1000,
            "pretty": True
        }
    response = requests.get(url = "http://www.chatnoir.eu/api/v1/_search", params = params)
    return response

def clean_punct(s):
    s = re.sub(r'[^\w\s]','',s)
    return s

from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity
bc = BertClient()

import re

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext.lower()

def make_scores_lstm(query, answers, emb_size = 768, hidden_size = 64):
    query_embs = bc.encode((clean_punct(cleanhtml(query))).split())
    tensor_query_embs = torch.tensor(query_embs)
    
    my_rnn = nn.RNN(emb_size, hidden_size, batch_first=True) 
    all_query_hidden, last_query_hidden = my_rnn(tensor_query_embs.unsqueeze(0))
    
    #last_answers_hidden = [my_rnn(torch.tensor(bc.encode((clean_punct(cleanhtml(answer))).split())).unsqueeze(0))[1] for answer in answers]
    last_answers_hidden = [my_rnn(torch.tensor(np.copy(bc.encode((clean_punct(cleanhtml(answer))).split()))).unsqueeze(0))[1] for answer in answers]
    
    
    aa = last_query_hidden.detach().numpy()
    print ("aa shape", aa.shape)
    bb = last_answers_hidden[0].detach().numpy()
    print ("bb shape", bb.shape)
    
    scores = [cosine_similarity(last_query_hidden.squeeze(0).detach().numpy(), last_answer_hidden.squeeze(0).detach().numpy())[0][0] for last_answer_hidden in last_answers_hidden]
    return scores
    
    
    # take title from seacrh output from clueweb
    # return a cosine similarity between bert embedding of sentences and article title
    
def run_baseline(output_dir = '/notebook/touche/output/', input_dir = '/notebook/touche/', input_file = 'topics-task-2-only-titles.xml'):
    list_of_tuples = read_xml(input_dir + input_file)
    common_list = []
        
    print ("lstm")    
    for elem in list_of_tuples:
        qid = elem[0]
        Q0 = 'Q0'
        query = elem[1]
        tag = 'Method3'
        response = make_a_search_request(query)
        try:
            getted_request = response.json()
        except:
            print ("exept0")
            return getted_request
        
        n_request = 100
        while (n_request > 0 and 'results' not in getted_request):
            response = make_a_search_request(query)
            getted_request = response.json()
            n_request == n_request - 1
            print ("rerequest", query)
            
        try:
            scores0 = [elem['score'] for elem in getted_request['results']]
            print ("0")
            docs = [elem['trec_id'] for elem in getted_request['results']]
            print ("1")
            titles = [elem['title'] for elem in getted_request['results']]
            print ("2")
            answers_bodies = [cleanhtml(elem['snippet']) for elem in getted_request['results']]
            print ("3")
            # print (scores0, scores3, scores)
        except:
            print ("exept1")
            return getted_request
        scores = make_scores_lstm(query, answers_bodies)
        qids = [qid]*len(scores)
        Q0s = [Q0 for elem in scores]
        queries = query*len(scores)
        tags = [tag for elem in scores]
        part_of_commom_list = list(zip(qids, Q0s, docs, scores, tags))
        part_of_commom_list = sorted(part_of_commom_list, key = lambda x: x[3], reverse = True) 
        
        qids, Q0s, docs, scores, tags = zip(*part_of_commom_list)
        
        ranks = range(1, len(scores) + 1)
        part_of_commom_list = list(zip(qids, Q0s, docs, ranks, scores, tags))
        common_list = common_list + part_of_commom_list

    with open(output_dir + 'run.txt', 'w') as fp:
        fp.write('\n'.join('%s %s %s %s %s %s' % x for x in common_list))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('--i', default='/notebook/touche/')
    parser.add_argument('--o', default='/notebook/touche/output/')
    parser.add_argument('--inp_file', default = 'topics.xml')
    args = parser.parse_args()
    run_baseline(output_dir = args.o, input_dir = args.i, input_file = args.inp_file)