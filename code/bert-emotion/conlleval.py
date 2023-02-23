"""
参考:
https://www.clips.uantwerpen.be/conll2000/chunking/output.html
https://github.com/sighsmile/conlleval

This script applies to IOB2 or IOBES tagging scheme.
If you are using a different scheme, please convert to IOB2 or IOBES.
IOB2:
- B = begin, 
- I = inside but not the first, 
- O = outside
e.g. 
John   lives in New   York  City  .
B-PER  O     O  B-LOC I-LOC I-LOC O
IOBES:
- B = begin, 
- E = end, 
- S = singleton, 
- I = inside but not the first or the last, 
- O = outside
e.g.
John   lives in New   York  City  .
S-PER  O     O  B-LOC I-LOC E-LOC O
prefix: IOBES
chunk_type: PER, LOC, etc.
"""
from __future__ import division, print_function, unicode_literals
from pickletools import read_long1

import sys
from collections import defaultdict

from numpy import arange
import torch
from tqdm import tqdm
import json



def split_tag(chunk_tag):
    """
    split chunk tag into IOBES prefix and chunk_type
    e.g. 
    B-PER -> (B, PER)
    O -> (O, None)
    """
    if chunk_tag == 'O':
        return ('O', None)
    return chunk_tag.split('-')

def is_chunk_end(prev_tag, tag):
    """
    check if the previous chunk ended between the previous and current word
    e.g. 
    (B-PER, I-PER) -> False
    (B-LOC, O)  -> True
    Note: in case of contradicting tags, e.g. (B-PER, I-LOC)
    this is considered as (B-PER, B-LOC)
    """
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix1 == 'O':
        return False
    if prefix2 == 'O':
        return prefix1 != 'O'

    if chunk_type1 != chunk_type2:
        return True

    return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']

def is_chunk_start(prev_tag, tag):
    """
    check if a new chunk started between the previous and current word
    """
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix2 == 'O':
        return False
    if prefix1 == 'O':
        return prefix2 != 'O'

    if chunk_type1 != chunk_type2:
        return True

    return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']


def calc_metrics(tp, p, t, percent=True):
    """
    compute overall precision, recall and FB1 (default values are 0.0)
    if percent is True, return 100 * original decimal value
    """
    precision = tp / p if p else 0
    recall = tp / t if t else 0
    fb1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    if percent:
        return 100 * precision, 100 * recall, 100 * fb1
    else:
        return precision, recall, fb1


def count_flags(true_seqs, pred_seqs):
    """
    true_seqs: a list of true tags
    pred_seqs: a list of predicted tags
    return: 
    correct_counts: a dict (counter), 
                    key = flag num, 
                    value = number of correctly identified flag per type
    true_counts:    a dict, number of true flags per type
    pred_counts:    a dict, number of identified flags per type
    """

    correct_counts = torch.zeros(38)
    true_counts = torch.zeros(38)
    pred_counts = torch.zeros(38)


    for true_tag, pred_tag in zip(true_seqs, pred_seqs):
        if true_tag == pred_tag:#预测正 实际正 TP
            correct_counts[true_tag] += 1
        true_counts[true_tag] += 1#(TP+FN)所有实际正的
        pred_counts[pred_tag] += 1#(TP+FP)所有预测正的

    return (correct_counts, true_counts, pred_counts)

def get_result(correct_counts, true_counts, pred_counts, verbose=True):
    """
    if verbose, print overall performance, as well as preformance per chunk type;
    otherwise, simply return overall prec, rec, f1 scores
    """
    corr_list =torch.zeros(38)
    accu_list = torch.zeros(38)
    prec_list = torch.zeros(38)
    rec_list =torch.zeros(38)
    f1_list = torch.zeros(38)
    count_list = torch.zeros(38)

    iter_count = 0
    for iter in range(len(correct_counts)):
        prec,rec,f1 = calc_metrics(correct_counts[iter],pred_counts[iter],true_counts[iter])
        corr_list[iter] = correct_counts[iter]
        if true_counts[iter] == 0:
            accu_list[iter] = -1
        else:
            accu_list[iter] = 100*correct_counts[iter]/true_counts[iter]
        prec_list[iter] = prec
        rec_list[iter] = rec
        f1_list[iter] = f1
        iter_count+=1

    # sum counts
    sum_correct_counts = correct_counts.sum()
    sum_pred_counts = pred_counts.sum()
    sum_true_counts = true_counts.sum()

    

    # compute overall precision, recall and FB1 (default values are 0.0)
    prec, rec, f1 = calc_metrics(sum_correct_counts, sum_pred_counts, sum_true_counts)
    iter_count -=1
    corr_list[iter_count] = sum_correct_counts
    accu_list[iter_count] = 100*sum_correct_counts/sum_true_counts
    prec_list[iter_count] = prec
    rec_list[iter_count] = rec
    f1_list[iter_count] = f1

    res = (corr_list,accu_list,prec_list, rec_list, f1_list,iter_count,true_counts)
    if not verbose:
        return res

    # print overall performance, and performance per chunk type

    print("found: %i phrases; correct: %i.\n accuracy: %6.2f%%; precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" 
    % (sum_pred_counts, sum_correct_counts,100*sum_correct_counts/sum_true_counts,prec, rec, f1))
    # print("accuracy: %6.2f%%; " % (100*sum_correct_counts/sum_true_counts))
    # print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" % (prec, rec, f1))
    # print(len(true_counts))

    # for each chunk type, compute precision, recall and FB1 (default values are 0.0)
    # for t in arange(len(true_counts)):
    #     prec, rec, f1 = calc_metrics(correct_counts[t], pred_counts[t], true_counts[t])
    #     print("%17s: " %t , end='')
    #     print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" %
    #                 (prec, rec, f1), end='')

    return res
    # you can generate LaTeX output for tables like in
    # http://cnts.uia.ac.be/conll2003/ner/example.tex
    # but I'm not implementing this


# def count_chunks(true_seqs, pred_seqs):

#     # true_seqs: a list of true tags
#     # pred_seqs: a list of predicted tags
#     # return: 
#     # correct_chunks: a dict (counter), 
#     #                 key = chunk types, 
#     #                 value = number of correctly identified chunks per type
#     # true_chunks:    a dict, number of true chunks per type
#     # pred_chunks:    a dict, number of identified chunks per type
#     # correct_counts, true_counts, pred_counts: similar to above, but for tags

#     correct_chunks = defaultdict(int)
#     true_chunks = defaultdict(int)
#     pred_chunks = defaultdict(int)

#     correct_counts = defaultdict(int)
#     true_counts = defaultdict(int)
#     pred_counts = defaultdict(int)

#     prev_true_tag, prev_pred_tag = 'O', 'O'
#     correct_chunk = None

#     for true_tag, pred_tag in zip(true_seqs, pred_seqs):
#         if true_tag == pred_tag:
#             correct_counts[true_tag] += 1
#         true_counts[true_tag] += 1
#         pred_counts[pred_tag] += 1

#         _, true_type = split_tag(true_tag)
#         _, pred_type = split_tag(pred_tag)

#         if correct_chunk is not None:
#             true_end = is_chunk_end(prev_true_tag, true_tag)
#             pred_end = is_chunk_end(prev_pred_tag, pred_tag)

#             if pred_end and true_end:
#                 correct_chunks[correct_chunk] += 1
#                 correct_chunk = None
#             elif pred_end != true_end or true_type != pred_type:
#                 correct_chunk = None

#         true_start = is_chunk_start(prev_true_tag, true_tag)
#         pred_start = is_chunk_start(prev_pred_tag, pred_tag)

#         if true_start and pred_start and true_type == pred_type:
#             correct_chunk = true_type
#         if true_start:
#             true_chunks[true_type] += 1
#         if pred_start:
#             pred_chunks[pred_type] += 1

#         prev_true_tag, prev_pred_tag = true_tag, pred_tag
#     if correct_chunk is not None:
#         correct_chunks[correct_chunk] += 1

#     return (correct_chunks, true_chunks, pred_chunks, 
#         correct_counts, true_counts, pred_counts)
# def get_result(correct_chunks, true_chunks, pred_chunks,
#     correct_counts, true_counts, pred_counts, verbose=True):
    
#     # if verbose, print overall performance, as well as preformance per chunk type;
#     # otherwise, simply return overall prec, rec, f1 scores
    
#     # sum counts
#     sum_correct_chunks = sum(correct_chunks.values())
#     sum_true_chunks = sum(true_chunks.values())
#     sum_pred_chunks = sum(pred_chunks.values())

#     sum_correct_counts = sum(correct_counts.values())
#     sum_true_counts = sum(true_counts.values())

#     nonO_correct_counts = sum(v for k, v in correct_counts.items() if k != 'O')
#     nonO_true_counts = sum(v for k, v in true_counts.items() if k != 'O')

#     chunk_types = sorted(list(set(list(true_chunks) + list(pred_chunks))))

#     # compute overall precision, recall and FB1 (default values are 0.0)
#     prec, rec, f1 = calc_metrics(sum_correct_chunks, sum_pred_chunks, sum_true_chunks)
#     res = (prec, rec, f1)
#     if not verbose:
#         return res

#     # print overall performance, and performance per chunk type
    
#     print("processed %i tokens with %i phrases; " % (sum_true_counts, sum_true_chunks), end='')
#     print("found: %i phrases; correct: %i.\n" % (sum_pred_chunks, sum_correct_chunks), end='')
        
#     print("accuracy: %6.2f%%; (non-O)" % (100*nonO_correct_counts/nonO_true_counts))
#     print("accuracy: %6.2f%%; " % (100*sum_correct_counts/sum_true_counts), end='')
#     print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" % (prec, rec, f1),end="")
#     print("  (%d & %d) = %d" % (sum_pred_chunks,sum_true_chunks,sum_correct_chunks))


#     # for each chunk type, compute precision, recall and FB1 (default values are 0.0)
#     for t in chunk_types:
#         prec, rec, f1 = calc_metrics(correct_chunks[t], pred_chunks[t], true_chunks[t])
#         print("%17s: " %t , end='')
#         print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" %
#                     (prec, rec, f1), end='')
#         print("  (%d & %d) = %d" % (pred_chunks[t],true_chunks[t],correct_chunks[t]))

#     return prec, rec, f1
#     # you can generate LaTeX output for tables like in
#     # http://cnts.uia.ac.be/conll2003/ner/example.tex
#     # but I'm not implementing this

def evaluate(true, pred, verbose=True):
    # 计算情感分类的precision,recall和f1
    #其中f1分别计算Macro和Micro

    (correct_counts, true_counts, pred_counts) = count_flags(true, pred)
    result = get_result(correct_counts, true_counts, pred_counts, verbose=verbose)
    return result

# def after_process(input):
#     span1 = input.split("[")
#     span2 = span

#     output = [head,reln,tail]
#     return output



def PE_count_chunks(true_seqs, pred_seqs):
    """
    true_seqs: a list of true tags
    pred_seqs: a list of predicted tags
    return: 
    correct_chunks: a dict (counter), 
                    key = chunk types, 
                    value = number of correctly identified chunks per type 
    true_chunks:    a dict, number of true chunks per type / 每种类型的真实的chunk的数量
    pred_chunks:    a dict, number of identified chunks per type / 每种类型识别到的（预测的）chunk的数量
    correct_counts, true_counts, pred_counts: similar to above, but for tags
    """
    correct_counts = 0
    true_counts = 0
    pred_counts = 0

    for true_tag, pred_tag in zip(true_seqs, pred_seqs):
        if true_tag == pred_tag and true_tag != "" and pred_tag != "":
            correct_counts += 1
        if true_tag != "":
            true_counts += 1
        if pred_tag != "":
            pred_counts += 1

    return (correct_counts, true_counts, pred_counts)

def PE_count_chunks_modify(true_seqs, pred_seqs):
    """
    true_seqs: a list of true tags
    pred_seqs: a list of predicted tags
    return: 
    correct_chunks: a dict (counter), 
                    key = chunk types, 
                    value = number of correctly identified chunks per type 
    true_chunks:    a dict, number of true chunks per type / 每种类型的真实的chunk的数量
    pred_chunks:    a dict, number of identified chunks per type / 每种类型识别到的（预测的）chunk的数量
    correct_counts, true_counts, pred_counts: similar to above, but for tags
    """
    correct_counts = defaultdict(int)
    true_counts = defaultdict(int)
    pred_counts = defaultdict(int)

    correct_counts = {0:0,1:0}
    true_counts = {0:0,1:0}
    pred_counts = {0:0,1:0}
    for true_tag, pred_tag in tqdm(zip(true_seqs, pred_seqs)):
        
        if true_tag == []:
            true_counts[0] += 1
        else:
            true_counts[1] += len(true_tag)
        if pred_tag == []:
            pred_counts[0] += 1
        else:
            pred_counts[1] += len(pred_tag)
        
        if true_tag == [] and pred_tag == []:
            correct_counts[0] += 1
        elif true_tag != [] and pred_tag != []:
            for true_triple in true_tag:
                for pred_triple in pred_tag:
                    if true_triple[0] == pred_triple[0] and true_triple[1] == pred_triple[1] and true_triple[2] == pred_triple[2]:
                        correct_counts[1] += 1

    return (correct_counts, true_counts, pred_counts)

def PE_evaluate(true_seqs, pred_seqs, verbose=True):
    (correct_counts, true_counts, pred_counts) = PE_count_chunks(true_seqs, pred_seqs)
    prec, rec, f1 = calc_metrics(correct_counts, pred_counts, true_counts)
    print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" %
                    (prec, rec, f1), end='')
    
    return prec, rec, f1

def PE_get_result_modify(correct_counts, true_counts, pred_counts, verbose=True):
    """
    if verbose, print overall performance, as well as preformance per chunk type;
    otherwise, simply return overall prec, rec, f1 scores
    """
    # sum counts
    sum_correct_counts = sum(correct_counts.values())
    sum_true_counts = sum(true_counts.values())
    sum_pred_counts = sum(pred_counts.values())

    nonO_correct_counts = sum(v for k, v in correct_counts.items() if k != 'O')
    nonO_pred_counts = sum(v for k, v in pred_counts.items() if k != 'O')
    nonO_true_counts = sum(v for k, v in true_counts.items() if k != 'O')

    chunk_types = sorted(list(set(list(true_counts) + list(pred_counts))))

    # compute overall precision, recall and FB1 (default values are 0.0)
    prec, rec, f1 = calc_metrics(sum_correct_counts, sum_true_counts, sum_pred_counts)
    # prec, rec, f1 = calc_metrics(sum_correct_counts, sum_pred_counts, sum_true_counts)
    res = (prec, rec, f1)
    if not verbose:
        return res
    # print overall performance, and performance per chunk type

    # print("processed %i tokens with %i phrases; " % (sum_true_counts, sum_true_chunks), end='')
    print("processed %i relations; " % (sum_true_counts), end='')
    print("found: %i relations; correct: %i.\n" % (sum_pred_counts, sum_correct_counts), end='')
    # print("accuracy: %6.2f%%; (non-O)" % (100*nonO_correct_counts/nonO_true_counts))
    print("accuracy: %6.2f%%; " % (100*sum_correct_counts/sum_true_counts), end='')
    print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" % (prec, rec, f1),end="")
    print("  (%d & %d) = %d" % (sum_pred_counts,sum_true_counts,sum_correct_counts))

    f1_list = []
    # for each chunk type, compute precision, recall and FB1 (default values are 0.0)
    for t in chunk_types:
        prec, rec, f1 = calc_metrics(correct_counts[t], pred_counts[t], true_counts[t])
        f1_list.append(f1)
        print("%17s: " %t , end='')
        print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" %
                    (prec, rec, f1), end='')
        print("  (%d & %d) = %d" % (pred_counts[t],true_counts[t],correct_counts[t]))

    return res, f1_list

def convert_seqs_to_triples(seqs):
    backup_personas = []
    for predict in seqs:
        backup_triples = []
        if "<triplet>" in predict and " <subj> " in predict and " <obj> " in predict:
            persons = predict.split("<triplet> ")
            for person_triple in persons[1:]:
                if " <subj> " not in person_triple or " <obj> " not in person_triple:
                    continue
                temp = person_triple.split(" <subj> ")
                head = temp[0]
                for reltail in temp[1:]:
                    if " <obj> " in reltail:
                        rel= reltail.split(" <obj> ")[0]
                        tail= reltail.split(" <obj> ")[1]
                        backup_triples.append([head, rel, tail])
        if len(backup_triples) == 0:
            backup_personas.append([])
        else:
            backup_personas.append(backup_triples)
    return backup_personas

def convert_text_to_triples(seqs):
    with open("/home/hutu/PersonaDataset/PE/data/data_analyse/schema.json", 'r') as f:
        schema = json.load(f)
    
    schema = get_change_schema(schema)
    
    relations = [type['type'].replace("_",' ') for h_type in schema['relation_types'] for type in h_type['types']]
    #relation 长度从长到短排序
    relations = sorted(relations, key=len, reverse=True)
    
    triples = []
    for predict in seqs:
        predict_list = [predict]
        predict = predict_list
        rel_store = [0]
        for rel in relations:
            count = 1
            tmp = []
            for idx, span in enumerate(predict):
                span = span.split(rel)  # ['i', 'dog has age 13', 'apple has age 14']
        #         print(span, idx, rel_store, predict)
                for _ in range(len(span) - 1):
                    rel_store.insert(idx + count, rel)
                    count += 1
                for s in span:
                    s = s.strip()
                    tmp.append(s)
            predict = tmp
        triple = []
        for i in range(len(rel_store)):
            if i >= 1:
                triple.append(['i', rel_store[i], predict[i]])
        triples.append(triple)
    return triples

def PE_evaluate_modify_non_triple(true_seqs, pred_seqs, verbose=True):
    true_personas = convert_text_to_triples(true_seqs)
    pred_personas = convert_text_to_triples(pred_seqs)
    (correct_counts, true_counts, pred_counts) = PE_count_chunks_modify(true_personas, pred_personas)
    (prec, rec, f1),[neg_f1, pos_f1] = PE_get_result_modify(correct_counts, true_counts, pred_counts, verbose=verbose)
    return prec, rec, f1, neg_f1, pos_f1


def PE_evaluate_modify(true_seqs, pred_seqs, verbose=True):
    true_personas = convert_seqs_to_triples(true_seqs)
    pred_personas = convert_seqs_to_triples(pred_seqs)
    (correct_counts, true_counts, pred_counts) = PE_count_chunks_modify(true_personas, pred_personas)
    (prec, rec, f1),[neg_f1, pos_f1] = PE_get_result_modify(correct_counts, true_counts, pred_counts, verbose=verbose)
    return prec, rec, f1, neg_f1, pos_f1
# def PE_evaluate(preds, golds):

#     count = 0
#     for i in tqdm(range(len(preds))):
#         # if preds[i*4] == golds[i] or preds[i*4 + 1] == golds[i] or preds[i*4 + 2] == golds[i] or preds[i*4 + 3] == golds[i]:
#         # if preds[i] == "":
#         #     if preds[i] == gold[i]:
#         #         count += 1
#         # else:
#         #     pred = after_process(preds[i])
#         #     gold = after_process(golds[i])
#         #     if gold[0] == pred[0] and gold[1] == pred[1] and gold[2] == pred[2]:
#         #         count += 1
#         if preds[i] == golds[i]:    
#             count += 1
#     return count / len(preds)
# def evaluate(true, pred):
#     # 计算情感分类的precision,recall和f1
#     #其中f1分别计算Macro和Micro

#     (correct_counts, true_counts, pred_counts) = count_flags(true, pred)
#     result = get_result(correct_counts, true_counts, pred_counts, verbose=True)
#     return result


def RE_get_result(correct_counts, true_counts, pred_counts, verbose=True):
    """
    if verbose, print overall performance, as well as preformance per chunk type;
    otherwise, simply return overall prec, rec, f1 scores
    """
    # sum counts
    sum_correct_counts = sum(correct_counts.values())
    sum_true_counts = sum(true_counts.values())
    sum_pred_counts = sum(pred_counts.values())

    nonO_correct_counts = sum(v for k, v in correct_counts.items() if k != 'O')
    nonO_pred_counts = sum(v for k, v in pred_counts.items() if k != 'O')
    nonO_true_counts = sum(v for k, v in true_counts.items() if k != 'O')

    chunk_types = sorted(list(set(list(true_counts) + list(pred_counts))))

    # compute overall precision, recall and FB1 (default values are 0.0)
    prec, rec, f1 = calc_metrics(nonO_correct_counts, nonO_pred_counts, nonO_true_counts)
    # prec, rec, f1 = calc_metrics(sum_correct_counts, sum_pred_counts, sum_true_counts)
    res = (prec, rec, f1)
    if not verbose:
        return res
    
    # for each chunk type, compute precision, recall and FB1 (default values are 0.0)
    for t in chunk_types:
        prec, rec, f1 = calc_metrics(correct_counts[t], pred_counts[t], true_counts[t])
        print("%17s: " %t , end='')
        print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" %
                    (prec, rec, f1), end='')
        print("  (%d & %d) = %d" % (pred_counts[t],true_counts[t],correct_counts[t]))
    
    # print overall performance, and performance per chunk type
    # print("processed %i tokens with %i phrases; " % (sum_true_counts, sum_true_chunks), end='')
    print("processed %i relations; " % (sum_true_counts), end='')
    print("found: %i relations; correct: %i.\n" % (sum_pred_counts, sum_correct_counts), end='')
    print("accuracy: %6.2f%%; (non-O)" % (100*nonO_correct_counts/nonO_true_counts))
    print("accuracy: %6.2f%%; " % (100*sum_correct_counts/sum_true_counts), end='')
    print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" % (prec, rec, f1),end="")
    print("  (%d & %d) = %d" % (pred_counts,sum_true_counts,sum_correct_counts))


    

    return res
    # you can generate LaTeX output for tables like in
    # http://cnts.uia.ac.be/conll2003/ner/example.tex
    # but I'm not implementing this

def RE_count_chunks(true_seqs, pred_seqs):
    """
    true_seqs: a list of true tags
    pred_seqs: a list of predicted tags
    return: 
    correct_chunks: a dict (counter), 
                    key = chunk types, 
                    value = number of correctly identified chunks per type 
    true_chunks:    a dict, number of true chunks per type / 每种类型的真实的chunk的数量
    pred_chunks:    a dict, number of identified chunks per type / 每种类型识别到的（预测的）chunk的数量
    correct_counts, true_counts, pred_counts: similar to above, but for tags
    """
    correct_counts = defaultdict(int)
    true_counts = defaultdict(int)
    pred_counts = defaultdict(int)

    for true_tag, pred_tag in zip(true_seqs, pred_seqs):
        if true_tag == pred_tag:
            correct_counts[true_tag] += 1
        true_counts[true_tag] += 1
        pred_counts[pred_tag] += 1

    return (correct_counts, true_counts, pred_counts)


def RE_evaluate(true_seqs, pred_seqs, verbose=True):
    (correct_counts, true_counts, pred_counts) = RE_count_chunks(true_seqs, pred_seqs)
    result = RE_get_result(correct_counts, true_counts, pred_counts, verbose=verbose)
    return result


def PD_acc_evaluate(preds,golds):
    count = [0,0,0,0,0]
    for pred,gold in zip(preds,golds):
        for i in range(len(pred)):
            if pred[i] == gold[i]:
                count[i] += 1
    for i in range(len(count)):
        count[i] = count[i] / len(preds)
    return count


def evaluate_conll_file(fileIterator):
    """
    NER评估函数
    :param fileIterator: NER得到的txt文件
    :return: 如下评估指标信息，着重关注最后一行的准确率
    eg:
        processed 4503502 tokens with 93009 phrases; found: 92829 phrases; correct: 89427.
        accuracy:  97.43%; (non-O)
        accuracy:  99.58%; precision:  96.34%; recall:  96.15%; FB1:  96.24
                    COM: precision:  96.34%; recall:  96.15%; FB1:  96.24  92829

        分别表示：
        txt文件一共包含 4503502 个字符， 其中共 93009 个实体（gold）， 模型预测实体共有 92829 个， 其中正确的有 89427.
        只看实体名（non-O）的字符级准确率 97.43%（字符级）
        所有的字符级准确率 99.58%（字符级）     后面三个 p/r/f 和下一行相同。
                    实体为COM的短语级别 precision/recall/FB1 分别为96.34%; 96.15%; 96.24 (这三个都是短语级，当整个实体的BI全部预测正确才算正确)
    """
    true_seqs, pred_seqs = [], []
    for line in fileIterator:
        cols = line.strip().split()
        # each non-empty line must contain >= 3 columns
        if not cols:
            true_seqs.append('O')
            pred_seqs.append('O')
        elif len(cols) < 3:
            raise IOError("conlleval: too few columns in line %s\n" % line)
        else:
            # extract tags from last 2 columns
            true_seqs.append(cols[-2])
            pred_seqs.append(cols[-1])
    return evaluate(true_seqs, pred_seqs)

if __name__ == '__main__':
    """
    usage:     conlleval < file
    """
    evaluate_conll_file(sys.stdin)