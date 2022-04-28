# import 




from cProfile import label
from traceback import print_tb


beta=1

beta_square = beta ** 2

only_gross = True
f_type = 'micro'
    



def _bmeso_tag_to_spans(tags, ignore_labels=None):
    r"""
    给定一个tags的lis，比如['O', 'B-singer', 'M-singer', 'E-singer', 'O', 'O']。
    返回[('singer', (1, 4))] (左闭右开区间)

    :param tags: List[str],
    :param ignore_labels: List[str], 在该list中的label将被忽略
    :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
    """
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bmes_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bmes_tag, label = tag[:1], tag[2:]
        if bmes_tag in ('b', 's'):
            spans.append((label, [idx, idx]))
        elif bmes_tag in ('m', 'e') and prev_bmes_tag in ('b', 'm') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bmes_tag == 'o':
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bmes_tag = bmes_tag
    return [(span[0], (span[1][0], span[1][1] + 1))
            for span in spans
            if span[0] not in ignore_labels
            ]

def _bio_tag_to_spans(tags, ignore_labels=None):
    r"""
    给定一个tags的lis，比如['O', 'B-singer', 'I-singer', 'I-singer', 'O', 'O']。
        返回[('singer', (1, 4))] (左闭右开区间)

    :param tags: List[str],
    :param ignore_labels: List[str], 在该list中的label将被忽略
    :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
    """
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bio_tag = None
    for idx, tag in enumerate(tags): 
        tag = tag.lower()
        bio_tag, label = tag[:1], tag[2:]
        if bio_tag == 'b':
            spans.append((label, [idx, idx]))
        elif bio_tag == 'i' and prev_bio_tag in ('b', 'i') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bio_tag == 'o':  # o tag does not count
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bio_tag = bio_tag
    return [(span[0], (span[1][0], span[1][1] + 1)) for span in spans if span[0] not in ignore_labels]



def _compute_f_pre_rec(beta_square, tp, fn, fp):
    r"""

    :param tp: int, true positive true negative 
    :param fn: int, false negative 
    :param fp: int, false positive

    :return: (f, pre, rec)
    """
    pre = tp / (fp + tp + 1e-13)
    rec = tp / (fn + tp + 1e-13)
    f = (1 + beta_square) * pre * rec / (beta_square * pre + rec + 1e-13)

    return f, pre, rec




# eval_file ='/data/zbh/yuanshi_bert/BERT-NER-master_tf/zhNER/0406output_weibo36w/label_test3.txt'
# eval_file = '/data/zbh/yuanshi_bert/BERT-NER-master_tf/zhNER/0406output_resume_36w/label_test.txt'
# eval_file = '/data1/zbh/pre_trainmodel/ner_model/0415with_scope/label_test3.txt'
# eval_file ="/data1/zbh/pre_trainmodel/ner_model/0415rongruMLM/label_test3.txt"
# eval_file ="/data1/zbh/pre_trainmodel/ner_model/0415nonsp_noatt/label_test3.txt"
# eval_file ='/data1/zbh/pre_trainmodel/ner_model/0415yuanshi/label_test3.txt'
# eval_file = '/data1/zbh/pre_trainmodel/ner_model/0415yuanshi/label_dev.txt'
# eval_file = '/data/zbh/yuanshi_bert/BERT-NER-master_tf/fastnlp/result.txt'

eval_file = '/data1/zbh/pre_trainmodel/ner_model/0425yuanshi_ontoNote/label50w.txt'



f1 = open(eval_file,'r',encoding='utf-8')
tp,fn,fp = 0,0,0
x1 = f1.readlines()
label_gold = []
label_pred = []
for line in x1:
    

    if not line:
        print(label_pred,label_gold)


        label_gold = []
        label_pred = []
    else:
        if( line == '\n'):
            continue
        label_pred.append(line.strip('\n').split('\t')[-1])
        label_gold.append(line.strip('\n').split('\t')[1])


# def get_labels():
#     file1 = '/data/zbh/NER_MODEL/NER_data/ner/datas/WeiboNER/weiboNER_2nd_conll.test_deseg'
#     file2 = '/data/zbh/yuanshi_bert/CLUE/output/yuanshi_weibo20/label_test.txt'
#     label_gold = []
#     label_pred = []
#     f1 = open(file1,'r',encoding='utf-8')
#     count = 0
#     space_count = 0
#     list1 = []
#     temp = []
#     list2 = []
#     for i in f1.readlines():
        
#         # if(i == )
#         i = i.strip()
#         if not i:
            
#             list1.append(temp)
#             temp = []
#             space_count += 1
#             continue
#         count +=1 
#         try:
#             label_gold.append(i.strip().split(' ')[1])
#             temp.append(i.strip().split(' ')[1])
#         except Exception as e:
#             print(e)
#             continue
#     f2 = open(file2,'r',encoding='utf-8')
#     # print(len(list(f2.readlines())),space_count,'c',count)
#     for i in f2.readlines():
#         list2.append(i.strip().split(' '))
#         label_pred.extend(i.strip().split(' '))
    
#     print(len(label_gold),len(label_pred),len(list1))
    
#     return label_gold,label_pred,list1,list2


# label_gold,label_pred,list1,list2 = get_labels()

# for i in range(len(list1)):
#     if(len(list1[i])!= len(list2[i])):
#         print(len(list1[i]),len(list2[i]))
#         print(i,list1[i],list2[i])
#         if(len(list1[i])>len(list2[i])):
#             while len(list1[i])!=len(list2[i]):
#                 list2[i].append('O')
#         else:
#             list2[i] = list2[i][0:len(list1[i])]


# def merge1(list3):
#     list4 = []
#     for i in list3:
#         list4.extend(i)
#     return list4

# label_gold1 = merge1(list1)
# label_pred1 = merge1(list2)

# print(len(label_gold1),len(label_pred1))
# label_g1 = _bio_tag_to_spans(label_gold1)
# label_p1 = _bio_tag_to_spans(label_pred1)

# print(len(label_g1),len(label_p1))

_true_positives, _false_positives, _false_negatives = {},{},{}

label_g1 = _bmeso_tag_to_spans(label_gold)
label_p1 = _bmeso_tag_to_spans(label_pred)




for i in label_g1:
    _true_positives[i[0]] = 0
    _false_positives[i[0]] = 0
    _false_negatives[i[0]] = 0


for span in label_p1:
    if span in label_g1:
        _true_positives[span[0]] += 1
        label_g1.remove(span)
    else:
        try:
            _false_positives[span[0]] += 1
        except Exception as e:
            print(e)

for span in label_g1:
    _false_negatives[span[0]] += 1
evaluate_result = {}

tags = set(_false_negatives.keys())
tags.update(set(_false_positives.keys()))
tags.update(set(_true_positives.keys()))
f_sum = 0
pre_sum = 0
rec_sum = 0
for tag in tags:
    tp = _true_positives[tag]
    fn = _false_negatives[tag]
    fp = _false_positives[tag]
    f, pre, rec = _compute_f_pre_rec(beta_square, tp, fn, fp)
    f_sum += f
    pre_sum += pre
    rec_sum += rec
    print(tp,fn,fp)
    if not only_gross and tag != '':  # tag!=''防止无tag的情况
        f_key = 'f-{}'.format(tag)
        pre_key = 'pre-{}'.format(tag)
        rec_key = 'rec-{}'.format(tag)
        evaluate_result[f_key] = f
        evaluate_result[pre_key] = pre
        evaluate_result[rec_key] = rec

if  f_type == 'macro':
    evaluate_result['f'] = f_sum / len(tags)
    evaluate_result['pre'] = pre_sum / len(tags)
    evaluate_result['rec'] = rec_sum / len(tags)


if f_type == 'micro':
            f, pre, rec = _compute_f_pre_rec(beta_square,
                                             sum(_true_positives.values()),
                                             sum(_false_negatives.values()),
                                             sum(_false_positives.values()))
            evaluate_result['f'] = f
            evaluate_result['pre'] = pre
            evaluate_result['rec'] = rec
print(evaluate_result)

print(evaluate_result)

# print(label_p1,label_g1)