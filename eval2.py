# import 




from cProfile import label


beta=1

beta_square = beta ** 2

only_gross = True
f_type = 'macro'
    

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
# eval_file ="/data1/zbh/pre_tra inmodel/ner_model/0415rongruMLM/label_test3.txt"
# eval_file ="/data1/zbh/pre_trainmodel/ner_model/0415nonsp_noatt/label_test3.txt"
# eval_file ='/data1/zbh/pre_trainmodel/ner_model/0415yuanshi/label_test3.txt'
# eval_file = '/data1/zbh/pre_trainmodel/ner_model/0415yuanshi/label_dev.txt'
eval_file = '/data/zbh/yuanshi_bert/BERT-NER-master_tf/fastnlp/result2.txt'
f1 = open(eval_file,'r',encoding='utf-8')
tp,fn,fp = 0,0,0
x1 = f1.readlines()
label_all = []
pred_all = []
label_gold = []
label_pred = []
for line in x1:
    
    if line == '\n':
        # print(label_pred,label_gold)
        label_all.append(label_gold)
        pred_all.append(label_pred)


        label_gold = []
        label_pred = []
    else:

        label_pred.append(line.strip('\n').split('\t')[-1])
        label_gold.append(line.strip('\n').split('\t')[1])



_true_positives, _false_positives, _false_negatives = {},{},{}

# for list1 in label_all:
#     for i in list1:
#         _true_positives[i[0]] = 0
#         _false_positives[i[0]] = 0
#         _false_negatives[i[0]] = 0

for label_gold, label_pred in zip(label_all,pred_all):
    label_g1 = _bio_tag_to_spans(label_gold)
    label_p1 = _bio_tag_to_spans(label_pred)
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
    print(tp,fn,fp,f)
    pre_sum += pre
    rec_sum += rec
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

print(evaluate_result)

# print(label_p1,label_g1)