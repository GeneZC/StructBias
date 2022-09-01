# -*- coding: utf-8 -*-


def exact_match_f1(outputs, targets):
    TP, FP, FN = 0, 0, 0
    num_instances = len(targets)
    assert num_instances == len(outputs)

    for i in range(num_instances):
        num_hits = len(set(outputs[i]) & set(targets[i]))
        num_output_triplets = len(outputs[i])
        num_target_triplets = len(targets[i])
        TP += num_hits
        FP += (num_output_triplets - num_hits)
        FN += (num_target_triplets - num_hits)
    precision = float(TP) / float(TP + FP + 1e-5)
    recall = float(TP) / float(TP + FN + 1e-5)

    f1 = 2 * precision * recall / (precision + recall + 1e-5)

    return precision, recall, f1

def overlap(x_beg, x_end, y_beg, y_end):
    if x_beg >= y_beg:
        return (max(y_end - x_beg + 1, 0) - max(y_end - x_end, 0)) / max(x_end - x_beg + 1, y_end - y_beg + 1)
    else:
        return (max(x_end - y_beg + 1, 0) - max(x_end - y_end, 0)) / max(x_end - x_beg + 1, y_end - y_beg + 1)

def fuzzy_match_f1(outputs, targets):
    TP, FP, FN = 0, 0, 0
    num_instances = len(targets)
    assert num_instances == len(outputs)

    for i in range(num_instances):
        num_hits = 0
        for output in outputs[i]:
            output_t_beg, output_t_end, output_o_beg, output_o_end, output_s = [int(x) for x in output.split('-')]
            for target in targets[i]:
                target_t_beg, target_t_end, target_o_beg, target_o_end, target_s = [int(x) for x in target.split('-')]
                num_hits += overlap(output_t_beg, output_t_end, target_t_beg, target_t_end) * overlap(output_o_beg, output_o_end, target_o_beg, target_o_end) * float(output_s == target_s)
        num_output_triplets = len(outputs[i])
        num_target_triplets = len(targets[i])
        TP += num_hits
        FP += (num_output_triplets - num_hits)
        FN += (num_target_triplets - num_hits)
    precision = float(TP) / float(TP + FP + 1e-5)
    recall = float(TP) / float(TP + FN + 1e-5)

    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    
    return precision, recall, f1
