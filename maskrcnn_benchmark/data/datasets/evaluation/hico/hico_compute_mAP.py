import os
import argparse
import numpy as np
import pickle
from multiprocessing import Pool
from maskrcnn_benchmark.utils.miscellaneous import mkdir, dump_json_object, load_json_object


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'Data'))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--pred_hoi_dets',
    type=str,
    default=None,
    required=True,
    help='Path to predicted hoi detections pkl file')
parser.add_argument(
    '--out_dir',
    type=str,
    default=None,
    required=True,
    help='Output directory')
parser.add_argument(
    '--subset',
    type=str,
    default='test',
    choices=['train', 'test', 'val', 'train_val'],
    help='Subset of data to run the evaluation on')
parser.add_argument(
    '--num_processes',
    type=int,
    default=12,
    help='Number of processes to parallelize across')

def compute_area(bbox,invalid=None):
    x1,y1,x2,y2 = bbox

    if (x2 <= x1) or (y2 <= y1):
        area = invalid
    else:
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

    return area

def compute_iou(bbox1, bbox2, verbose=False):
    x1, y1, x2, y2 = bbox1
    x1_, y1_, x2_, y2_ = bbox2

    x1_in = max(x1, x1_)
    y1_in = max(y1, y1_)
    x2_in = min(x2, x2_)
    y2_in = min(y2, y2_)

    intersection = compute_area(bbox=[x1_in, y1_in, x2_in, y2_in], invalid=0.0)
    area1 = compute_area(bbox1)
    area2 = compute_area(bbox2)
    union = area1 + area2 - intersection
    iou = intersection / (union + 1e-6)

    if verbose:
        return iou, intersection, union

    return iou

def match_hoi(pred_det, gt_dets):
    is_match = False
    remaining_gt_dets = [gt_det for gt_det in gt_dets]
    for i, gt_det in enumerate(gt_dets):
        human_iou = compute_iou(pred_det['human_box'], gt_det['human_box'])
        if human_iou > 0.5:
            object_iou = compute_iou(pred_det['object_box'], gt_det['object_box'])
            if object_iou > 0.5:
                is_match = True
                del remaining_gt_dets[i]
                break
        # remaining_gt_dets.append(gt_det)

    return is_match, remaining_gt_dets


def compute_ap(precision, recall):
    if np.any(np.isnan(recall)):
        return np.nan

    ap = 0
    for t in np.arange(0, 1.1, 0.1):  # 0, 0.1, 0.2, ..., 1.0
        selected_p = precision[recall >= t]
        if selected_p.size == 0:
            p = 0
        else:
            p = np.max(selected_p)
        ap += p / 11.

    return ap


def compute_pr(y_true, y_score, npos):
    sorted_y_true = [y for y, _ in
                     sorted(zip(y_true, y_score), key=lambda x: x[1], reverse=True)]
    tp = np.array(sorted_y_true)
    fp = ~tp
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    if npos == 0:
        recall = np.nan * tp
    else:
        recall = tp / npos
    precision = tp / (tp + fp)
    return precision, recall


def compute_normalized_pr(y_true, y_score, npos, N=196.45):
    sorted_y_true = [y for y, _ in
                     sorted(zip(y_true, y_score), key=lambda x: x[1], reverse=True)]
    tp = np.array(sorted_y_true)
    fp = ~tp
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    if npos == 0:
        recall = np.nan * tp
    else:
        recall = tp / npos
    precision = recall * N / (recall * N + fp)
    nap = np.sum(precision[sorted_y_true]) / (npos + 1e-6)
    return precision, recall, nap

def compute_mAP(APs,hoi_ids):
    return sum([APs[hoi_id] for hoi_id in hoi_ids]) / len(hoi_ids)

def eval_hoi(hoi_id, global_ids, gt_dets, pred_dets, out_dir, hoi_map_to_obj_vb):
    print(f'Evaluating hoi_id: {hoi_id} ...')
    # pred_dets = h5py.File(pred_dets_hdf5, 'r')
    pred_dets = pickle.load( open( pred_dets, "rb" ) )
    y_true = []
    y_score = []
    det_id = []
    npos = 0
    for global_id in global_ids:
        if hoi_id in gt_dets[global_id]:
            candidate_gt_dets = gt_dets[global_id][hoi_id]
        else:
            candidate_gt_dets = []
        npos += len(candidate_gt_dets)

        image_id = int(global_id.split('_')[2])
        det_this_image = pred_dets[image_id]
        obj_id = hoi_map_to_obj_vb[hoi_id]['object_id']
        verb_id = hoi_map_to_obj_vb[hoi_id]['verb_id']

        hoi_det_this_image = []
        for inst_detection in det_this_image:
            if inst_detection[2][0] == obj_id:
                temp = {}
                temp['human_box'] = inst_detection[0].tolist()
                temp['object_box'] = inst_detection[1].tolist()
                temp['score'] = inst_detection[3][verb_id] * inst_detection[4] * inst_detection[5][0]
                hoi_det_this_image.append(temp)

        num_dets = len(hoi_det_this_image)
        sorted_idx = [idx for idx, _ in sorted(
            zip(range(num_dets), hoi_det_this_image),
            key=lambda x: x[1]['score'],
            reverse=True)]

        for i in sorted_idx:
            pred_det = {
                'human_box': hoi_det_this_image[i]['human_box'],
                'object_box': hoi_det_this_image[i]['object_box'],
                'score': hoi_det_this_image[i]['score']
            }
            is_match, candidate_gt_dets = match_hoi(pred_det, candidate_gt_dets)
            y_true.append(is_match)
            y_score.append(pred_det['score'])
            det_id.append((global_id, i))

    # Compute PR
    precision, recall = compute_pr(y_true, y_score, npos)
    # nprecision,nrecall,nap = compute_normalized_pr(y_true,y_score,npos)

    # Compute AP
    verb_name = hoi_map_to_obj_vb[hoi_id]['verb']
    object_name = hoi_map_to_obj_vb[hoi_id]['object']
    ap = compute_ap(precision, recall)
    print(f'hoi_id: {hoi_id} {verb_name}_{object_name} AP:{ap}')

    # Save AP data
    ap_data = {
        'y_true': y_true,
        'y_score': y_score,
        'det_id': det_id,
        'npos': npos,
        'ap': ap,
    }
    np.save(
        os.path.join(out_dir, f'{hoi_id}_ap_data.npy'),
        ap_data)

    return (ap, hoi_id)


def load_gt_dets(proc_dir, global_ids_set, subset):
    # Load anno_list
    print('Loading anno_list.json ...')
    anno_traintest_pkl = os.path.join(proc_dir, 'hico_anno_traintest_list.pkl')
    anno_traintest_list = pickle.load(open(anno_traintest_pkl, "rb"))

    if subset == 'test':
        anno_list = anno_traintest_list['test']
    else:
        anno_list = anno_traintest_list['train']

    gt_dets = {}
    for anno in anno_list:
        if anno['global_id'] not in global_ids_set:
            continue

        global_id = anno['global_id']
        gt_dets[global_id] = {}
        for hoi in anno['hois']:
            hoi_id = hoi['id']
            gt_dets[global_id][hoi_id] = []
            for human_box_num, object_box_num in hoi['connections']:
                human_box = hoi['human_bboxes'][human_box_num]
                object_box = hoi['object_bboxes'][object_box_num]
                det = {
                    'human_box': human_box,
                    'object_box': object_box,
                }
                gt_dets[global_id][hoi_id].append(det)

    return gt_dets


def compute_hico_map(out_dir, pred_hoi_dets, subset, num_processes=8):
    print('Creating output dir ...')
    mkdir(out_dir)

    # Load hoi_list
    hoi_list_pkl = os.path.join(DATA_DIR, 'hico_hoi_list.pkl')
    hoi_list = pickle.load(open(hoi_list_pkl, "rb"))
    # hoi_list = io.load_json_object(hoi_list_json)

    hoi_map_to_obj_vb = pickle.load(open(DATA_DIR + "/hico_hoi_map_to_obj_vb.pkl", "rb"))

    # Load subset ids to eval on
    split_ids_pkl = os.path.join(DATA_DIR, 'hico_split_ids.pkl')
    split_ids = pickle.load(open(split_ids_pkl, "rb"))
    # split_ids = io.load_json_object(split_ids_json)
    global_ids = split_ids[subset]
    global_ids_set = set(global_ids)

    # Create gt_dets
    print('Creating GT dets ...')
    gt_dets = load_gt_dets(DATA_DIR, global_ids_set, subset)

    eval_inputs = []
    for hoi in hoi_list:
        eval_inputs.append(
            (hoi['id'], global_ids, gt_dets, pred_hoi_dets, out_dir, hoi_map_to_obj_vb))

    print(f'Starting a pool of {num_processes} workers ...')
    p = Pool(num_processes)

    print(f'Begin mAP computation ...')
    output = p.starmap(eval_hoi, eval_inputs)
    # output = eval_hoi(245, global_ids, gt_dets, pred_hoi_dets, out_dir, hoi_map_to_obj_vb)

    p.close()
    p.join()

    mAP = {
        'AP': {},
        'mAP': 0,
        'invalid': 0,
    }
    map_ = 0
    count = 0
    for ap, hoi_id in output:
        mAP['AP'][hoi_id] = ap
        if not np.isnan(ap):
            count += 1
            map_ += ap

    mAP['mAP'] = map_ / count
    mAP['invalid'] = len(output) - count

    mAP_json = os.path.join(
        out_dir,
        'mAP.json')
    dump_json_object(mAP, mAP_json)

    print(f'APs have been saved to {out_dir}')

    ###############################################################################
    ################################# analysis ####################################
    ###############################################################################
    mAP = load_json_object(os.path.join(out_dir, 'mAP.json'))
    APs = mAP['AP']
    APs = {int(key): value for key, value in APs.items()}
    bin_to_hoi_ids = pickle.load(open(os.path.join(DATA_DIR, 'hico_bin_to_hoi_ids.pkl'), "rb"))

    bin_map = {}
    for bin_id, hoi_ids in bin_to_hoi_ids.items():
        bin_map[bin_id] = compute_mAP(APs, hoi_ids)

    non_rare_hoi_ids = []
    for ul in bin_to_hoi_ids.keys():
        if ul == '10':
            continue
        non_rare_hoi_ids += bin_to_hoi_ids[ul]

    sample_complexity_analysis = {
        'bin': bin_map,
        'full': compute_mAP(APs, APs.keys()),
        'rare': bin_map['10'],
        'non_rare': compute_mAP(APs, non_rare_hoi_ids)
    }

    sample_complexity_analysis_json = os.path.join(
        out_dir,
        f'sample_complexity_analysis.json')
    dump_json_object(
        sample_complexity_analysis,
        sample_complexity_analysis_json)

    bin_names = sorted([int(ul) for ul in bin_map.keys()])
    bin_names = [str(ul) for ul in bin_names]
    bin_headers = ['0'] + bin_names
    bin_headers = [bin_headers[i] + '-' + str(int(ul) - 1) for i, ul in enumerate(bin_headers[1:])]
    headers = ['Full', 'Rare', 'Non-Rare'] + bin_headers

    sca = sample_complexity_analysis
    values = [sca['full'], sca['rare'], sca['non_rare']] + \
             [bin_map[name] for name in bin_names]
    values = [str(round(v * 100, 2)) for v in values]

    fmodel = open(os.path.join(out_dir, 'model_map.txt'), "w+")
    fmodel.write(' '.join(headers) + '\n')
    fmodel.write(' '.join(values) + '\n')
    fmodel.write('\n')

    print('Space delimited values that can be copied to spreadsheet and split by space')
    print(' '.join(headers))
    print(' '.join(values))

def main():
    args = parser.parse_args()
    compute_hico_map(args.out_dir, args.pred_hoi_dets, args.subset, args.num_processes)


if __name__ == '__main__':
    main()