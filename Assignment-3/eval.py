from submission import train_and_test

def eval(args):
    readline = lambda line: line.strip().split('/')
    words = []
    groundtruth_tags = []
    predicted_tags = []
    with open(args.gold) as fgold, open(args.pred) as fpred:
        for g, p in zip(fgold, fpred):
            gw, gt = readline(g)
            pw, pt = readline(p)
            if gw == '###':
                continue
            words.append(gw)
            predicted_tags.append(pt)
            groundtruth_tags.append(gt)
    acc = sum([pt == gt for gt, pt in zip(groundtruth_tags, predicted_tags)]) / len(predicted_tags)
    print('accuracy={}'.format(acc))
    return acc


