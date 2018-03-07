import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import collections
import time
import cPickle as pickle

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_roc = []
_roc_r = []
_iter = [0]

def tick():
    _iter[0] += 1

def tick_reset():
    _iter[0] = 0

def plot(name, value):
    _since_last_flush[name][_iter[0]] = value

def plot_roc(val,anom):
    _roc.append([val, anom, _iter[0]])

def plot_roc_r(val,anom):
    _roc_r.append([val, anom, _iter[0]])

def flush():
    prints = []

    for name, vals in _since_last_flush.items():
        labels = None
        if ':' in name:
            name,labels=name.split(':')
            labels=labels.split(',')

        prints.append("{}\t{}".format(name, np.mean(vals.values(), axis=0)))
        _since_beginning[name].update(vals)

        x_vals = np.sort(_since_beginning[name].keys())
        y_vals = [_since_beginning[name][x] for x in x_vals]

        plt.clf()
        plt.plot(x_vals, y_vals)
        plt.xlabel('iteration')
        plt.ylabel(name)
        if labels != None: plt.legend(labels, loc='upper right')
        plt.savefig(name.replace(' ', '_')+'.jpg')

    if len(_roc) > 0:
        l = len(_roc) - 1
        ROC('discriminator_ROC_', [_roc[l/3], _roc[2*l/3], _roc[l]])

    if len(_roc_r) > 0:
        l = len(_roc_r) - 1
        ROC('generator_ROC_', [_roc_r[l/3], _roc_r[2*l/3], _roc_r[l]])

    print "iter {}\t{}".format(_iter[0], "\n".join(prints))
    _since_last_flush.clear()

    with open('log.pkl', 'wb') as f:
        pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)

def log_auc(x, y, l):
    prt = [(np.log10(x[i+1]) - np.log10(x[i])) * (0.5*(y[i+1]+y[i])) for i in xrange(len(x)-1) ]
    return np.sum(prt) / abs(np.log10(l))

def fpr_tpr(val,anom):
    labels = np.concatenate([ np.zeros_like(val), np.ones_like(anom)], axis=0)
    scores = np.concatenate([ val, anom], axis=0 )

    sort_order = np.argsort(-scores)
    labels = np.array(labels)[sort_order]
    scores = np.array(scores)[sort_order]

    length = len(anom) + 2
    fp,tp = np.zeros(length),np.zeros(length)
    f,p = 0,0
    for l,s in zip(labels,scores):
        if l==1:
            p += 1
            fp[p],tp[p] = f,p
        else: f += 1
    
    fp[-1],tp[-1] = len(val),len(anom)
    fpr,tpr = np.array(fp + 0.0) / len(val), np.array(tp + 0.0) / len(anom)
    fpr[fpr==0] = 0.5 / len(val)
    return fpr,tpr

'''
def fpr_tpr(val,anom):
    tp = np.arange(1,len(anom)+2)
    tp[-1] = tp[-2]
    
    anom_rsort = np.sort(anom)[::-1]
    val_rsort = np.sort(val)[::-1]
    fp = np.zeros_like(anom_rsort)
    
    _fp = 0
    for ind,value in enumerate(anom_rsort):
        while _fp < len(val) and val_rsort[_fp] > value : _fp += 1
        fp[ind] = _fp

    fp = np.append(fp, len(val))
    return ((np.array(fp) + 0.0) / len(val), (np.array(tp) + 0.0) / len(anom))
'''


def ROC_clean(name, discs, recons):
    plt.clf()
    plt.title('Receiver Operating Characteristic')
    
    for cost_name,scores in zip(('Discriminator', 'Generator'),(discs,recons)):
        x,y = fpr_tpr(scores[0],scores[1])
        plt.plot(x,y, label='{0}, AUC={1:0.2f}'.format(cost_name, np.trapz(y,x)))
    plt.plot([0, 1], [0, 1],'r--', label='Chance')
    plt.legend(loc = 0)
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(name + '.png')
    plt.clf()

def ROC_clean_log(name, discs, recons):
    l = 0.5 / len(discs[0])
    
    plt.clf()
    plt.title('Receiver Operating Characteristic')
    for cost_name,scores in zip(('D.', 'G.'),(discs,recons)):
        x,y = fpr_tpr(scores[0],scores[1])
        logfar = np.log10(x[1])
        plt.semilogx(x,y, label='{0} AUC={1:0.2f} lgAUC={2:0.2f} lgFAR={3:0.2f}'.format(
            cost_name, np.trapz(y,x), log_auc(x,y,l), logfar))
    plt.semilogx(np.linspace(l,1,100), np.linspace(l,1,100),'r--', label='Chance')
    plt.legend(loc = 'upper right')
    plt.xlim([l, 1.0])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(name + '.png')
    plt.clf()
    



def ROC(name, args, log=False):
    #args : 3 triples:
    #validation scores, anomaly scores

    fprs, tprs = [], []
    for i in xrange(3):
        granularity = 1000
        val = args[i][0]
        anom = args[i][1]

        min_score = min(np.min(val), np.min(anom))
        max_score = max(np.max(val), np.max(anom))
        thr = np.linspace(max_score, min_score, granularity)
        
        roc_x = [np.sum(val  >= T) for T in thr]
        roc_y = [np.sum(anom >= T) for T in thr]
        
        fprs.append(( np.array(roc_x) + 0.0) / len(val) )
        tprs.append(( np.array(roc_y) + 0.0) / len(anom))

    plt.clf()
    plt.title('Receiver Operating Characteristic')
    colors = ['lightgreen', 'lightskyblue', 'b']
    linestyles = ['--','--', '-']
    for i in xrange(3):
        plt.plot(fprs[i], tprs[i], color=colors[i], linestyle=linestyles[i], label='ROC curve, {0} iters, auc = {1:0.2f}'.format(args[i][2], np.trapz(tprs[i],fprs[i])))
    plt.plot([0, 1], [0, 1],'r--', label='Guessing equivalent curve')
    plt.legend(loc = 'lower right')
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(name + str(_iter[0]) + '.png')
    plt.clf()

    if log:
        plt.title('Receiver Operating Characteristic')
        colors = ['lightgreen', 'lightskyblue', 'b']
        linestyles = ['--','--', '-']
        for i in xrange(3):
            plt.semilogx(fprs[i], tprs[i], color=colors[i], linestyle=linestyles[i], label='ROC curve, {0} iters, auc = {1:0.2f}'.format(args[i][2], log_auc(fprs[i],tprs[i])))
        plt.plot([0, 1], [0, 1],'r--', label='Guessing equivalent curve')
        plt.legend(loc = 'lower right')
        plt.xlim([0, 1.05])
        plt.ylim([0, 1.05])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(name + 'log_' + str(_iter[0]) + '.png')
        plt.clf()

def hist_disc(val, anom):
    val,anom = -val,-anom
    
    #Negates and zero-shifts the values
    #This makes the histogram more directly comparable to the generator histogram
    min_value = min(np.min(val), np.min(anom))
    val -= min_value
    anom -= min_value

    iteration = _iter[0]
    
    plt.close()
    plt.hist([val, anom], color=['g','r'], label=['Validation', 'Anomaly'], alpha=0.8, bins=25, normed=True)

    plt.title('Histogram of negated and zeroshifted discrimination scores ' + str(iteration))
    plt.xlabel('Discrimination score')
    plt.ylabel('Normalized sample count')

    plt.legend(loc='upper right')
    plt.savefig('discriminator_negshiftscore_histogram_{}'.format(iteration))
    plt.close()

def hist_gen(val, anom):
    iteration = _iter[0]
    hist_max = 5*np.mean(val) + 0.5*np.max(val)
    anom[anom>hist_max] = hist_max
    val[val>hist_max] = hist_max

    plt.close()
    plt.hist([val, anom], color=['g','r'], label=['Validation', 'Anomaly'], alpha=0.8, bins=25, normed=True)

    plt.title('Histogram of reconstruction costs ' + str(iteration))
    plt.xlabel('Reconstruction cost')
    plt.ylabel('Normalized sample count')

    plt.legend(loc='upper right')
    plt.savefig('generator_recreation_cost_histogram_{}'.format(iteration))
    plt.close()