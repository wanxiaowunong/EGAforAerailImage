# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np

label = ['Imprevious surfaces',
         'Buildings',
         'Tree',
         'car',
         'Clutter/background',
         'low vegetable']

class runningScore(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))  #init confusion_matrix

    def _fast_hist(self, label_true, label_pred, n_class):    # input order by ground truth、predict、class number
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)    #?
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        for id in range(6):
            print ('===>' + label[id] + ':' + str(iu[id]))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {'Overall Acc: \t': acc,   # PA
                'Mean Acc : \t': acc_cls,  #mpa
                'FreqW Acc : \t': fwavacc,
                'Mean IoU : \t': mean_iu,}, cls_iu

    def get_all(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()    #pixel accuracy

        acc_cl = np.diag(hist) / hist.sum(axis=1)    #sum(axis=1)  hang  yongyu jisuan PA
        acc_cls = np.nanmean(acc_cl)  #calculate MPA

        acc_precission = np.diag(hist).sum() / (hist.sum(axis=0)+hist.sum(axis=1)+np.diag(hist).sum()-2*np.diag(hist))  # jisuan over accuracy
        OA=np.nanmean(acc_precission)

        F1=2*np.diag(hist) /(hist.sum(axis=1)+hist.sum(axis=0) )
        mF1=np.nanmean(F1)
        
        #acc_precission = np.diag(hist)/ hist.sum(axis=1)  # sum(axis=0) 
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))   #iou

        for id in range(self.n_classes):
            print ('===>' + label[id] + '   IoU:' + str(iu[id])+'   class pixel accuracy:'+str(acc_cl[id])+ '    acc_precission:'+str(acc_precission[id]))
        mean_iu = np.nanmean(iu)             #miou
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {'PA: \t': acc,
                'OA: \t': OA,
                'mF1 : \t': mF1,
                'MPA: \t': acc_cls,
                'FreqW Acc : \t': fwavacc,
                'Mean IoU : \t': mean_iu,}, cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
