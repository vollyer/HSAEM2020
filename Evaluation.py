import numpy as np
import sklearn.metrics as sk_metrics



#用sklearn 计算评价指标  测试集可以是1：1  也可以是全部负样本都加入到测试集
#测试术数据在mask中标注为1
def calculate_metrics_sk(pred, truth, mask):
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    mask = np.asarray(mask)
    R_vers = np.ones(truth.shape) - truth
    mask = mask + R_vers
    r_num = truth.shape[0]
    auc_test = 0
    aupr_test = 0
    pre_test = 0
    rec_test = 0
    f1_test = 0
    mcc_test = 0

    bad_rows = 0
    for i in range(r_num):
        row_truth_test = truth[i, mask[i, :] == 1]
        row_pred_test = pred[i, mask[i, :] == 1]
        mean = np.mean(row_truth_test)
        if mean == 0 or mean == 1 or row_truth_test.shape[0] < 2:
            bad_rows = bad_rows + 1
            continue
        gather_idxs = np.argsort(row_pred_test)[-1::-1]
        row_truth_test_gathered = np.take_along_axis(row_truth_test, gather_idxs, 0)
        non_zeros = np.count_nonzero(row_truth_test)
        row_pred_test_binary = np.zeros(row_pred_test.shape)
        row_pred_test_binary[:non_zeros] = 1

        auc_test = auc_test + sk_metrics.roc_auc_score(row_truth_test, row_pred_test)
        aupr_test = aupr_test + sk_metrics.average_precision_score(row_truth_test, row_pred_test)

        pre_test = pre_test + sk_metrics.precision_score(row_truth_test_gathered, row_pred_test_binary)
        rec_test = rec_test + sk_metrics.recall_score(row_truth_test_gathered, row_pred_test_binary)
        f1_test = f1_test + sk_metrics.f1_score(row_truth_test_gathered, row_pred_test_binary)
        mcc_test = mcc_test + sk_metrics.matthews_corrcoef(row_truth_test_gathered, row_pred_test_binary)

    if not r_num == bad_rows:
        auc_test = auc_test / (r_num - bad_rows)
        aupr_test = aupr_test / (r_num - bad_rows)
        pre_test = pre_test / (r_num - bad_rows)
        rec_test = rec_test / (r_num - bad_rows)
        f1_test = f1_test / (r_num - bad_rows)
        mcc_test = mcc_test / (r_num - bad_rows)

    print("auc,  aupr,  f1, mcc")
    return  auc_test,  aupr_test,  f1_test, mcc_test




#!!!!!mask矩阵里面的1是被测试的正样本   其余部分都为0
def AUC_main(pred_mat, truth_mat, mask_mat):
    def ROC(Rank, n):
        Rank = Rank.T
        loop = Rank.shape[0]
        numerator = 0
        TP = 0
        FP = 0
        ALLTP = len([val for val in Rank if val > 0])

        for i in range(0, loop):
            if Rank[i] == 1:
                TP = TP + 1
            else:
                FP = FP + 1
                numerator = numerator + TP
            if FP >= n:
                break

        denominator = FP * ALLTP
        if numerator == 0:
            z = 0
        else:
            z = numerator / denominator
        return z

    # def AUPR(predict, real):
    #     index = (-predict).argsort()
    #     order = real[index]
    #     num = len([val for val in real if val != 0])
    #     location = [i for (i, val) in enumerate(order) if val > 0]
    #     pr = np.divide(range(1, num + 1), np.array(location) + 1)
    #     pr[num - 1] = pr[num - 1]
    #     area = sum(pr) / num
    #     return area

    index = (-pred_mat).argsort()
    pnum = pred_mat.shape[0]
    gnum = pred_mat.shape[1]
    denominator = pnum
    auc = 0
    auc10 = 0
    auc20 = 0
    auc50 = 0
    auc100 = 0
    aupr = 0
    for j in range(0, pnum):
        mask_j = mask_mat[j, :]
        predict_j = mask_mat[j, index[j, :]]
        predict_j1 = truth_mat[j, index[j, :]]
        if sum(mask_j) > 0:
            auc = auc + ROC(predict_j, gnum)
            auc10 = auc10 + ROC(predict_j, 10)
            auc20 = auc20 + ROC(predict_j, 20)
            auc50 = auc50 + ROC(predict_j, 50)
            auc100 = auc100 + ROC(predict_j, 100)
        else:
            denominator = denominator - 1

    AUC = auc / denominator
    AUC10 = auc10 / denominator
    AUC20 = auc20 / denominator
    AUC50 = auc50 / denominator
    AUC100 = auc100 / denominator
    # AUPR = aupr / denominator
    print("AUC  AUC10, AUC20, AUC50, AUC100")
    return AUC, AUC10, AUC20, AUC50, AUC100


# def get_accuracy_scores(pred_mat, truth_mat):
#     pnum = pred_mat.shape[0]
#     gnum = pred_mat.shape[1]
#     roc_sc = 0
#     aupr_sc = 0
#     for i in range(pnum):
#         roc_sc = roc_sc + sk_metrics.roc_auc_score(truth_mat[i], pred_mat[i])
#         aupr_sc = aupr_sc + sk_metrics.average_precision_score(truth_mat[i], pred_mat[i])
#     roc_sc = roc_sc / pnum
#     aupr_sc = aupr_sc / pnum
#
#     return roc_sc, aupr_sc
