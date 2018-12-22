import pandas as pd
import numpy as np
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

'''
    this file is to extract feature.
'''

"""Configuration"""
FEATURE_PATH = './feature/'
DATA_PATH = './original_data/'

api_vec = TfidfVectorizer(ngram_range=(1, 4),
                          min_df=3,
                          max_df=0.9,
                          strip_accents='unicode',
                          use_idf=1, smooth_idf=1, sublinear_tf=1)

# load original data, include train and text data set
def loadData():
    print("Loading original data ...")
    train_o_d = pd.read_csv(DATA_PATH+'/train.csv')
    test_o_d = pd.read_csv(DATA_PATH+'/test.csv')
    print("Loading completes")
    return train_o_d, test_o_d

def tfidfModelTrain(train, test):
    tr_api = train.groupby('file_id')['api'].apply(lambda x: ' '.join(x)).reset_index()
    te_api = test.groupby('file_id')['api'].apply(lambda x: ' '.join(x)).reset_index()
    tr_api_vec = api_vec.fit_transform(tr_api['api'])
    val_api_vec = api_vec.transform(te_api['api'])
    return (tr_api_vec, val_api_vec)


# NB-LR
def pr(x, y_i, y):
    p = x[y == y_i].sum(0)
    return (p + 1) / ((y == y_i).sum() + 1)


def get_mdl(x, y):
    y = y.values
    r = np.log(pr(x, 1, y) / pr(x, 0, y))
    np.random.seed(0)
    m = LogisticRegression(C=6, dual=True, random_state=0)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r


def nblrTrain(tr_tfidf_rlt, te_tfidf_rlt, train):
    label_fold = []
    preds_fold_lr = []
    lr_oof = pd.DataFrame()
    preds_te = []
    skf = StratifiedKFold(n_splits=5, random_state=4, shuffle=True)
    for fold_i, (tr_index, val_index) in enumerate(skf.split(train, train['label'])):
        if fold_i >= 0:
            tr, val = train.iloc[tr_index], train.iloc[val_index]
            x = tr_tfidf_rlt[tr_index, :]
            test_x = tr_tfidf_rlt[val_index, :]
            preds = np.zeros((len(val), 1))
            preds_te_i = np.zeros((te_tfidf_rlt.shape[0], 1))
            labels = [i for i in range(1)]
            for i, j in enumerate(labels):
                print('fit', j)
                m, r = get_mdl(x, tr['label'] == j)
                preds[:, i] = m.predict_proba(test_x.multiply(r))[:, 1]
                preds_te_i[:, i] = m.predict_proba(te_tfidf_rlt.multiply(r))[:, 1]
            preds_te.append(preds_te_i)
            preds_lr = preds
            lr_oof_i = pd.DataFrame({'file_id': val['file_id']})
            for i in range(1):
                lr_oof_i['prob' + str(i)] = preds[:, i]
            lr_oof = pd.concat([lr_oof, lr_oof_i], axis=0)

            for i, j in enumerate(preds_lr):
                preds_lr[i] = j / sum(j)

            label_fold.append(val['label'].tolist())
            preds_fold_lr.append(preds_lr)

            lr_oof = lr_oof.sort_values('file_id')
            preds_te_avg = (np.sum(np.array(preds_te), axis=0) / 5)
            lr_oof_te = pd.DataFrame({'file_id': range(0, te_tfidf_rlt.shape[0])})
            for i in range(1):
                lr_oof_te['prob' + str(i)] = preds_te_avg[:, i]
    return (lr_oof, lr_oof_te)

def makeFeature(data, is_train=True):
    '''
    file_cnt: How many samples does file have
    tid_distinct_cnt: How many threads the file launch
    api_distinct_cnt: How many distinct numbers of APT called by file
    tid_api_cnt_max,tid_api_cnt_min,tid_api_cnt_mean: the max/min/mean number of API called threads of file
    tid_api_distinct_cnt_max, tid_api_distinct_cnt_min, tid_api_distinct_cnt_mean: the distinct max/min/mean number of API called threads of file
    '''

    if is_train:
        return_data = data[['file_id', 'label']].drop_duplicates()
    else:
        return_data = data[['file_id']].drop_duplicates()

    feat = data.groupby(['file_id']).tid.count().reset_index(name='file_cnt')
    return_data = return_data.merge(feat, on='file_id', how='left')

    ################################################################################
    feat = data.groupby(['file_id']).agg({'tid': pd.Series.nunique, 'api': pd.Series.nunique}).reset_index()
    feat.columns = ['file_id', 'tid_distinct_cnt', 'api_distinct_cnt']
    return_data = return_data.merge(feat, on='file_id', how='left')
    ################################################################################
    feat_tmp = data.groupby(['file_id', 'tid']).agg({'index': pd.Series.count, 'api': pd.Series.nunique}).reset_index()
    feat = feat_tmp.groupby(['file_id'])['index'].agg(['max', 'min', 'mean']).reset_index()
    feat.columns = ['file_id', 'tid_api_cnt_max', 'tid_api_cnt_min', 'tid_api_cnt_mean']
    return_data = return_data.merge(feat, on='file_id', how='left')

    feat = feat_tmp.groupby(['file_id'])['api'].agg(['max', 'min', 'mean']).reset_index()
    feat.columns = ['file_id', 'tid_api_distinct_cnt_max', 'tid_api_distinct_cnt_min', 'tid_api_distinct_cnt_mean']
    return_data = return_data.merge(feat, on='file_id', how='left')

    return return_data

def makeFeature_v2(data):
    return_data = data[['file_id']].drop_duplicates()

    # count the number of api called by file
    tmp = data.groupby(['file_id']).api.count()

    # caculate the min Index called by api
    feat = data.groupby(['file_id', 'api'])['index'].min().reset_index(name='val')
    feat = feat.pivot(index='file_id', columns='api', values='val')
    feat.columns = [feat.columns[i] + '_index_min' for i in range(feat.shape[1])]
    feat_withFileid = feat.reset_index()
    return_data = return_data.merge(feat_withFileid, on='file_id', how='left')
    #  count the number of called api
    feat = data.groupby(['file_id', 'api'])['index'].count().reset_index(name='val')
    feat = feat.pivot(index='file_id', columns='api', values='val')
    feat.columns = [feat.columns[i] + '_cnt' for i in range(feat.shape[1])]
    feat_withFileid = feat.reset_index()
    return_data = return_data.merge(feat_withFileid, on='file_id', how='left')
    #  caculate the proportion of api's calling
    feat_rate = pd.concat([feat, tmp], axis=1)
    feat_rate = feat_rate.apply(lambda x: x / feat_rate.api)
    feat_rate.columns = [feat_rate.columns[i] + '_rate' for i in range(feat_rate.shape[1])]
    feat_rate_withFileid = feat_rate.reset_index().drop(['api_rate'], axis=1)
    return_data = return_data.merge(feat_rate_withFileid, on='file_id', how='left')

    return return_data



if __name__ == '__main__':
    #load original data
    traindata, testdata = loadData()

    #make train data features
    train_base_feature_v1 = makeFeature(traindata, True)
    print('Base Train Data: ', train_base_feature_v1.shape)
    train_base_feature_v1.to_csv(FEATURE_PATH + 'train_base_features_v1.csv', index=None)

    train_base_feature_v2 = makeFeature_v2(traindata)
    print('Base Train Data: ', train_base_feature_v2.shape)
    train_base_feature_v2.to_csv(FEATURE_PATH + 'train_base_features_v2.csv', index=None)

    # make test data features
    test_base_feature_v1 = makeFeature(testdata, False)
    print('Base Test Data: ', test_base_feature_v1.shape)
    test_base_feature_v1.to_csv(FEATURE_PATH + 'test_base_features_v1.csv', index=None)

    test_base_feature_v2 = makeFeature_v2(testdata)
    print('Base Test Data: ', test_base_feature_v2.shape)
    test_base_feature_v2.to_csv(FEATURE_PATH + 'test_base_features_v2.csv', index=None)

    # make TFIDF and over_prob feature engineering
    tr_api_vec, val_api_vec = tfidfModelTrain(traindata, testdata)
    scipy.sparse.save_npz(FEATURE_PATH + 'tr_tfidf_rlt.npz', tr_api_vec)
    scipy.sparse.save_npz(FEATURE_PATH + 'te_tfidf_rlt.npz', val_api_vec)

    tr_prob, te_prob = nblrTrain(tr_api_vec, val_api_vec, train_base_feature_v1)
    tr_prob.to_csv(FEATURE_PATH + 'tr_lr_oof_prob.csv', index=False)
    te_prob.to_csv(FEATURE_PATH + 'te_lr_oof_prob.csv', index=False)