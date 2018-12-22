import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
'''
    this file is to predict.
'''

"""Configuration"""
FEATURE_PATH = './feature/'
RESULT_PATH = './result/'

# define the number of iterations
ROUND = 1000

# configuration the parameters of xgboost
xgb_params = {'objective':'multi:softprob',
              'num_class': 8,
              'eta': 0.04,
              'max_depth':6,
              'subsample':0.9,
              'colsample_bytree': 0.7,
              'lambda': 2,
              'alpha': 2,
              'gamma': 1,
              'scale_pos_weight': 20,
              'eval_metric': 'mlogloss',
              'silent': 0,
              'seed': 149}


def xgb_train(X_train, X_val, y_train, y_val, test, num_round):
    # multi-class model
    dtrain = xgb.DMatrix(X_train, y_train)
    dval = xgb.DMatrix(X_val, y_val)
    dtest = xgb.DMatrix(test.drop(['file_id'], axis=1))
    watchlist = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(xgb_params, dtrain, num_round, evals=watchlist, early_stopping_rounds=100, verbose_eval=100)
    p_val = pd.DataFrame(model.predict(dval, ntree_limit=model.best_iteration), index=X_val.index)
    p_test = pd.DataFrame(model.predict(dtest, ntree_limit=model.best_iteration), index=test.index)
    return (model, p_val, p_test)

def run():
    print("Loading feature ...")
    # load feature v1
    train_1 = pd.read_csv(FEATURE_PATH + 'train_base_features_v1.csv')
    test_1 = pd.read_csv(FEATURE_PATH + 'test_base_features_v1.csv')
    # load feature v2
    train_2 = pd.read_csv(FEATURE_PATH + 'train_base_features_v2.csv')
    test_2 = pd.read_csv(FEATURE_PATH + 'test_base_features_v2.csv')

    interaction_feat = train_2.columns[train_2.columns.isin(test_2.columns.values)].values
    train_2 = train_2[interaction_feat]
    test_2 = test_2[interaction_feat]

    # merge all features
    train = train_1.merge(train_2, on=['file_id'], how='left')
    test = test_1.merge(test_2, on=['file_id'], how='left')

    # train data prepare
    X = train.drop(['file_id', 'label'], axis=1)
    y = train['label']

    # add one_vs_rest prob
    extra_feat_val = pd.read_csv(FEATURE_PATH + 'tr_lr_oof_prob.csv')
    extra_feat_test = pd.read_csv(FEATURE_PATH + 'te_lr_oof_prob.csv')
    prob_list = ['prob' + str(i) for i in range(1)]
    X_extra = pd.concat(
        [X, extra_feat_val[prob_list]], axis=1)
    test_extra = pd.concat(
        [test, extra_feat_test[prob_list]], axis=1)
    print("Loading complete")

# multi-class model training
    logloss_rlt = []
    p_val_all = pd.DataFrame()
    # 8 catagories
    p_test_all = pd.DataFrame(np.zeros((test.shape[0], 8)))
    skf = StratifiedKFold(n_splits=5, random_state=4, shuffle=True)
    # start 5-fold CV
    for fold_i, (tr_index, val_index) in enumerate(skf.split(X, y)):
        print('FOLD -', fold_i, ' Start...')
        # Prepare train, val dataset
        X_train, X_val = X_extra.iloc[tr_index, :], X_extra.iloc[val_index, :]
        y_train, y_val = y[tr_index], y[val_index]
        # Train model

        model, p_val, p_test = xgb_train(X_train, X_val, y_train, y_val, test_extra, ROUND)
        # Evaluate Model and Concatenate Val-Prediction
        m_log_loss = log_loss(y_val, p_val)
        print('----------------log_loss : ', m_log_loss, ' ---------------------')
        logloss_rlt = logloss_rlt + [m_log_loss]
        truth_prob_df = pd.concat([y_val, p_val], axis=1)
        p_val_all = pd.concat([p_val_all, truth_prob_df], axis=0)
        # Predict Test Dataset
        p_test_all = p_test_all + 0.2 * p_test

    # generate submit file
    rlt = pd.concat([test['file_id'], p_test_all], axis=1)
    prob_list = ['prob' + str(i) for i in range(8)]
    rlt.columns = ['file_id'] + prob_list
    rlt.to_csv(RESULT_PATH + '/submit.csv', index=None)

if __name__ == '__main__':
    run()
