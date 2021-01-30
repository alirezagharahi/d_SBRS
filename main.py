"""
References:
Jannach, Dietmar, and Malte Ludewig. "When recurrent neural networks meet the neighborhood for session-based recommendation." Proceedings of the Eleventh ACM Conference on Recommender Systems. 2017.
Hidasi, Balázs, et al. "Session-based recommendations with recurrent neural networks." arXiv preprint arXiv:1511.06939 (2015).
"""
import os
import time
import numpy as np
import pickle as pkl
import pandas as pd
from _datetime import timezone, datetime
import gc
from models import sknn,vsknn,vstan,stan,ar,sr,mc
import evaluation as eval
import performance_measures as per

def create_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--input_dataset', default='',
            help='Input path of the dataset.')

    parser.add_argument(
            '--approach', default='',
            help='Diversification appraoch.')

    return parser

def load_data( path, file_name):
    '''
    Desc: Loads a tuple of training and test set with the given parameters.
    --------
    Input:
        path : Path of preprocessed data (str)
        file : Name of dataset
    --------
    Output : tuple of DataFrame (train, test)
    '''

    print('START load data')
    st = time.time()
    sc = time.perf_counter()

    train_appendix = '_train'
    test_appendix = '_test'

    train = pd.read_csv(path + file_name + train_appendix +'.txt', sep='\t', dtype={'ItemId':np.int64})
    test = pd.read_csv(path + file_name + test_appendix +'.txt', sep='\t', dtype={'ItemId':np.int64} )

    data_start = datetime.fromtimestamp( train.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( train.Time.max(), timezone.utc )

    print('train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n'.
          format( len(train), train.SessionId.nunique(), train.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )

    data_start = datetime.fromtimestamp( test.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( test.Time.max(), timezone.utc )

    print('test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n'.
          format( len(test), test.SessionId.nunique(), test.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )

    print( 'END load data ', (time.perf_counter()-sc), 'c / ', (time.time()-st), 's' )

    return (train, test)


if __name__ == '__main__':


    diversification = "ID" # use "D" for diverse neighbor/rule, "I" for diverse candidate item and None for the original methods
    # read pickle of article embeddings

# chamealon
    os.chdir('.../glob1') # set the currect directory
    filename = 'articles_embeddings.pickle'
    infile = open(filename,'rb')
    with open(filename, 'rb') as f:
        content = pkl.load(f)
    content = pkl.load(infile)
    data = pd.read_csv('articles_metadata.csv',index_col=False)
    meta = pd.Series(data.category_id.values,index=data.article_id).to_dict()
    data_path = '.../slices/'
    file_prefix = 'glob1'

    # create a list of metric classes to be evaluated
    metric = []

    metric.append(per.Precision(10))
    metric.append(per.Diversity(10) )
    metric.append(per.DiversityRankRelavance(10) )
    metric.append(per.topic_coverage(10))

    # predictor
    algs = {}

    # rule-based

    ara = ar.AssosiationRules();
    algs['ar'] = ara

    mca = mc.MarkovModel()
    algs['markov'] = mca

    sra = sr.SequentialRules( steps = 15, weighting='same', pruning=25, last_n_days=None )
    algs['sr'] = sra

    # neighborhood_based

    sknna = sknn.ContextKNN( 200, sample_size=500, similarity="cosine", extend=False )
    algs['sknn'] = sknna

    vknna = vsknn.VMContextKNN( 300, sample_size=2500, similarity="cosine",weighting="log",weighting_score="quadratic_score", last_n_days=None, extend=False )
    algs['vsknn'] = vknna

    stana = stan.STAN( 100, sample_size=2500, sampling='recent', remind=True, extend=False, lambda_spw=5, lambda_snh=4.42, lambda_inh=5 , session_key = 'SessionId', item_key= 'ItemId', time_key= 'Time' )
    algs['stan'] = stana

    vstana = vstan.VSKNN_STAN( 300, sample_size=2500, sampling='recent', remind=True, extend=False, similarity='cosine', lambda_spw=3.95, lambda_snh=2.71, lambda_inh=4.68, lambda_ipw=3.25, lambda_idf=1.5, session_key = 'SessionId', item_key= 'ItemId', time_key= 'Time' )
    algs['vstan'] = vstana

    # load data
    train, test = load_data(data_path, file_prefix)

    # train algorithms
    ts = time.time()

    for m in metric:
        if isinstance(m,per.topic_coverage):
            m.init(train,content,meta)
        else:
            m.init(train,content)

    # train algorithms
    for k, a in algs.items():
        ts = time.time()
        print('fit ', k)
        a.fit(train,content,d=diversification)
        print(k, ' time: ', (time.time() - ts))

        res = {};

        # evaluation
        keys = list(algs.keys())

    for k in keys:
        res[k] = eval.evaluate_sessions(algs[k], metric, test, train)
        del algs[k]
        for i in range(3): gc.collect()

    # print results
    for k, l in res.items():
        for e in l:
            print(k, ':', e[0], ' ', e[1])

    del metric
