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

import cknn
import evaluation as eval
import performance_measures as per

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
   
    # read pickle of article embeddings
    os.chdir('../')
    filename = 'article_embeddings.pickle'
    infile = open(filename,'rb')
    content = pkl.load(infile)

    # read the preprocessed data
#    os.chdir('../')
    data_path = '../'
    file_prefix = 'adressa'
            
    # create a list of metric classes to be evaluated
    metric = []
    
    metric.append(per.Precision(20))
    metric.append(per.Diversity(20) )
    metric.append(per.DiversityRankRelavance(20) )


    # predictor
    cknn_model = cknn.ContextKNN( 100, 500, last_n_days=None, extend=False )
    
    # load data            
    train, test = load_data(data_path, file_prefix)
    item_ids = train.ItemId.unique()

    # train algorithms
    ts = time.time()
    cknn_model.fit(train,content)

    
#    # init metrics (for evaluation)
#    for m in metric:
#        m.init(content)
#     
#    # evaluation
#    result = eval.evaluate_sessions(vknna, metric, test, train)
#  
#    # print results
#    for e in result:
#        print( e[0], ' ', e[1])
#      
#    del metric