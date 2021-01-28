"""
References:
Jannach, Dietmar, and Malte Ludewig. "When recurrent neural networks meet the neighborhood for session-based recommendation." Proceedings of the Eleventh ACM Conference on Recommender Systems. 2017.
Hidasi, Balázs, et al. "Session-based recommendations with recurrent neural networks." arXiv preprint arXiv:1511.06939 (2015).
"""

import time
import numpy as np

def evaluate_sessions(fitted_predictor, metrics, test_set, train_set, session_key='SessionId', item_key='ItemId', time_key='Time', hiden=2): 
    '''
    Desc: Evaluates the fitted algoritm measured by accuracy and diversity metrics
    --------
    Input:
        fitted_algorithm : A trained instance of a predictor.
        metrics : List of metrics
        test_set : Interactions for testing. A dataframe with a column for session IDs, a column for item IDs and a column for the timestamp of the events
        train_data :  Training data. 
        session_key : Header of the session ID column in the input file (default: 'SessionId') str
        item_key : Header of the item ID column in the input file (default: 'ItemId') str
        time_key : Header of the timestamp column in the input file (default: 'Time') str
        hidden: number of events that should be hidden from each test session
    --------
    Output: List of tuples (metric_name, value)
    '''
    actions = len(test_set)
    sessions = len(test_set[session_key].unique())
    count = 0
    print('START evaluation of ', actions, ' actions in ', sessions, ' sessions')
    
    sc = time.clock();
    st = time.time();
    
    time_sum = 0
    time_sum_clock = 0
    time_count = 0
    
    for m in metrics:
        m.reset();
    
    test_set.sort_values([session_key, time_key], inplace=True)
    
    # some times there are duplicates in a session (user interacted with same article multiple times)
    train_set = train_set.drop_duplicates([session_key,item_key],keep= 'first')
    # test_data = test_data.drop_duplicates([session_key,item_key],keep= 'first')

    test_set = test_set.reset_index(drop=True)

    # for evaluation we can use items in the train set for scoring but in real prediction we will use available item set   
    items_to_predict = train_set[item_key].unique()
    
    # an array containing cumlated length of each session starting from 0
    offset_sessions = np.zeros(test_set[session_key].nunique()+1, dtype=np.int32)

    # an array containing length of each session
    length_session = np.zeros(test_set[session_key].nunique(), dtype=np.int32)
    
    offset_sessions[1:] = test_set.groupby(session_key).size().cumsum()
    length_session[0:] = test_set.groupby(session_key).size()
    
    current_session_idx = 0
    pos = offset_sessions[current_session_idx+1]-(hiden+1)
    finished = False

    # we put sessions in chunks of 1000 sessions
    while not finished:
        
        if count % 1000 == 0:
            print( '    eval process: ', count, ' of ', sessions, ' sessions: ', ( count / sessions * 100.0 ), ' % in',(time.time()-st), 's')
            
        crs = time.clock();
        trs = time.time();
        
        current_item = test_set[item_key][pos]
        current_session = test_set[session_key][pos]
        ts = test_set[time_key][pos]
        # items that predictor will use to predict next items
        item_set = test_set[item_key][offset_sessions[current_session_idx]:pos+1].values

        # ground truth
        rest = test_set[item_key][pos+1:offset_sessions[current_session_idx]+length_session[current_session_idx]].values
        # predictions
        preds = fitted_predictor.predict_next(current_session, current_item, item_set,items_to_predict, timestamp=ts)

        preds[np.isnan(preds)] = 0
        preds.sort_values( ascending=False, inplace=True )
        
        time_sum_clock += time.clock()-crs
        time_sum += time.time()-trs
        time_count += 1
        
        count += 1
        
        # evaluating using initiated metrics
        if len(rest) > 0:
            for m in metrics:
                m.add_set( preds, rest, for_item=current_item, session=current_session )
                
        current_session_idx += 1
        if current_session_idx == test_set[session_key].nunique():
            finished = True        
        else:
            pos = offset_sessions[current_session_idx+1]-(hiden+1)
    
    print( 'END evaluation in ', (time.clock()-sc), 'c / ', (time.time()-st), 's' )
    print( '    avg rt ', (time_sum/time_count), 's / ', (time_sum_clock/time_count), 'c' )
    print( '    time count ', (time_count), 'count/', (time_sum), ' sum' )
    
    res = []
    for m in metrics:
        res.append( m.result() )
    
    return res

