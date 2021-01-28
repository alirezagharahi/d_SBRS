"""
References:
Malte Ludewig and Dietmar Jannach. 2018. Evaluation of session-based recommendation algorithms.User Modeling and User-Adapted Interaction28,4-5 (2018), 331–390.
"""

import numpy as np
import pandas as pd
from math import log10
import collections as col
from datetime import datetime as dt
from datetime import timedelta as td

class SequentialRules:
    '''
    SequentialRules(steps = 10, weighting='div', pruning=20.0, last_n_days=None, session_key='SessionId', item_key='ItemId', time_key='Time')

    Parameters
    --------
    steps : int
        Number of steps to walk back from the currently viewed item. (Default value: 10)
    weighting : string
        Weighting function for the previous items (linear, same, div, log, qudratic). (Default value: div)
    pruning : int
        Prune the results per item to a list of the top N sequential co-occurrences. (Default value: 20).
    last_n_days : int
        Only use the last N days of the data for the training process. (Default value: None)
    session_key : string
        The data frame key for the session identifier. (Default value: SessionId)
    item_key : string
        The data frame key for the item identifier. (Default value: ItemId)
    time_key : string
        The data frame key for the timestamp. (Default value: Time)

    '''

    def __init__( self, steps = 10, weighting='div', pruning=20, last_n_days=None, session_key='SessionId', item_key='ItemId', time_key='Time' ):
        self.steps = steps
        self.pruning = pruning
        self.weighting = weighting
        self.last_n_days = last_n_days
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.session = -1
        self.session_items = []
        self.session_item_map = dict()
        self.session_diversity_map = dict()


    def fit( self,data,content, d = None):
        '''
        Trains the predictor.

        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).


        '''

        s = data.groupby("SessionId")["ItemId"].apply(lambda x: set(x.tolist()))
        self.session_item_map = s.to_dict()
        self.content = content
        self.d = d

        if self.last_n_days != None:

            max_time = dt.fromtimestamp( data[self.time_key].max() )
            date_threshold = max_time.date() - td( self.last_n_days )
            stamp = dt.combine(date_threshold, dt.min.time()).timestamp()
            train = data[ data[self.time_key] >= stamp ]

        else:
            train = data

        cur_session = -1
        last_items = []
        rules = dict()

        index_session = train.columns.get_loc( self.session_key )
        index_item = train.columns.get_loc( self.item_key )

        for row in train.itertuples( index=False ):

            session_id, item_id = row[index_session], row[index_item]

            if session_id != cur_session:
                cur_session = session_id
                last_items = []
            else:
                for i in range( 1, self.steps+1 if len(last_items) >= self.steps else len(last_items)+1 ):
                    prev_item = last_items[-i]

                    if not prev_item in rules :
                        rules[prev_item] = dict()

                    if not item_id in rules[prev_item]:
                        rules[prev_item][item_id] = 0

                    rules[prev_item][item_id] += getattr(self, self.weighting)( i )

            last_items.append(item_id)


        if self.pruning > 0 :
            self.prune( rules )

        self.rules = rules

    def linear(self, i):
        return 1 - (0.1*i) if i <= 100 else 0

    def same(self, i):
        return 1

    def div(self, i):
        return 1/i

    def log(self, i):
        return 1/(log10(i+1.7))

    def quadratic(self, i):
        return 1/(i*i)

    def predict_next(self, session_id, input_item_id, item_set, predict_for_item_ids, skip=False, type='view', timestamp=0):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.

        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.

        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.

        '''
        if session_id != self.session:
            self.session_items = []
            self.session = session_id

        if type == 'view':
            self.session_items.append( input_item_id )

        if skip:
            return

        self.session_item_map.update({session_id : set(item_set)})
        current_session_embedding = self.content_aggregator(self.content,item_set)
        current_embedding = self.content[input_item_id,:]

        preds = np.zeros( len(predict_for_item_ids))
        if input_item_id in self.rules:
            for key in self.rules[input_item_id]:
                item_embedding = self.content[key,:]
                w1 = self.cos_Dis_sim(item_embedding,current_embedding)
                w2 = self.cos_Dis_sim(item_embedding,current_session_embedding)
                if (self.d == "D"):
                    preds[ predict_for_item_ids == key ] = self.rules[input_item_id][key]*w1
                elif (self.d == "I"):
                    preds[ predict_for_item_ids == key ] = self.rules[input_item_id][key]*w2
                elif (self.d == "ID"):
                    preds[ predict_for_item_ids == key ] = self.rules[input_item_id][key]*w1*w2
                else:
                    preds[ predict_for_item_ids == key ] = self.rules[input_item_id][key]

        series = pd.Series(data=preds, index=predict_for_item_ids)

        series = series / series.max()

        return series

    def prune(self, rules):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
        Parameters
            --------
            rules : dict of dicts
                The rules mined from the training data
        '''
        for k1 in rules:
            tmp = rules[k1]
            if self.pruning < 1:
                keep = len(tmp) - int( len(tmp) * self.pruning )
            elif self.pruning >= 1:
                keep = self.pruning
            counter = col.Counter( tmp )
            rules[k1] = dict()
            for k2, v in counter.most_common( keep ):
                rules[k1][k2] = v


    def items_for_session(self, session):
        '''
        Returns all items in the session

        Parameters
        --------
        session: Id of a session

        Returns
        --------
        out : set
        '''
        return self.session_item_map.get(session);

    def content_aggregator (self, content, indexes):
        '''
        content: input from article emdeddings containing labelencoders, dataframe and embeddings
        indexes: article ids in the session
        '''
#        content_df = content[1]
        embeddings = content[indexes]
        aggregate = np.mean(embeddings, axis=0)

        return aggregate

    def vector_variance (self, content, indexes):
        '''
        content: input from article emdeddings containing labelencoders, dataframe and embeddings
        indexes: article ids in the session
        '''
#        content_df = content[1]
        diags_cov_var_matrix=np.cov(np.transpose(content[indexes])).diagonal()
        return diags_cov_var_matrix.sum()

    def cos_sim(self,a,b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return 1-(1-(dot_product / (norm_a * norm_b)))/2

    def cos_Dis_sim(self,a,b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return (1-(dot_product / (norm_a * norm_b)))/2

    def diversity_of_session(self, session):
        '''
        Returns artist diversity of a session

        Parameters
        --------
        session: Id of a session

        Returns
        --------
        out : set
        '''
        session_items = list(self.items_for_session(session))
        dis = 0.0
        pairs = 0.0
        for i in range(len(session_items)):
            contentA = self.content[ session_items[i],: ]
            for j in range(i+1,len(session_items)):
                contentB = self.content[ session_items[j],:]
                dis += self.cos_Dis_sim(contentA,contentB)
                pairs += 1.0
        diversity = dis / pairs if pairs > 0 else 0
        return diversity
