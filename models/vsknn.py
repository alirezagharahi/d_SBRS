"""
References:
Malte Ludewig and Dietmar Jannach. 2018. Evaluation of session-based recommendation algorithms.User Modeling and User-Adapted Interaction28,4-5 (2018), 331–390.

Garg, Diksha, et al. "Sequence and time aware neighborhood for session-based recommendations: Stan." Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval. 2019.
"""

from _operator import itemgetter
from math import sqrt
import random
import time

from pympler import asizeof
import numpy as np
import pandas as pd
from math import log10
from datetime import datetime as dt
from datetime import timedelta as td
import math

class VMContextKNN:
    '''
    VMContextKNN( k, sample_size=1000, sampling='recent', similarity='cosine', weighting='div', dwelling_time=False, last_n_days=None, last_n_clicks=None, extend=False, weighting_score='div_score', weighting_time=False, normalize=True, session_key = 'SessionId', item_key= 'ItemId', time_key= 'Time')

    Parameters
    -----------
    k : int
        Number of neighboring session to calculate the item scores from. (Default value: 100)
    sample_size : int
        Defines the length of a subset of all training sessions to calculate the nearest neighbors from. (Default value: 500)
    sampling : string
        String to define the sampling method for sessions (recent, random). (default: recent)
    similarity : string
        String to define the method for the similarity calculation (jaccard, cosine, binary, tanimoto). (default: jaccard)
    weighting : string
        Decay function to determine the importance/weight of individual actions in the current session (linear, same, div, log, quadratic). (default: div)
    weighting_score : string
        Decay function to lower the score of candidate items from a neighboring sessions that were selected by less recently clicked items in the current session. (linear, same, div, log, quadratic). (default: div_score)
    weighting_time : boolean
        Experimental function to give less weight to items from older sessions (default: False)
    dwelling_time : boolean
        Experimental function to use the dwelling time for item view actions as a weight in the similarity calculation. (default: False)
    last_n_days : int
        Use only data from the last N days. (default: None)
    last_n_clicks : int
        Use only the last N clicks of the current session when recommending. (default: None)
    extend : bool
        Add evaluated sessions to the maps.
    normalize : bool
        Normalize the scores in the end.
    session_key : string
        Header of the session ID column in the input file. (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file. (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file. (default: 'Time')
    '''

    def __init__( self, k, sample_size=1000, sampling='recent', similarity='cosine', weighting='div', dwelling_time=False, last_n_days=None, last_n_clicks=None, extend=False, weighting_score='div_score', weighting_time=False, normalize=True, session_key = 'SessionId', item_key= 'ItemId', time_key= 'Time'):

        self.k = k
        self.sample_size = sample_size
        self.sampling = sampling
        self.weighting = weighting
        self.dwelling_time = dwelling_time
        self.weighting_score = weighting_score
        self.weighting_time = weighting_time
        self.similarity = similarity
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.extend = extend
        self.normalize = normalize
        self.last_n_days = last_n_days
        self.last_n_clicks = last_n_clicks

        #updated while recommending
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()

        # cache relations once at startup
        self.session_item_map = dict()
        self.item_session_map = dict()
        self.session_time = dict()

         ## Added lines
        self.session_diversity = dict()
        # content vectors
        self.content = np.array
        ## Added lines

        self.min_time = -1

        self.sim_time = 0

    def content_aggregator (self, content, indexes):
        '''
        content: input from article emdeddings containing labelencoders, dataframe and embeddings
        indexes: article ids in the session
        '''
#        content_df = content[1]
        embeddings = content[indexes]
        aggregate = np.mean(embeddings, axis=0)

        return aggregate


    def content_Weighted_aggregator (self, content, indexes, map):
        '''
        content: input from article emdeddings containing labelencoders, dataframe and embeddings
        indexes: article ids in the session
        '''
        weight = []
        for item in map:
            weight.append(map[item])
        embeddings = content[indexes]
        aggregate = np.average(embeddings,weights=weight, axis=0)

        return aggregate

    def fit(self, data, content, d=None):
        '''
        Trains the predictor.

        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).

        '''

        if self.last_n_days != None:

            max_time = dt.fromtimestamp( data[self.time_key].max() )
            date_threshold = max_time.date() - td( self.last_n_days )
            stamp = dt.combine(date_threshold, dt.min.time()).timestamp()
            train = data[ data[self.time_key] >= stamp ]

        else:
            train = data

        self.num_items = train[self.item_key].max()

        index_session = train.columns.get_loc( self.session_key )
        index_item = train.columns.get_loc( self.item_key )
        index_time = train.columns.get_loc( self.time_key )

        self.content = content
        self.d = d
        session = -1
        session_items = set()
        time = -1
        #cnt = 0
        for row in train.itertuples(index=False):
            # cache items of sessions
            if row[index_session] != session:
                if len(session_items) > 0:
                    self.session_item_map.update({session : session_items})
                    self.session_diversity.update({session: self.diversity_of_session(session)})
                    # cache the last time stamp of the session

                    self.session_time.update({session : time})
                    if time < self.min_time:
                        self.min_time = time

                session = row[index_session]
                session_items = set()

            time = row[index_time]
            session_items.add(row[index_item])


            # cache sessions involving an item
            map_is = self.item_session_map.get( row[index_item] )
            if map_is is None:
                map_is = set()
                self.item_session_map.update({row[index_item] : map_is})
            map_is.add(row[index_session])



        # Add the last tuple
        self.session_item_map.update({session : session_items})
        self.session_time.update({session : time})
        self.session_diversity.update({session: self.diversity_of_session(session)})


    def predict_next( self,session_id, input_item_id,item_set, predict_for_item_ids, skip=False, type='view', timestamp=0):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.

        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event. Must be in the set of item IDs of the training set.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.

        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.

        '''

#         gc.collect()
#         process = psutil.Process(os.getpid())
#         print( 'cknn.predict_next: ', process.memory_info().rss, ' memory used')

        if( self.session != session_id ): #new session

            if( self.extend ):
                item_set = set( self.session_items )
                self.session_item_map[self.session] = item_set;
                for item in item_set:
                    map_is = self.item_session_map.get( item )
                    if map_is is None:
                        map_is = set()
                        self.item_session_map.update({item : map_is})
                    map_is.add(self.session)

                ts = time.time()
                self.session_time.update({self.session : ts})

            self.last_ts = -1
            self.session = session_id
            self.session_items = list()
            self.dwelling_times = list()
            self.relevant_sessions = set()

        if type == 'view':
            self.session_items.append( input_item_id )
            if self.dwelling_time:
                if self.last_ts > 0:
                    self.dwelling_times.append( timestamp - self.last_ts )
                self.last_ts = timestamp

        if skip:
            return

        self.session_item_map.update({session_id : set(item_set)})
        current_session_diversity = self.diversity_of_session(session_id)
        self.session_diversity.update({session_id : current_session_diversity})

        items = self.session_items if self.last_n_clicks is None else self.session_items[-self.last_n_clicks:]
        neighbors = self.find_neighbors(items, input_item_id, session_id, self.dwelling_times, timestamp )
        scores = self.score_items( neighbors, items, timestamp )

        # Create things in the format ..
        predictions = np.zeros(len(predict_for_item_ids))
        mask = np.in1d( predict_for_item_ids, list(scores.keys()) )

        items = predict_for_item_ids[mask]
        values = [scores[x] for x in items]
        predictions[mask] = values
        series = pd.Series(data=predictions, index=predict_for_item_ids)

        if self.normalize:
            series = series / series.max()

        return series

    def item_pop(self, sessions):
        '''
        Returns a dict(item,score) of the item popularity for the given list of sessions (only a set of ids)

        Parameters
        --------
        sessions: set

        Returns
        --------
        out : dict
        '''
        result = dict()
        max_pop = 0
        for session, weight in sessions:
            items = self.items_for_session( session )
            for item in items:

                count = result.get(item)
                if count is None:
                    result.update({item: 1})
                else:
                    result.update({item: count + 1})

                if( result.get(item) > max_pop ):
                    max_pop =  result.get(item)

        for key in result:
            result.update({key: ( result[key] / max_pop )})

        return result

    def jaccard(self, first, second):
        '''
        Calculates the jaccard index for two sessions

        Parameters
        --------
        first: Id of a session
        second: Id of a session

        Returns
        --------
        out : float value
        '''
        sc = time.clock()
        intersection = len(first & second)
        union = len(first | second )
        res = intersection / union

        self.sim_time += (time.clock() - sc)

        return res

    def cosine(self, first, second):
        '''
        Calculates the cosine similarity for two sessions

        Parameters
        --------
        first: Id of a session
        second: Id of a session

        Returns
        --------
        out : float value
        '''
        li = len(first&second)
        la = len(first)
        lb = len(second)
        result = li / sqrt(la) * sqrt(lb)

        return result

    def tanimoto(self, first, second):
        '''
        Calculates the cosine tanimoto similarity for two sessions

        Parameters
        --------
        first: Id of a session
        second: Id of a session

        Returns
        --------
        out : float value
        '''
        li = len(first&second)
        la = len(first)
        lb = len(second)
        result = li / ( la + lb -li )

        return result

    def binary(self, first, second):
        '''
        Calculates the ? for 2 sessions

        Parameters
        --------
        first: Id of a session
        second: Id of a session

        Returns
        --------
        out : float value
        '''
        a = len(first&second)
        b = len(first)
        c = len(second)

        result = (2 * a) / ((2 * a) + b + c)

        return result

    def vec(self, first, second, map):
        '''
        Calculates the ? for 2 sessions

        Parameters
        --------
        first: Id of a session
        second: Id of a session

        Returns
        --------
        out : float value
        '''
        a = first & second
        sum = 0
        for i in a:
            sum += map[i]
        result = sum / len(map)

        return result

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

    def vec_for_session(self, session):
        '''
        Returns all items in the session

        Parameters
        --------
        session: Id of a session

        Returns
        --------
        out : set
        '''
        return self.session_vec_map.get(session);

    def sessions_for_item(self, item_id):
        '''
        Returns all session for an item

        Parameters
        --------
        item: Id of the item session

        Returns
        --------
        out : set
        '''
        return self.item_session_map.get( item_id ) if item_id in self.item_session_map else set()

    def diversity_session(self, session):
        '''
        Returns artist diversity of a session

        Parameters
        --------
        session: Id of a session

        Returns
        --------
        out : set
        '''
        return self.session_diversity.get(session);


    def most_recent_sessions( self, sessions, number ):
        '''
        Find the most recent sessions in the given set

        Parameters
        --------
        sessions: set of session ids

        Returns
        --------
        out : set
        '''
        sample = set()

        tuples = list()
        for session in sessions:
            time = self.session_time.get( session )
            if time is None:
                print(' EMPTY TIMESTAMP!! ', session)
            tuples.append((session, time))

        tuples = sorted(tuples, key=itemgetter(1), reverse=True)
        #print 'sorted list ', sortedList
        cnt = 0
        for element in tuples:
            cnt = cnt + 1
            if cnt > number:
                break
            sample.add( element[0] )
        #print 'returning sample of size ', len(sample)
        return sample


    def possible_neighbor_sessions(self, session_items, input_item_id, session_id):
        '''
        Find a set of session to later on find neighbors in.
        A self.sample_size of 0 uses all sessions in which any item of the current session appears.
        self.sampling can be performed with the options "recent" or "random".
        "recent" selects the self.sample_size most recent sessions while "random" just choses randomly.

        Parameters
        --------
        sessions: set of session ids

        Returns
        --------
        out : set
        '''

        self.relevant_sessions = self.relevant_sessions | self.sessions_for_item( input_item_id )

        if self.sample_size == 0: #use all session as possible neighbors

            print('!!!!! runnig KNN without a sample size (check config)')
            return self.relevant_sessions

        else: #sample some sessions

            if len(self.relevant_sessions) > self.sample_size:

                if self.sampling == 'recent':
                    sample = self.most_recent_sessions( self.relevant_sessions, self.sample_size )
                elif self.sampling == 'random':
                    sample = random.sample( self.relevant_sessions, self.sample_size )
                else:
                    sample = self.relevant_sessions[:self.sample_size]

                return sample
            else:
                return self.relevant_sessions

    def calc_similarity(self, content, session_items, sessions, dwelling_times, timestamp ):
        '''
        Calculates the configured similarity for the items in session_items and each session in sessions.

        Parameters
        --------
        session_items: set of item ids
        sessions: list of session ids

        Returns
        --------
        out : list of tuple (session_id,similarity)
        '''

        pos_map = {}
        length = len( session_items )

        count = 1

        for item in session_items:
            if self.weighting is not None:
                pos_map[item] = getattr(self, self.weighting)( count, length )
                count += 1
            else:
                pos_map[item] = 1

        if self.dwelling_time:
            dt = dwelling_times.copy()
            dt.append(0)
            dt = pd.Series(dt, index=session_items)
            dt = dt / dt.max()
            dt[session_items[-1]] = 1
            for i in range(len(dt)):
                pos_map[session_items[i]] *= dt.iloc[i]

        items = set(session_items)
        neighbors = []
        cnt = 0

        for session in sessions:
            cnt = cnt + 1
            # get items of the session, look up the cache first
            n_items = self.items_for_session( session )
            sts = self.session_time[session]

            similarity = self.vec(items, n_items, pos_map)
            if similarity > 0:

                if self.weighting_time:
                    diff = timestamp - sts
                    days = round( diff/ 60 / 60 / 24 )
                    decay = pow( 7/8, days )
                    similarity *= decay

                neighbors.append((session, similarity,self.diversity_session(session),))

        return neighbors


    #-----------------
    # Find a set of neighbors, returns a list of tuples (sessionid: similarity)
    #-----------------
    def find_neighbors( self, session_items, input_item_id, session_id, dwelling_times, timestamp ):
        '''
        Finds the k nearest neighbors for the given session_id and the current item input_item_id.

        Parameters
        --------
        session_items: set of item ids
        input_item_id: int
        session_id: int

        Returns
        --------
        out : list of tuple (session_id, similarity)
        '''
        possible_neighbors = self.possible_neighbor_sessions( session_items, input_item_id, session_id )
        # I ADDED DIVERSITY = TRUE
        possible_neighbors = self.calc_similarity(self.content, session_items, possible_neighbors, dwelling_times, timestamp)

        if (self.d == "D" or self.d == "ID"):
            possible_neighbors = sorted( possible_neighbors, reverse=True, key=lambda x: x[1]*(x[2]))
        else:
            possible_neighbors = sorted( possible_neighbors, reverse=True, key=lambda x: x[1])

        possible_neighbors = possible_neighbors[:self.k]

        return possible_neighbors


    def score_items(self, neighbors, current_session, timestamp):
        '''
        Compute a set of scores for all items given a set of neighbors.

        Parameters
        --------
        neighbors: set of session ids

        Returns
        --------
        out : list of tuple (item, score)
        '''
        # Added by me

        current_session_embedding = self.content_aggregator(self.content,list(current_session))
        # End of added part

        # now we have the set of relevant items to make predictions
        scores = dict()
        # iterate over the sessions
        for session in neighbors:
            # get the items in this session
            items = self.items_for_session( session[0] )

            step = 1

            for item in reversed( current_session ):
                if item in items:
                    decay = getattr(self, self.weighting_score)( step )
                    break
                step += 1

            diversity = session[2]
            if (self.d == "ID" or self.d == "D"):
                new_score = session[1]*diversity
            else:
                new_score = session[1]

            for item in items:
                old_score = scores.get( item )

                if old_score is None:
                    scores.update({item : ( new_score * decay ) })

                else:
                    new_score = old_score + ( new_score * decay )

                    scores.update({item : new_score})
        if (self.d == "ID" or self.d == "I"):
            for w, v in scores.items():
                    item_embedding = self.content[w,:]
                    scores[w] = v * (self.cos_Dis_sim(item_embedding,current_session_embedding))  # Apply a weight for diversity

        return scores

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


    def linear_score(self, i):
        return 1 - (0.1*i) if i <= 100 else 0

    def same_score(self, i):
        return 1

    def div_score(self, i):
        return 1/i

    def log_score(self, i):
        return 1/(log10(i+1.7))

    def quadratic_score(self, i):
        return 1/(i*i)

    def linear(self, i, length):
        return 1 - (0.1*(length-i)) if i <= 10 else 0

    def same(self, i, length):
        return 1

    def div(self, i, length):
        return i/length

    def log(self, i, length):
        return 1/(log10((length-i)+1.7))

    def quadratic(self, i, length):
        return (i/length)**2

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
