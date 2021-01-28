"""
References:
Malte Ludewig and Dietmar Jannach. 2018. Evaluation of session-based recommendation algorithms.User Modeling and User-Adapted Interaction28,4-5 (2018), 331–390.
"""

from _operator import itemgetter
from math import sqrt
import random
import time
import numpy as np
import pandas as pd



class ContextKNN:
    '''
    ContextKNN( k, sample_size=500, sampling='recent',  similarity = 'jaccard', remind=False, pop_boost=0, session_key = 'SessionId', item_key= 'ItemId')

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
    remind : bool
        Should the last items of the current session be boosted to the top as reminders
    pop_boost : int
        Push popular items in the neighbor sessions by this factor. (default: 0 to leave out)
    extend : bool
        Add evaluated sessions to the maps
    normalize : bool
        Normalize the scores in the end
    session_key : string
        Header of the session ID column in the input file. (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file. (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file. (default: 'Time')
    '''

    def __init__( self, k, sample_size=1000, sampling='recent',  similarity = 'jaccard', content_similarity = 'jaccard', remind=False, pop_boost=0, extend=False, normalize=True, session_key = 'SessionId', item_key= 'ItemId', time_key= 'Time'):

        self.remind = remind
        self.k = k
        self.sample_size = sample_size
        self.sampling = sampling
        self.similarity = similarity
        self.content_similarity = content_similarity
        self.pop_boost = pop_boost
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.extend = extend
        self.normalize = normalize
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()
        self.session_item_map = dict()
        self.item_session_map = dict()
        self.session_variance = dict()
        self.session_diversity_map = dict()
        self.session_time = dict()
        self.sim_time = 0

    def content_aggregator (self, content, indexes):
        '''
        Desc: Aggreage the content of articles in the session

        Input
        --------
        content: article embeddings
        indexes: index of articles in the session

        Output
        --------
        aggregated vector
        '''
        embeddings = content[indexes]
        aggregate = np.mean(embeddings, axis=0)

        return aggregate

    def fit(self, train, content, d=None):

        '''
        Desc: Trains the predictor. This is a memory-based method therefore there is no explicit parameter to learn. In this method we memorize information in the sessions.
        -------
        Input:
            data: training data (sessions, dataframe). It contains the transactions of the users (sessions). It has one column for user (session IDs), one for item IDs and one for the timestamp of the events (unix timestamps).
            content: item meta-data (numpy array)
        --------
        Output: Mappings between sessions, items and time (dicts)
        '''

        index_session = train.columns.get_loc( self.session_key )
        index_item = train.columns.get_loc( self.item_key )
        index_time = train.columns.get_loc( self.time_key )


        self.content = content
        self.d = d
        session = -1
        session_items = set()

        time = -1

        for row in train.itertuples(index=False):
            # cache items of sessions
            if row[index_session] != session:
                if len(session_items) > 0:
                    self.session_item_map.update({session : session_items})
                    self.session_diversity_map.update({session : self.diversity_of_session(session)})
                    self.session_time.update({session : time})

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
        self.session_time.update({session : time})
        self.session_item_map.update({session : session_items})
        self.session_diversity_map.update({session : self.diversity_of_session(session)})

        self.session_diversity_map = {k: self.diversity_of_session(k) for k, v in self.session_item_map.items()}

    def predict_next( self, session_id, input_item_id, item_set, predict_for_item_ids, skip=False, type='view', timestamp=0 ):
        '''
        Desc: Gives predicton scores for a selected set of items on how likely they be the next item in the session.

        Input:
        --------
        user_id: user_id that we want to predict the next items (int or hash)
        input_item_id: the last item_id of the user (int)
        item_set: items in the user (session) history (list)
        predict_for_item_ids: IDs of items for which the model should give scores. Every ID must be in the set of item IDs of the training set (an array)

        --------
        Output: Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs (pandas.Series).

        '''

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


            self.session = session_id
            self.session_items = list()
            self.relevant_sessions = set()

        if type == 'view':
            self.session_items.append( input_item_id )

        if skip:
            return

        self.session_item_map.update({session_id : set(item_set)})
        self.session_diversity_map.update({session_id : self.diversity_of_session(session_id)})

        neighbors = self.find_neighbors( set(item_set), input_item_id, session_id)
        scores = self.score_items( neighbors , item_set)

        # add some reminders
        if self.remind:

            reminderScore = 5
            takeLastN = 3

            cnt = 0
            for elem in self.session_items[-takeLastN:]:
                cnt = cnt + 1

                oldScore = scores.get( elem )
                newScore = 0
                if oldScore is None:
                    newScore = reminderScore
                else:
                    newScore = oldScore + reminderScore
                # update the score and add a small number for the position
                newScore = (newScore * reminderScore) + (cnt/100)
                scores.update({elem : newScore})

        #push popular ones
        if self.pop_boost > 0:

            pop = self.item_pop( neighbors )
            # Iterate over the item neighbors
            #print itemScores
            for key in scores:
                item_pop = pop.get(key)
                # Gives some minimal MRR boost?
                scores.update({key : (scores[key] + (self.pop_boost * item_pop))})

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
        Desc: Returns a dict (item,score) of the item popularity for the given list of sessions (only a set of ids)

        Input
        --------
        sessions: a set of sessions (neighbor sessions)

        --------
        Output : a dict with popularity scores
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

    def random(self, first, second):
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
        return random.random()


    def items_for_session(self, session):
        '''
        Desc: Returns all items in the session

        Input
        --------
        session: Id of a session

        Output
        --------
        set of items
        '''
        return self.session_item_map.get(session);


    def sessions_for_item(self, item_id):
        '''
        Desc: Returns all items in the session

        Input
        --------
        session: Id of an item

        Output
        --------
        set of sessions
        '''
        return self.item_session_map.get( item_id )

    def diversity_for_session(self, session_id):
        '''
        Desc: Returns diversity of a session

        Input
        --------
        session: Id of a session

        Output
        --------
        diversity of the session (float)
        '''
        return self.session_diversity_map.get( session_id )

    def diversity_of_session(self, session):
        '''
        Desc: Calculate the diversity of a session

        Input
        --------
        session: session_id

        Output
        --------
        diversity of the session
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

    def most_recent_sessions( self, sessions, number ):
        '''
        Desc: Find the most recent sessions in the courpus

        Input
        --------
        sessions: set of session ids
        number_of_sessions: number of session that we want to filter

        Output
        --------
        set of sessions
        '''
        sample = set()

        tuples = list()
        for session in sessions:
            time = self.session_time.get( session )
            if time is None:
                print(' EMPTY TIMESTAMP!! ', session)
            tuples.append((session, time))

        tuples = sorted(tuples, key=itemgetter(1), reverse=True)
        cnt = 0
        for element in tuples:
            cnt = cnt + 1
            if cnt > number:
                break
            sample.add( element[0] )

        return sample


    def possible_neighbor_sessions(self, session_items, input_item_id, session_id):
        '''
        Desc: Find potential neighbors. With self.sample_size of 0 it uses all sessions in which the current item of the current session appears.

        Input
        --------
        Current item id

        Output
        --------
        Set of potential neighbors
        '''


        self.relevant_sessions = self.relevant_sessions | self.sessions_for_item( input_item_id );

        if self.sample_size == 0: #use all session as possible neighbors

            print('!!!!! runnig KNN without a sample size (check config)')
            return self.relevant_sessions

        else: #sample some sessions

            self.relevant_sessions = self.relevant_sessions | self.sessions_for_item( input_item_id );

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

    def calc_similarity (self, content, session_items, sessions):
        '''
        Desc: Calculates the similarity for the items in current session_items and each neighbor session in sessions.

        Input
        --------
        session_items: set of item ids in the current session
        sessions: list of session ids
        dwelling_times:
        timestamp:

        Output
        --------
        list of tuples (session_id, vector similarity, content similarity, diversity of session_id)
        '''

        neighbors = []
        cnt = 0
        for session in sessions:
            cnt = cnt + 1
            # get items of the session, look up the cache first
            session_items_test = self.items_for_session( session )
            similarity = getattr(self , self.similarity)(session_items_test, session_items)
            if similarity > 0:
                neighbors.append((session, similarity,(self.diversity_for_session(session))))

        return neighbors

    #-----------------
    # Find a set of neighbors, returns a list of tuples (sessionid: similarity)
    #-----------------
    def find_neighbors( self, session_items, input_item_id, session_id):
        '''
        Desc: Finds the k nearest neighbors for the given session_id and the current item input_item_id.

        Input
        --------
        session_items: set of item ids for current session
        input_item_id: the current input id in the session
        session_id: current session_id
        timestamp: timestamp of session

        Output
        --------
        list of tuples (session_id, vector similarity, content similarity, diversity of neighbor session)
        '''
        possible_neighbors = self.possible_neighbor_sessions( session_items, input_item_id, session_id )
        possible_neighbors = self.calc_similarity(self.content, session_items, possible_neighbors)
        if (self.d == "D" or self.d == "ID"):
            possible_neighbors = sorted( possible_neighbors, reverse=True, key=lambda x: x[1]*(x[2]))
        else:
            possible_neighbors = sorted( possible_neighbors, reverse=True, key=lambda x: x[1])

        possible_neighbors = possible_neighbors[:self.k]

        return possible_neighbors


    def score_items(self, neighbors, current_session):
        '''
        Desc: Compute a set of scores for all items given a set of neighbors.

        Input
        --------
        neighbors: set of neighbor session ids
        current_session: current session items
        timestamp: timestamp of current session

        Output
        --------
        list of tuple (item, score)
        '''
        current_session_embedding = self.content_aggregator(self.content,list(current_session))

        # now we have the set of relevant items to make predictions
        scores = dict()
        # iterate over the sessions
        for session in neighbors:
            # get the items in this session
            items = self.items_for_session(session[0])
            diversity = session[2]
            if (self.d == "ID" or self.d == "D"):
                new_score = session[1]*diversity
            else:
                new_score = session[1]

            for item in items:
                old_score = scores.get( item )

                if old_score is None:
                    scores.update({item : ( new_score )})

                else:
                    new_score = old_score + new_score
                    scores.update({item : new_score})

        if (self.d == "ID" or self.d == "I"):
            for w, v in scores.items():
                    item_embedding = self.content[w,:]
                    scores[w] = v * (self.cos_Dis_sim(item_embedding,current_session_embedding))  # Apply a weight for diversity

        return scores


    def cos_sim(self,a,b):
        '''
        Desc: Calculate the modified (between 0 and 1) cosine similarity

        Input
        --------
        two vectors

        Output
        --------
        a float number between 0 and 1
        '''
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return ((dot_product / (norm_a * norm_b))+1)/2

    def cos_Dis_sim(self,a,b):
        '''
        Desc: Calculate the modified (between 0 and 1) cosine dissimilarity

        Input
        --------
        two vectors

        Output
        --------
        a float number between 0 and 1
        '''
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return (1-(dot_product / (norm_a * norm_b)))/2
