"""
References:
Jannach, Dietmar, and Malte Ludewig. "When recurrent neural networks meet the neighborhood for session-based recommendation." Proceedings of the Eleventh ACM Conference on Recommender Systems. 2017.
Hidasi, Balázs, et al. "Session-based recommendations with recurrent neural networks." arXiv preprint arXiv:1511.06939 (2015).
"""

import numpy as np
import math

class Precision: 
    
    def __init__(self, length=20):
        self.length = length;
    
    def init(self, content):
        return
        
    def reset(self):
        '''
        Desc: Reset for usage in multiple evaluations
        '''
        self.test=0;
        self.hit=0
    
    def add_set(self, result, next_items, for_item=0, session=0):
        '''
        Desc: Update the metric with predictions and the correct next items.
        
        Input
        --------
        result: pandas.Series of scores with the item id as the index
        '''
        self.test += 1
        self.hit += len( set(next_items) & set(result[:self.length].index) ) / self.length
        
    def result(self):
        '''
        Desc: Return a tuple of a description string and the current averaged value
        '''
        return ("Precision@" + str(self.length) + ": "), (self.hit/self.test)
    
class Recall: 
    
    def __init__(self, length=20):
        self.length = length;
    
#    def init(self, train):
    def init(self, content):

        '''
        Do initialization work here.
        
        Parameters
        --------
        train: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        '''
        return
        
    def reset(self):
        '''
        Desc: Reset for usage in multiple evaluations
        '''
        self.test=0;
        self.hit=0 
       
    def add_set(self, result, next_items, for_item=0, session=0):
        '''
        Desc: Update the metric with predictions and the correct next items.
        
        Input
        --------
        result: pandas.Series of scores with the item id as the index
        '''
        self.test += 1
        self.hit += len( set(next_items) & set(result[:self.length].index) ) / len(next_items)
        
    def result(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        return ("Recall@" + str(self.length) + ": "), (self.hit/self.test)

class Diversity:
    '''
    ArtistCoherence( length=20 )

    Used to iteratively calculate the artist diversity of music recommendation lists of an algorithm. 

    Parameters
    -----------
    length : int
        Coverage@length
    '''    
    def __init__(self, length=20):
        self.length = length
        self.average = 0.0
        self.count = 0.0
        
    def init(self, content):
        '''
        Do initialization work here.
        
        Parameters
        --------
        train: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        '''
                        
        self.content = content
        
        
    def reset(self):
        '''
        Reset for usage in multiple evaluations
        '''
        self.average = 0.0
        self.count = 0.0
        
    def skip(self, for_item = 0, session = -1 ):
        pass

    def add_set(self, result, next_item, for_item=0, session=0, pop_bin=None, position=None):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        recs = result[:self.length]
        
        dis = 0.0
        pairs = 0.0
        
        for i in range(len(recs)):
            contentA = self.content[ recs.index[i],: ]
            for j in range(i+1,len(recs)):
                contentB = self.content[ recs.index[j],:]
                dis += self.cos_Dis_sim(contentA,contentB)
                pairs += 1.0
        
        self.average += dis / pairs if pairs > 0 else 0
        self.count += 1
        
    def result(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        return ("Diversity@" + str(self.length) + ": "), ( self.average / self.count )
    
    def cos_Dis_sim(self,a,b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return (1-(dot_product / (norm_a * norm_b)))/2
    

class DiversityRankRelavance:
    '''
    ArtistCoherence( length=20 )

    Used to iteratively calculate the artist diversity of music recommendation lists of an algorithm. 

    Parameters
    -----------
    length : int
        Coverage@length
    '''
    
    def __init__(self, length=20):
        self.length = length
        self.average = 0.0
        self.count = 0.0
        
    def init(self,content):
        '''
        Do initialization work here.
        
        Parameters
        --------
        train: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        '''
                
        self.content = content
        
        
    def reset(self):
        '''
        Reset for usage in multiple evaluations
        '''
        self.average = 0.0
        self.count = 0.0
        
    def skip(self, for_item = 0, session = -1 ):
        pass
        
    def add_set(self, result, next_item, for_item=0, session=0, pop_bin=None, position=None):
        '''
        Update the metric with a result set and the correct next item.
        Result must be sorted correctly.
        
        Parameters
        --------
        result: pandas.Series
            Series of scores with the item id as the index
        '''
        recs = result[:self.length]
        negative_sample = 0.01
        
        avg_dists = []
        disc_weights = []
        for i in range(len(recs)-1):
            dists = []
            weights = []
            contentA = self.content[ recs.index[i],: ]
            for j in range(i+1,len(recs)):
                #Ignoring self-similarity
#                dist = 0.01
                if j == i:
                    continue
                contentB = self.content[ recs.index[j],: ]
                dist = self.cos_Dis_sim(contentA,contentB)
                relevance_j = 1 if recs.index[j] in next_item else negative_sample
                rel_discount = self.log_rank_discount(max(0, j-i-1))
                dists.append(dist * rel_discount * relevance_j)
                weights.append(rel_discount * relevance_j)
#                print(weights,sum(weights),float(sum(weights)))
#            print (i,dists,weights)
            avg_dists_i = sum(dists)/float(sum(weights))

            #Weights item by relevance
            relevance_i = 1 if recs.index[i] in next_item else negative_sample

            #Logarithmic rank discount, to prioritize more diverse items in the top of the list
            rank_discount_i = self.log_rank_discount(i)
            avg_dists.append(avg_dists_i * rank_discount_i * relevance_i)
            disc_weights.append(rank_discount_i)
                

        #Expected Intra-List Diversity (EILD) with logarithmic rank discount
        #From "Incorporating Diversity in a Learning to Rank Recommender System" (2016)
        avg_cos_dist = sum(avg_dists) / float(sum(disc_weights))
 
        self.average += avg_cos_dist
        self.count += 1
        

    def result(self):
        '''
        Return a tuple of a description string and the current averaged value
        '''
        return ("DiversityRR@" + str(self.length) + ": "), ( self.average / self.count )
    
    def log_rank_discount(self,k):
        return 1./math.log2(k+2)
    
    
    def cos_Dis_sim(self,a,b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return (1-(dot_product / (norm_a * norm_b)))/2