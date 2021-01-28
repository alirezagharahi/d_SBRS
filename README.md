# Diversification in Session-based News Recommender Systems

This page contains support material for the paper: A Gharahighehi and C Vens. “Diversification in Session-based News Recommender Systems”, under review in Personal and Ubiquitous Computing.

This research is built on implementation by [Malte Ludewig and Dietmar Jannach](https://rn5l.github.io/session-rec/index.html) [1]. In this paper We make rule-based and neighborhood based session-based recommenders, diversity-aware using news article embeddings.

Four datasets are used in this study:

- Adressa [2]: You can download the dataset from this [link](http://reclab.idi.ntnu.no/dataset/).
- Globo.com [3]: You can download the dataset from this [link](https://www.kaggle.com/gspmoreira/news-portal-user-interactions-by-globocom).
- Kwestie
- Roularta

The SKNN_D scenario can be tested by using the "diversity" variable in "score_items" function in "cknn.py" file. The SKNN_C scenario can be tested by using "cos_Dis_sim" in the "score_items" function in "cknn.py" file.

References:

[1] Malte Ludewig and Dietmar Jannach. 2018. Evaluation of session-based recommendation algorithms.User Modeling and User-Adapted Interaction28,4-5 (2018), 331–390.

[2] Jon Atle Gulla, Lemei Zhang, Peng Liu, Özlem Özgöbek, and Xiaomeng Su. 2017. The Adressa dataset for news recommendation. InProceedings of theinternational conference on web intelligence. 1042–1048.

[3] P Moreira Gabriel De Souza, Dietmar Jannach, and Adilson Marques Da Cunha. 2019. Contextual hybrid session-based news recommendation withrecurrent neural networks.IEEE Access7 (2019), 169185–169203.
