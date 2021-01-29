# Diversification in Session-based News Recommender Systems

This page contains support material for the paper: A Gharahighehi and C Vens. “Diversification in Session-based News Recommender Systems”, submitted for the theme issue on [Intelligent Systems for Tackling Online Harms](https://www.springer.com/journal/779/updates/18096208) of the journal of [Personal and Ubiquitous Computing](https://www.springer.com/journal/779/).

This research is built on implementation by [Malte Ludewig, Noemi Mauro, Sara Latifi and Dietmar Jannach](https://rn5l.github.io/session-rec/index.html) [1]. In this paper we make rule-based and neighborhood based session-based recommenders, diversity-aware using news article embeddings.

Four datasets are used in this study:

- Adressa [2]: You can download the dataset from this [link](http://reclab.idi.ntnu.no/dataset/).
- Globo.com [3]: You can download the dataset from this [link](https://www.kaggle.com/gspmoreira/news-portal-user-interactions-by-globocom).
- Kwestie
- Roularta

The diversification approach can be set in the [main.py](https://github.com/alirezagharahi/d_SBRS/blob/main/main.py) file. For instance "D" refers to divers neighbor/rule approach.

References:

[1] Ludewig, M., Mauro, N., Latifi, S., Jannach, D. 2019. Performance comparison of neural andnon-neural approaches to session-based recommendation.  In: Proceedings of the 13thACM Conference on Recommender Systems, pp. 462–466.

[2] Jon Atle Gulla, Lemei Zhang, Peng Liu, Özlem Özgöbek, and Xiaomeng Su. 2017. The Adressa dataset for news recommendation. InProceedings of theinternational conference on web intelligence. 1042–1048.

[3] P Moreira Gabriel De Souza, Dietmar Jannach, and Adilson Marques Da Cunha. 2019. Contextual hybrid session-based news recommendation withrecurrent neural networks.IEEE Access7 (2019), 169185–169203.
