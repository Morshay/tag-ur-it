# tag-ur-it

**Deep Learning** project, part of a data science course under Naya College.

*The goal* is to be able to tag anime artwork using [booru-style tags](https://safebooru.donmai.us/tags).  

*The model* is based on [efficient net b4](https://arxiv.org/abs/1905.11946) (V2 wasn't available for me at the time) and is derived from previous work by [RF5](https://github.com/RF5/danbooru-pretrained) and [anthony-dipofi](https://github.com/anthony-dipofi/danbooru-tagger).

*The data* used for training was taken from the [Great Danbooru Dataset](https://www.gwern.net/Danbooru2021), more specifically - the [safe 512px dataset small subset](https://www.kaggle.com/datasets/muoncollider/danbooru2020small).

The dataprep process can be seen in the files, but in short:
- I removed all non-tag related metadata, the non 'general' labeled tags, and removed data for images not marked as 'safe'.
- I ranked the tags by usage, took those with 10K or above, and removed the rest from the metadata.
- I removed any image (both file and data) which had below 5 or above 20 tags - which left me with 10K images for training, 10 for validation, and 

I trained the entire model for 100 epochs using around 1000 train images and 100 val images. later on I exposed the model to the rest of the data (in total 10K train and 1K val), and froze the efficientnet layers with the state dict up to that moment.

The entire thing is extremely bare-bones since it's a hand-in project. i do plan on expanding it to include much, much more as i venture forward into the anime-deep-learning space.