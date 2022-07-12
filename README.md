# tag-ur-it

**Deep Learning** project, part of a data science course under Naya College.

*The goal* is to be able to tag anime artwork using [booru-style tags](https://safebooru.donmai.us/tags) - a *multi label* classification problem.  

*The model* is based on [efficient net b4](https://arxiv.org/abs/1905.11946) (V2 wasn't available for me at the time),  
and is derived from previous work by [RF5](https://github.com/RF5/danbooru-pretrained) and [anthony-dipofi](https://github.com/anthony-dipofi/danbooru-tagger).

*The data* used for training was taken from the [Great Danbooru Dataset](https://www.gwern.net/Danbooru2021), specifically - the [safe 512px dataset small subset](https://www.kaggle.com/datasets/muoncollider/danbooru2020small).

The dataprep process can be seen in the files, but in short I:
- removed all non-tag related metadata, the non 'general' labeled tags, and removed data for images not marked as 'safe'.
- ranked the tags by usage, took those with 10K or above, and removed the rest from the metadata.
- deleted any image (both file and data) which had below 5 or above 20 tags, and 'old' images (label below 1mil) - which left me with 11K images for training, 1.2K for validation, and another 1.2K for testing.

Training was done in two parts:
1. Full model training for 100 epochs (~30hrs) with a batch size of 32, exposing it to around 1K images for training and 100 for validation.
2. Freezing the effnet base, increasing the batch size to 256 and exposing the model to the entire set. ran for ~30 hours (17 epochs).

Training and testing can be seen in the files as well, and a sample is provided for illustration purposes but is not representative of the full process.

The tagger notebook presents the model alongside (very insufficent) results and metrics for the process.

The entire thing is extremely bare-bones since it's a hand-in project.  
I do plan on expanding it to include much, much more as I venture forward into the anime-deep-learning space.
