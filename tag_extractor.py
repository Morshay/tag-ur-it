# this script assumes you have a folder with danbooru data
# structured in the same manner as the 2020 safe dataset on kaggle

import json
import pandas as pd
from pathlib import Path
import os

tags_dict = {}

for file in Path('metadata').rglob('*'):

    print(f'reading {file.name}}}...', end=' ')

    metadata = [
        json.loads(line)
        for line
        in open(
            str(file),
            'r', encoding='utf-8'
        )
    ]

    print('done!')
    
    print('extracting tags from data...', end=' ')

    tag_dict.update({
        int(entry['id']): [tag['name']
                           for tag
                           in entry['tags']
                           if tag['category'] == '0']
        for entry
        in metadata
        if entry['rating'] == 's'
    })

    print('done!')
    
    del(metadata)

tags_df = pd.DataFrame(
    {
     'id':tags_dict.keys(),
     'tags':tags_dict.values()
    }
)

# deleting files that aren't in tags data

img_paths = Path("images").rglob("./*.jpg")

ids = [p.name.split('.')[0] for p in img_paths]

ids_to_del = set(ids).difference(set(tags_df.id))

files_to_del = [p for p in img_paths if p.name.split('.')[0] in ids_to_del]

for file in files_to_del:
    os.remove(file)

# taking only the most used tags

vc = pd.Series(
    [tag for tags in tags_dict.values() for tag in tags]
).value_counts()

top_tags = pd.Series(vc[vc >= 1000].index.rename('tag')) # 3773 total

top_tags.to_csv('top_tags.csv', index=False)

# filtering tag data 

tags_srs = tags_df[
    tags_df.id.isin(ids)
].set_index('id').squeeze() # remains a string to fit filenames

filtered_tags = tags_srs.apply(
    lambda tag_list: list(
        set(tag_list).intersection(top_tags)
    )
)

filtered_tags.to_csv('img_tags.csv')