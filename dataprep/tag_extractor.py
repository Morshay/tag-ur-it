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
    }) # maybe content filtering in the future

    print('done!')
    
    del(metadata)

# fn = 'danbooru_2020_safe_tags.json'

# with open(fn, 'w') as f:
#     json.dump(tags_dict)

# with open(fn, 'r') as f:
#     tags_dict = json.load(f)

tags_df = pd.DataFrame(
    {
     'id':tags_dict.keys(),
     'tags':tags_dict.values()
    }
)

# taking only the most used tags

vc = pd.Series(
    [tag for tags in tags_dict.values() for tag in tags]
).value_counts()

top_tags = pd.Series(vc[vc >= 10000].index.rename('tag')) # 1002 total

# vc>=100 gives ~10k tags
# vc>=1000 gives 3773 tags
# vc>=100000 gives 132 tags

top_tags.to_csv('top_tags.csv', index=False)

# filtering tag data 

img_paths = Path("all-images").rglob("./*.jpg")

ids = [p.stem for p in img_paths]

tags_srs = tags_df[
    tags_df.id.isin(ids)
].set_index('id').squeeze()

filtered_tags = tags_srs.apply(
    lambda tag_list: list(
        set(tag_list).intersection(top_tags)
    )
)

# this is basically to ease on the nn

tag_nums = filtered_tags.apply(len).sort_values()
num_ids = tag_nums[(tag_nums<=20) & (tag_nums>=5)].index

final_tags = tags_srs[tags_srs.index.isin(num_ids)]
final_tags.to_csv('img_tags.csv')

# deleting images out of tag data

files_to_del = [p for p in img_paths if p.stem not in num_ids]

for file in files_to_del:
    os.remove(file)