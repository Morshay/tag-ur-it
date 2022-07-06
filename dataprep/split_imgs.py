import shutil
from pathlib import Path

files = [f for f in Path('all-images').rglob('./*.jpg')]

val_files = [files[i] for i in range(0, len(files), 10)]
for f in val_files:
    shutil.move(f, Path('val') / f.name)

test_files = [files[i] for i in range(1, len(files), 10)]
for f in test_files:
    shutil.move(f, Path('test') / f.name)

train_files = list(set(files).difference(set(val_files + test_files)))
for f in train_files:
    shutil.move(f, Path('train') / f.name)