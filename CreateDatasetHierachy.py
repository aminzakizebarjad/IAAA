from pathlib import Path
import pandas
import glob
import pydicom
import shutil
import tqdm
import random
import shutil

TrainPath = Path('datasets/IAAA/train')
ValPath = Path('datasets/IAAA/val')
if TrainPath.exists():
    shutil.rmtree(TrainPath)
if ValPath.exists():
    shutil.rmtree(ValPath)

Path.mkdir(TrainPath/'normal', parents=True)
Path.mkdir(TrainPath/'abnormal', parents=True)
Path.mkdir(ValPath/'normal', parents=True)
Path.mkdir(ValPath/'abnormal', parents=True)


Labels = pandas.read_csv(filepath_or_buffer='data/labels.csv')
print(Labels)
# print(type(Labels.columns[0]))

# bar = tqdm.tqdm()
for img_path in tqdm.tqdm(glob.iglob('data/images/*.dcm')):
    img = pydicom.dcmread(img_path)
    matchRow = Labels[img.SOPInstanceUID == Labels.SOPInstanceUID]
    # print(matchRow)
    lbl = matchRow.loc[matchRow.index[0], 'Label']
    # print(lbl)
    if random.random()< .1:
        if lbl == 'normal':
            shutil.copyfile(Path(img_path).as_posix(), ValPath / 'normal' / Path(img_path).name)
            # print(Path(img_path).as_posix())
        else:
            shutil.copyfile(Path(img_path).as_posix(), ValPath / 'abnormal' / Path(img_path).name)
    else:
        if lbl == 'normal':
            shutil.copyfile(Path(img_path).as_posix(), TrainPath/'normal'/Path(img_path).name)
            # print(Path(img_path).as_posix())
        else:
            shutil.copyfile(Path(img_path).as_posix(), TrainPath/'abnormal'/Path(img_path).name)