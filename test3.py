import os
import shutil
import pandas as pd

src = r'E:\Dane\input\sample_images'
dst0 = r'E:\Test'
dst1 = r'E:\Test\input'
dst2 = r'E:\Test\input\sample_images'
dst3 = r'E:\Test\input\stage1_labels.csv'

for folder in (dst0, dst1, dst2):
    if not os.path.exists(folder):
        os.mkdir(folder)

df = pd.read_csv(r'E:\Dane\input\stage1_labels.csv')
"""
cancer = df.loc[df['cancer'] == 1]
non_cancer = df.loc[df['cancer'] == 0].head(len(cancer.index))
patients = [cancer, non_cancer]
test = pd.concat(patients)
"""

test = df.head(530)
test.to_csv(dst3, index=False)
remove = False

i = 0
for pat in test.id:
    src = r'E:\Dane\input\sample_images\{}'.format(pat)
    dst = r'E:\Test\input\sample_images\{}'.format(pat)
    if not os.path.exists(dst):
        shutil.copytree(src, dst)
        print(i, 'Copying')
    else:
        print(i, 'Already in')
    if remove:
        if i >= 500:
            shutil.rmtree(r'E:\Test\input\sample_images\{}'.format(pat))
    i += 1

# src = r'E:\Dane\input\sample_images'
# dst0 = r'E:\Test2'
# dst1 = r'E:\Test2\input'
# dst2 = r'E:\Test2\input\sample_images'
# dst3 = r'E:\Test2\input\stage1_labels.csv'
#
# for folder in (dst0, dst1, dst2):
#     if not os.path.exists(folder):
#         os.mkdir(folder)
#
# df = pd.read_csv(r'E:\Dane\input\stage1_labels.csv')
# test = df.head(520)
# test.to_csv(dst3, index=False)
# i = 0
# for pat in test.id:
#     src = r'E:\Dane\input\sample_images\{}'.format(pat)
#     dst = r'E:\Test2\input\sample_images\{}'.format(pat)
#     if not os.path.exists(dst):
#         shutil.copytree(src, dst)
#         print(i, 'Copying')
#     i += 1
