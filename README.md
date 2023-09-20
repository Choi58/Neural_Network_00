# Neural_Network_00
-----------------------------------
I. Project title
Referring image segmentation(RIS)

II. Project introduction
Objective: RIS aims to generate segmentation mask for the target object referred by a given query expressions in natural language

Motivation: It is necessary to perform various tasks in language as well as fixed tasks in human-robot interaction and image editing

Technical limitations: existing methods does not capture the meaning of text queries and depend on specific words

III. Dataset description

images: https://cocodataset.org/#download  => 2014 Train images download
annotation: https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip
------------------------------------
Dataset preparation

1) Put the folder of COCO training set ("train2014") under data/images/.

2) Download the RefCOCO dataset and extract them to data/. Then run the script for data preparation under data/:

3) run: python data/data_process_v2.py --data_root . --output_dir data_v2 --dataset refcoco --split unc --generate_mask
