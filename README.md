# Neural_Network_00

Dataset preparation

1) Put the folder of COCO training set ("train2014") under data/images/.

2) Download the RefCOCO dataset and extract them to data/. Then run the script for data preparation under data/:

3) run: python data/data_process_v2.py --data_root . --output_dir data_v2 --dataset refcoco --split unc --generate_mask
