#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 19:05:01 2024

@author: becky
"""

import splitfolders

# path_data = "data/train_val"

# dest_img = "data_jpg/images"
# dest_mask = "data_jpg/masks"


input_folder = "cluster_data"
output = "output" #where you want the split datasets saved. one will be created if it does not exist or none is set

splitfolders.ratio(input_folder, output=output, seed=42, ratio=(.9, .1, 0))