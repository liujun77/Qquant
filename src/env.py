#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:53:59 2017

@author: junliu
"""

import numpy as np
import pandas as pd


class Stock:
    
    def __init__(self, filename):
        self.data = pd.read_csv(filename, index_col=0)
        self.data.index = pd.DatetimeIndex(self.data.index)
        self.data = self.data.dropna()
        
        