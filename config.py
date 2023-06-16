#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   config.py    
@Contact :   mqlqianli@sjtu.edu.cn
@License :   (C)Copyright 2021-2022, Qianli Ma

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/5/28 13:14     mql        1.0         None
'''
# config for adv_training
import os
class Config(object):

    # freelb
    adv_K = 3
    adv_lr = 1e-2
    adv_init_mag = 2e-2
    adv_max_norm = 0
    adv_norm_type = 'l2'
    base_model = 'bert'



opt = Config()

if __name__ == "__main__":
    print(opt.adv_K)