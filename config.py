#!/usr/bin/env python
# -*- encoding: utf-8 -*-


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