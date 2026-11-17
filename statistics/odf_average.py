#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt

def plot_dod(*args, sym_group:str="orthorhombic"):
    ax = plt.subplot(*args)
    return 