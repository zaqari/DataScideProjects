import torch
import pandas as pd
import torch.nn as nn
import numpy as np


def square_combine(current, update, dfC, dfU, update_columns):
    updated_df = pd.concat([dfC, dfU], ignore_index=True)
    dimX1 = update.shape[1]
    dimX2 = current.shape[1]

    new1 = torch.cat([current, torch.zeros(size=(current.shape[0],dimX1))], dim=-1)
    new1 = torch.cat([new1, torch.zeros(size=(update.shape[0],new1.shape[1]))], dim=0)

    new2 = torch.cat([torch.zeros(size=(update.shape[0], dimX2)), update], dim=-1)
    new2 = torch.cat([torch.zeros(size=(current.shape[0], new2.shape[1])),new2], dim=0)

    new = new1 + new2
    return new, updated_df


