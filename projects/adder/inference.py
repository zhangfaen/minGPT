import os
import sys

# after python 3.3, relative import is not allowed, below lines setup ../../ as one module search path
# getting the name of the directory where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
# Getting the parent directory name where the current directory is present.
current = os.path.dirname(os.path.dirname(current))
# adding the parent directory to the sys.path.
sys.path.append(current)
 
# importing
import json
import torch

from mingpt.model import GPT
from mingpt.utils import set_seed, setup_logging, CfgNode as CN
from adder import AdditionDataset

def get_config():
    C = CN()
    C.ndigit = 3
    # system
    C.system = CN()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-nano'
    C.model.vocab_size = 10
    C.model.block_size = 3 * 3 + 1 - 1

    return C

if __name__ == "__main__":
    state_dict_path = current + "/out/adder/model-3digits-20230524.pt"

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])

    print(config)
    model = GPT(config.model)

    model.load_state_dict(torch.load(state_dict_path))
    # switch from trainning mode to eval mode, so that model.transformer.drop  dropout is off
    model.eval()

    while True:
        inputs = input("Please type in 2 3-digits numbers:\n")
        if inputs == 'stop':
            break
        dix = [int(s) for s in inputs] # convert each character to its token index
        # x will be input to GPT and y will be the associated expected outputs
        # inputs to model is [batch, src_len] indexs
        dix = [dix]
        x = torch.tensor(dix, dtype=torch.long)
        outputs = model.generate(x, 4, do_sample=False)
        print(outputs)
        


