import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import utils
import Learner

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def main():
    args = utils.get_args()
    Learner.Trainer_RDVC(args).train()
    return 0


if __name__ == "__main__":
    main()
