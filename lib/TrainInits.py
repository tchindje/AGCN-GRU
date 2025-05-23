import torch
import random
import numpy as np


def init_seed(seed):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def init_device(opt):
    '''
        Init device : cpu or cuda for gpu
    '''
    if torch.cuda.is_available():
        opt.cuda = True
        torch.cuda.set_device(int(opt.device[5]))
    else:
        opt.cuda = False
        opt.device = 'cpu'
    return opt

def init_optim(model, opt):
    '''
    Initialize optimizer: 

     param : model.parameters ( parameter of model we want to optimize)
     lr : leaning rate
    '''
    return torch.optim.Adam(params=model.parameters(),lr=opt.lr_init)


def init_lr_scheduler(optim, opt):
    '''
    Initialize the learning rate scheduler :  permet de modifier dynamiquement 
    le learning rate en fonction des epoques :

        milistones: list des epoques.  Doit etre croisant
        gamma : facteur multiplicatif du learning rate courant

            lr = 0.05 initiale
            milestones = [30,80] 
            gamma = 0.1

            if epoch < 30  alors lr =  0.05
            if 30 <= epoch < 80  alors lr =  0.05*0.1 = 0.005
            if epoch > 80  alors lr =  0.005*0.1 = 0.0005

    '''
    #return torch.optim.lr_scheduler.StepLR(optimizer=optim,gamma=opt.lr_scheduler_rate,step_size=opt.lr_scheduler_step)
    return torch.optim.lr_scheduler.MultiStepLR(optimizer=optim, milestones=opt.lr_decay_steps,
                                                gamma=opt.lr_scheduler_rate)


def print_model_parameters(model, only_num=True):
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('*****************Finish print_model_parameter function****************')


def get_memory_usage(device):
    allocated_memory = torch.cuda.memory_allocated(device) / (1024*1024.)
    cached_memory = torch.cuda.memory_cached(device) / (1024*1024.)
    #print('Allocated Memory: {:.2f} MB, Cached Memory: {:.2f} MB'.format(allocated_memory, cached_memory))
    return allocated_memory, cached_memory
