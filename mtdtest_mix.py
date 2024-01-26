import __init__
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
import torch_geometric.datasets as GeoData
from torch_geometric.data import DenseDataLoader
import torch_geometric.transforms as T
import torch.nn.functional as F
from config import OptInit
from architecture import DenseDeepGCN
from utils.ckpt_util import load_pretrained_models, load_pretrained_optimizer, save_checkpoint
import logging
from attacks import NU_attack_exp, tar_NU_attack_exp, NB_attack_exp, tar_NB_attack_exp
from attacks import torchattacks
import copy
import time
import os
import os.path as osp

data_train_clean_path = 'data/deepgcn/S3DIS/train_clean'
data_test_clean_path = 'data/deepgcn/S3DIS/test_clean'
data_train_aug_path = 'data/deepgcn/S3DIS/train_aug'
data_test_aug_path = 'data/deepgcn/S3DIS/test_aug'

class ColorPerturbation(object):
    def __init__(self, intensity=0.1):
        self.intensity = intensity
    def __call__(self, data):
        if hasattr(data, 'color') and data.color is not None:
            perturbation = (torch.rand_like(data.color) - 0.5) * 2 * self.intensity
            data.color = torch.clamp(data.color + perturbation, min=0, max=1)
        return data

class RandomFlip(T.RandomFlip):
    def __init__(self, p=0.5, axis=0):
        super().__init__(p=p, axis=axis)

data_augmentation_transformations = T.Compose([
    ColorPerturbation(0.05), #(0.2), #(0.2),
    T.RandomJitter (0.03), # (0.03), #(0.03),  # Parameters for RandomJitter
    RandomFlip (0.05), # (0.5), #(0.5),  # Parameters for RandomFlip
    T.RandomScale ((0.8,0.8)), # ((0.8,1.2)), #((0.8, 1.2)),  # Parameters for RandomScale
    T.RandomRotate(degrees=30, axis=0), # 30 # Parameters for RandomRotate
    T.RandomRotate(degrees=30, axis=1),
    T.RandomRotate(degrees=30, axis=2),
    T.RandomShear(1),  # Parameters for RandomShear
    T.NormalizeScale()  # Normalizing the scale
])

def helper_get_accuracy(opt,model,data_loader,adversarial_attack_model=None):
    model.eval() # set model to evaluate
    acc = np.empty(len(data_loader))
    for i, data in enumerate(tqdm(data_loader)):
        data = data.to(opt.device)
        inputs = torch.cat((data.pos.transpose(2, 1).unsqueeze(3), data.x.transpose(2, 1).unsqueeze(3)), 1)
        gt = data.y
        # make adversarial data
        if adversarial_attack_model:
            inputs = adversarial_attack_model(inputs,gt) # clean data becomes adversarial data
        out = model(inputs)
        pred = out.max(dim=1)[1]
        pred_label = pred.detach().cpu().numpy()
        acc[i] = pred.eq(gt.view_as(pred)).sum().item() / 4096
    return np.mean(acc)

# for calculating transferability, test set'non shuffled', and attack model is provided
def helper_get_labels(opt,model,data_loader,adversarial_attack_model):
    model.eval() # set model to evaluate
    all_predicted_labels = np.array([])
    for i, data in enumerate(tqdm(data_loader)):
        data = data.to(opt.device)
        inputs = torch.cat((data.pos.transpose(2, 1).unsqueeze(3), data.x.transpose(2, 1).unsqueeze(3)), 1)
        gt = data.y
        # make adversarial data
        if adversarial_attack_model:
            inputs = adversarial_attack_model(inputs,gt) # clean data becomes adversarial data
        out = model(inputs)
        pred = out.max(dim=1)[1]
        pred_label = pred.detach().cpu().numpy()
        all_predicted_labels = np.concatenate((all_predicted_labels, pred_labels))
    return all_predicted_labels

#train for only 1 epoch, to be called repeatedly #if adversarial attack model exist, data used to trained is adversarially attacked
def helper_train(optimizer, criterion, opt, model, dataset, adversarial_attack_model=None): 
    model.train() # set model to train
    #shuffled training data distinct for each child model
    data_loader = DenseDataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    for i, data in enumerate(data_loader.dataset):
        if not opt.multi_gpus:
            data = data.to(opt.device)
        inputs = torch.cat((data.pos.transpose(2, 1).unsqueeze(3), data.x.transpose(2, 1).unsqueeze(3)), 1)
        gt = data.y.to(opt.device)
        # make adversarial data
        if adversarial_attack_model:
            inputs = adversarial_attack_model(inputs,gt) # clean data becomes adversarial data
        # ------------------ zero, output, loss
        optimizer.zero_grad()
        out = model(inputs)
        loss = criterion(out, gt)
        # ------------------ optimization
        loss.backward()
        optimizer.step()
    return model

#def child_model_generation(fb, x_train, x_test, lamda=0.1, e=0.01, max_acc_loss=0.1, adv_train=True): # algorithm 1 / from Morphence paper
def child_model_generation(optimizer,criterion,opt,base_model,base_model_acc, test_clean_loader,lamda=0.1, epsilon=0.1, max_acc_loss=0.1, adversarial_attack_model=None):
    # training dataset should be unique for each child, hence shuffle=true
    # testing dataset remains the same
    train_augmented_dataset = GeoData.S3DIS(data_train_aug_path, 5, train=True, pre_transform=data_augmentation_transformations)
    
    total_size = len(train_augmented_dataset)
    subset_size = int(datasize*total_size)
    train_augmented_dataset = train_augmented_dataset[:subset_size]
    
    train_augmented_loader = DenseDataLoader(train_augmented_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    child_model = None
    #debug_step12 = 0
    while True:
        # STEP 1 : create child model from base model and perturb model weights # using laplacian noise distribution
        child_model = copy.deepcopy(base_model) #.state_dict())
        for i in child_model.state_dict():
            if 'backbone' in i and 'running' in i: # perturb specific structure of resgcn
                child_model.state_dict()[i] += torch.cuda.FloatTensor(np.random.laplace(loc=0.0, scale=lamda, size=child_model.state_dict()[i].size()))
        # STEP 2: retraining on augmented data
        child_model = retrain(optimizer, criterion, opt, child_model,train_augmented_loader,test_clean_loader,epsilon=0.1,adversarial_attack_model=None)    
        acc_s = helper_get_accuracy(opt, child_model, test_clean_loader, adversarial_attack_model=None)
        if base_model_acc - acc_s < max_acc_loss:
            break
        lamda *= 0.75 # this ensures lamda is not 0 that would result in child_model = base_model
        print('new lambda of child model', lamda)
    # STEP 3: retraining on adversarial data
    #debugbreak = 0
    if adversarial_attack_model:
        child_model = retrain(optimizer, criterion, opt, child_model,train_augmented_loader,test_clean_loader,epsilon=0.1,adversarial_attack_model=adversarial_attack_model)    
        while True:
            acc_s = helper_get_accuracy(child_model,test_clean_loader,opt,adversarial_attack_model=None)
            if base_model_acc - acc_s < max_acc_loss:
                break
            #debugbreak += 1
            #if debugbreak > 2:
            #    print('retrain on advesarial data early break accuracy',acc_s,'accuracy loss from base model', base_model_acc - acc_s)
            #    break
            child_model = retrain(optimizer, criterion, opt, child_model,train_augmented_loader,test_clean_loader,epsilon=0.1,adversarial_attack_model=None)   
    # for debugging
    if adversarial_attack_model:
        print('generation of child-model-on-adversarial-data with,lamda-',lamda,'base-vs-child accuracy', base_model_acc ,' | ', acc_s)        
    else:
        print('generation of child-model-on-augmented  -data with,lamda-',lamda,'base-vs-child accuracy', base_model_acc ,' | ', acc_s)        
    return child_model

#def retrain(fs,x_retrain,x_test,e=0.1,adv_train=True): # child model retraining / algorithm 2 / from morphence paper
def retrain(optimizer, criterion, opt, model,data_loader_augmented,data_loader_test,epsilon=0.1,adversarial_attack_model=None):
    # //from research paper// for adversarial traning we use adversarial test examples for validation
    # if attack_model: # if attack model is provided, adversarial data is created
        # x_test = evasion(x_test) # evasion not implemented
        # research papers assume adversarial test dataset is created as a whole but in this implementation, adversarial data is created by piece
    acc_tmp = helper_get_accuracy(opt, model, data_loader_test, adversarial_attack_model)
    epoch = 0
    #epoch = 3 # debug
    while True:
        model = helper_train(optimizer, criterion, opt, model,data_loader_augmented, adversarial_attack_model)
        acc = helper_get_accuracy(opt, model, data_loader_test, adversarial_attack_model)
        print('retrain accuracy-',acc)
        # //from research paper// check training convergence
        if epoch > 2: # if epochs mod(5) = 0 then
            if abs(acc-acc_tmp) < epsilon: # acc - acctmp < e
                print('in retrain function early exit epsilon accuracy-', acc)
                break
            else:
                acc_tmp = acc
        epoch += 1
    return model #fs

def helper_get_transferability_rate(opt,test_clean_loader):
    correct_labels = np.array([])
    for data in test_clean_loader:
        data = data.to(opt.device)
        correct_labels = np.concatenate((correct_labels, data.y.cpu().numpy()))
    all_checkpoints = []
    for i in range(3):
        all_checkpoints.append('child_model_clean_'+str(i)+'.pth')
        all_checkpoints.append('child_model_adversarial_'+str(i)+'.pth')
    result = []    
    for i in range(len(all_checkpoints)):
        for j in range(len(all_checkpoints)):
            if i != j:
                # load model A to attack it
                checkpoint = torch.load(all_checkpoints[i])
                a_model = DenseDeepGCN(opt).to(opt.device)
                a_model.load_state_dict(checkpoint['model_state_dict'])
                adversarial_attack_model = torchattacks.NU_attack(a_model, c=1e-1, kappa=0, steps=1000, lr=0.1)
                a_labels = helper_get_labels(opt,a_model,test_clean_loader,adversarial_attack_model)
                # load model B to test transfer attack from A to B
                checkpoint = torch.load(all_checkpoints[j])
                b_model = DenseDeepGCN(opt).to(opt.device)
                b_model.load_state_dict(checkpoint['model_state_dict'])
                b_labels = helper_get_labels(opt,b_model,test_clean_loader,adversarial_attack_model)
                # get result
                accuracy_rate = (np.count_nonzero(correct_labels == b_labels) / np.size(b_labels)) 
                transferability_rate = (np.count_nonzero(a_labels == b_labels) / np.size(b_labels)) 
                result.append([a_labels,b_labels])
                print('accuracy of model',j,'=',accuracy_rate, 'with transferability rate from',i,'to',j,'=',transferability_rate)
    avg_result = np.mean(np.array(result), axis=0)
    print('all average accuracy & tranferability',avg_result)

def get_dataset(opt,data_path,train=True,pre_transform=T.NormalizeScale()):
    dataset = GeoData.S3DIS(data_path, 5, train=train, pre_transform=pre_transform)
    subset_size = int(opt.dataset_size_ratio * len(dataset))
    return dataset[:subset_size]
    
def main():
    opt = OptInit().get_args()
    # load s3dis dataset start
    train_clean_dataset = get_dataset(opt,data_train_clean_path,train=True,pre_transform=T.NormalizeScale())
    test_clean_dataset = get_dataset(opt,data_test_clean_path,train=False,pre_transform=T.NormalizeScale())
    train_aug_dataset = get_dataset(opt,data_train_clean_path,train=True,pre_transform=data_augmentation_transformations)
    test_aug_dataset = get_dataset(opt,data_test_clean_path,train=False,pre_transform=data_augmentation_transformations)
    ratio = int(opt.data_clean_to_aug_ratio * len(train_clean_dataset))
    train_mix_dataset = train_clean_dataset[:ratio] + train_aug_dataset[ratio:]
    # load s3dis dataset end   
    opt.n_classes = train_clean_loader.dataset.num_classes
    # load base model
    base_model = DenseDeepGCN(opt).to(opt.device)
    base_model, opt.best_value, opt.epoch = load_pretrained_models(base_model, opt.pretrained_model, opt.phase)
    # load model training components
    criterion = torch.nn.CrossEntropyLoss().to(opt.device)
    optimizer = torch.optim.Adam(base_model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_adjust_freq, opt.lr_decay_rate)
    optimizer, scheduler, opt.lr = load_pretrained_optimizer(opt.pretrained_model, optimizer, scheduler, opt.lr)
    # load adversarial_attack_model
    adversarial_attack_model = torchattacks.NU_attack(base_model, c=1e-1, kappa=0, steps=1000, lr=0.1)
    # get base model accuracy
    #base_model_accuracy = helper_get_accuracy(opt,base_model,test_clean_loader,adversarial_attack_model=None)
    print('base model accuracy on clean data-',base_model_accuracy)
    
    child_model_clean = child_model_generation(optimizer,criterion,opt,base_model,base_model_accuracy, test_clean_loader,lamda=0.1, epsilon=0.05, max_acc_loss=0.5, adversarial_attack_model=None)
    torch.save(child_model_clean.state_dict(),'child_model_mix_' + str(opt.data_clean_to_aug_ratio) +'.pth')
    helper_get_transferability_rate(opt,test_clean_loader)

if __name__ == '__main__':
    main()
