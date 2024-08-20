import torch
from torch.utils.data import TensorDataset, DataLoader
import time
# import foolbox as fb
import datetime
import json
import os
from autoattack import AutoAttack
from torch.autograd import Variable
import pdb
import torch.nn as nn
import torch.optim as optim
from timm.layers.activations import GELU

def create_subdirs(sub_dir):
    os.makedirs(sub_dir, exist_ok=True)

class MeanSparse(nn.Module):
    def __init__(self, in_planes):
        super(MeanSparse, self).__init__()

        self.register_buffer('running_mean', torch.zeros(in_planes))
        self.register_buffer('running_var', torch.zeros(in_planes))

        self.register_buffer('threshold', torch.tensor(0.0))

        self.register_buffer('flag_update_statistics', torch.tensor(0))
        self.register_buffer('batch_num', torch.tensor(0.0))

    def forward(self, input):
        
        if self.flag_update_statistics:
            self.running_mean += (torch.mean(input.detach().clone(), dim=(0, 2, 3))/self.batch_num)
            self.running_var += (torch.var(input.detach().clone(), dim=(0, 2, 3))/self.batch_num)

        bias = self.running_mean.view(1, self.running_mean.shape[0], 1, 1)
        crop = self.threshold * torch.sqrt(self.running_var).view(1, self.running_var.shape[0], 1, 1)

        diff = input - bias

        if self.threshold == 0:
            output = input
        else:
            output = torch.where(torch.abs(diff) < crop, bias*torch.ones_like(input), input)

        return output

def generate_unique_name():
    # Get the current datetime
    now = datetime.datetime.now()
    # Format it into a string with date and time components
    unique_name = now.strftime("%Y%m%d%H%M%S")
    return unique_name


def convert_to_serializable(item):
    """Recursively convert items to a JSON-serializable format."""
    if isinstance(item, dict):
        return {key: convert_to_serializable(value) for key, value in item.items()}
    elif isinstance(item, list):
        return [convert_to_serializable(element) for element in item]
    elif isinstance(item, torch.Tensor):
        # Ensure tensor is moved to CPU before converting
        return item.detach().cpu().numpy().tolist()
    else:
        return item


def save_list_with_json(list_to_save, file_path):
    # Convert the entire data structure to a serializable format
    serializable_data = convert_to_serializable(list_to_save)

    with open(file_path, 'w') as file:
        json.dump(serializable_data, file)
    print(f"List saved to {file_path}")


def read_list_with_json(file_path):
    with open(file_path, 'r') as file:
        list_to_load = json.load(file)
    return list_to_load


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

    def write_to_tensorboard(self, writer, prefix, global_step):
        for meter in self.meters:
            writer.add_scalar(f"{prefix}/{meter.name}", meter.val, global_step)

    def write_avg_to_tensorboard(self, writer, prefix, global_step):
        for meter in self.meters:
            writer.add_scalar(f"{prefix}/{meter.name}", meter.avg, global_step)
       

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

def calculate_statistics(model, x, y, batch_size=512):
    model.eval()
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for name, module in model.named_modules():
        if isinstance(module, MeanSparse):
            module.flag_update_statistics.data.fill_(1)
            module.batch_num.data.fill_(len(dataloader))
    
    start = time.time()
    with torch.no_grad():
        for batch_idx, (images, target) in enumerate(dataloader):

            output = model(images)

            if (batch_idx + 1) % 10 == 0:
                print('Batch %d out of %d processes'%(batch_idx, len(dataloader)))
                print(time.time()-start)
                
    for name, module in model.named_modules():
        if isinstance(module, MeanSparse):
            module.flag_update_statistics.data.fill_(0)
            print(module.running_mean.data)
            print(module.running_var.data)


def calculate_statistics_dataloader(model, dataloader,device):
    model.eval()

    for name, module in model.named_modules():
        if isinstance(module, MeanSparse):
            module.flag_update_statistics.data.fill_(1)
            module.batch_num.data.fill_(len(dataloader))
    
    start = time.time()
    with torch.no_grad():
        for batch_idx, (images, target) in enumerate(dataloader):
            images = images.to(device)
            output = model(images)

            if (batch_idx + 1) % 10 == 0:
                print('Batch %d out of %d processes'%(batch_idx, len(dataloader)))
                print(time.time()-start)
                
    for name, module in model.named_modules():
        if isinstance(module, MeanSparse):
            module.flag_update_statistics.data.fill_(0)
            print(module.running_mean.data)
            print(module.running_var.data)


def calculate_accuracy(model, x_test, y_test, batch_size=256):
    model.eval()
    dataset = TensorDataset(x_test, y_test)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    top1 = AverageMeter("Acc_1", ":6.2f")
    top2 = AverageMeter("Acc_2", ":6.2f")
    start = time.time()
    with torch.no_grad():
        for batch_idx, (images, target) in enumerate(dataloader):

            output = model(images)
            acc1, acc2 = accuracy(output, target, topk=(1, 2))
            top1.update(acc1[0], images.size(0))
            top2.update(acc2[0], images.size(0))

            if (batch_idx + 1) % 1 == 0:
                print('Batch %d out of %d processes'%(batch_idx, len(dataloader)))
    time_elapsed = time.time() - start
    result = {"top1": top1.avg, "top2":  top2.avg, "time": time_elapsed}
    return result


def search_fine(model, x, y, vec_threshold, batch_size, complete_address, attacks):
    model.eval()
    complete_address_json = complete_address.split('.')[0]+'.json'

    if not os.path.exists(complete_address):
        vec_aa = torch.zeros_like(vec_threshold)
        vec_ac = torch.zeros_like(vec_threshold)
        stat_dict = {'vec_threshold':vec_threshold, 'vec_aa':vec_aa, 'vec_ac':vec_ac}
        torch.save(stat_dict, complete_address)
    else:
        stat_dict = torch.load(complete_address)

    vec_threshold = stat_dict['vec_threshold']
    vec_aa = stat_dict['vec_aa']
    vec_ac = stat_dict['vec_ac']

    zero_indices = torch.where(vec_ac == 0)[0]
    if len(zero_indices) > 0:
        id_start = zero_indices[0].item()
    else:
        id_start = vec_aa.shape[0]

    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=attacks)
    adversary.apgd.n_restarts = 1

    # for index, threshold in enumerate(vec_threshold):
    for index in range(id_start, vec_aa.shape[0]):
        threshold = vec_threshold[index]
        for name, module in model.named_modules():
            if isinstance(module, MeanSparse):
                module.threshold.data.fill_(threshold)
        
        result = calculate_accuracy(model, x, y, batch_size=batch_size)
        x_auto, y_auto = adversary.run_standard_evaluation(x, y,bs=batch_size, return_labels=True)
        vec_ac[index] = result['top1']
        correct = (y_auto == y).float()
        accuracy = correct.mean()*100
        vec_aa[index] = accuracy
        stat_dict = {'vec_threshold':vec_threshold, 'vec_aa':vec_aa, 'vec_ac':vec_ac}
        save_list_with_json(torch.stack((vec_threshold, vec_ac, vec_aa), dim=1), complete_address_json)
        torch.save(stat_dict, complete_address)
    
    id_best = torch.argmax(vec_aa)
    threshold_best = vec_threshold[id_best]
    return threshold_best


def benchmark(model, x, y, batch_size, directory, name_file):
    model.eval()
    result = calculate_accuracy(model, x, y, batch_size=batch_size)
    print(result)
    adversary = AutoAttack(model, norm='Linf', eps=8/255)
    x_auto, y_auto = adversary.run_standard_evaluation(x, y,return_labels=True)
    correct = (y_auto == y).float()
    accuracy = correct.mean()
    result['Robust_accuracy'] = accuracy
    save_list_with_json(result, os.path.join(directory, name_file))
    return result


def calculate_avg_acc(batch_sizes, accuracies, rob_accuracies):
    count = 0
    sum_natural = 0
    sum_robust = 0
    
    for i, n in enumerate(batch_sizes):
        sum_natural += accuracies[i] * n
        sum_robust += rob_accuracies[i] * n
        count += n
    return sum_natural/count, sum_robust/count

def AutoAttack_Wrapper(model, device, X, y, args, logger, sub_dir_path, start_batch, end_batch, checkpoint_path = None, distance="linf", epsilon=8/255, attack_type=None, batch_size = 8, workers=4, **kwargs):
    """
        Evaluate on atuo-attack adversarial validation set inputs.
    """
    _dataset = TensorDataset(X[:], y[:])
    data_loader = DataLoader(_dataset, batch_size=batch_size, shuffle=None, sampler=None, num_workers=workers, pin_memory=True)
    
    # switch to evaluation mode
    model.eval()

    adversary = AutoAttack(model, norm="Linf" if distance=="linf" else "L2", eps=epsilon)
    if attack_type != None:
        adversary.attacks_to_run = attack_type

    batch_sizes = []
    accuracies = []
    rob_accuracies = []
    
    if checkpoint_path != None:
        checkpoint = torch.load(checkpoint_path)
        start_batch = checkpoint['start_batch']
        batch_sizes = checkpoint['batch_sizes']
        accuracies = checkpoint['acc']
        rob_accuracies = checkpoint['rob_acc']
    
    print('Starting iteration: ', start_batch)
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            torch.cuda.empty_cache()
            if i >= start_batch and i <= end_batch:
                _start = time.time()    
                images, target = data[0].to(device), data[1].to(device)

                # clean images
                output = model(images)
                acc = accuracy(output, target, topk=(1,))
                accuracies.append(acc[0])
                batch_sizes.append(images.size(0))
                
                # Adv images
                x_adv = adversary.run_standard_evaluation(images, target, bs=len(images))
                output = model(x_adv)
                acc_adv = accuracy(output, target, topk=(1,))
                rob_accuracies.append(acc_adv[0])
                # Save checkpoint
                result = calculate_avg_acc(batch_sizes, accuracies, rob_accuracies)
                checkpoint = {
                                'start_batch': i+1,
                                'batch_sizes': batch_sizes,
                                'acc': accuracies,
                                'rob_acc': rob_accuracies,
                                "Natural Accuracy until now": result[0].item(),
                                "Robust Accuracy until now": result[1].item()
                            }
                torch.save(checkpoint, os.path.join(sub_dir_path, "checkpoint.pth.tar"))
        
                if (i - start_batch) % args.print_freq == 0:
                    logger.info("Iteration: " + str(i) + " ----------- Natural Accuracy: " + str(result[0].item()) + " -------- Robust Accuracy: " + str(result[1].item()))
                print("------------ One iteration time:", time.time() - _start)
                        
    result = calculate_avg_acc(batch_sizes, accuracies, rob_accuracies)
    return {"Natural Accuracy": result[0].item(), "Robust Accuracy": result[1].item()}



def pgd_whitebox(
    model,
    x,
    y,
    device,
    epsilon,
    num_steps,
    step_size,
    clip_min,
    clip_max,
    is_random=True,
    distance="linf",
    fp16=False,
    baseOptimizer=None,
    perc=99,
    normalize=None
):
    assert distance in ["l1", "linf", "l2"]
    if normalize is None:
        normalize = lambda x: x
    if distance == 'l1':
        x_pgd = x
        r = torch.zeros_like(x_pgd, requires_grad=True)
        for _ in range(num_steps):
            with torch.enable_grad():
                loss = nn.CrossEntropyLoss()(model(normalize(x_pgd + r)), y)
            if fp16:
                with amp.scale_loss(loss, baseOptimizer) as scaled_loss:
                    grad = torch.autograd.grad(scaled_loss, [r], create_graph=False)[0].detach()
                    grad = grad / (scaled_loss / loss)
            else:
                grad = torch.autograd.grad(loss, [r], create_graph=False)[0].detach()
            grad_mag = torch.abs(grad)
            grad_perc = np.percentile(grad_mag.detach().cpu(), perc)
            e = torch.where(grad_mag >= grad_perc, torch.sign(grad), 0)
            r = r + step_size * e / torch.norm(e, p=1, dim=1)
            r = proj_l1ball(r, epsilon, device)
        x_pgd = torch.clamp(x.data + r, clip_min, clip_max)

    if distance == "linf":
        if is_random:
            random_noise = (
                torch.FloatTensor(x.shape)
                .uniform_(-epsilon, epsilon)
                .to(device)
                .detach()
            )
        x_pgd = Variable(x.detach().data + random_noise, requires_grad=True)
        for _ in range(num_steps):
            with torch.enable_grad():
                loss = nn.CrossEntropyLoss()(model(normalize(x_pgd)), y)
            if fp16:
                with amp.scale_loss(loss, baseOptimizer) as scaled_loss: 
                    grad = torch.autograd.grad(scaled_loss, [x_pgd], create_graph=False)[0].detach()
                    grad = grad / (scaled_loss / loss)
            else:
                grad = torch.autograd.grad(loss, [x_pgd], create_graph=False)[0].detach()
            x_pgd.data = x_pgd.data + step_size * grad.data.sign()
            eta = torch.clamp(x_pgd.data - x.data, -epsilon, epsilon)
            x_pgd.data = torch.clamp(x.data + eta, clip_min, clip_max)

    if distance == "l2":
        if is_random:
            random_noise = (
                torch.FloatTensor(x.shape).uniform_(-1, 1).to(device).detach()
            )
            random_noise.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_pgd = Variable(x.detach().data + random_noise, requires_grad=True)
        for _ in range(num_steps):
            with torch.enable_grad():
                loss = nn.CrossEntropyLoss()(model(normalize(x_pgd)), y)
            if fp16:
                with amp.scale_loss(loss, baseOptimizer) as scaled_loss: 
                    grad = torch.autograd.grad(scaled_loss, [x_pgd], create_graph=False)[0].detach()
                    grad = grad / (scaled_loss / loss)
            else:
                grad = torch.autograd.grad(loss, [x_pgd], create_graph=False)[0].detach()
            
            # renorming gradient
            grad_norms = grad.view(len(x), -1).norm(p=2, dim=1)
            grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                grad[grad_norms == 0] = torch.randn_like(
                    grad[grad_norms == 0]
                )
            x_pgd.data += step_size * grad.data
            eta = x_pgd.data - x.data
            eta.renorm_(p=2, dim=0, maxnorm=epsilon)
            x_pgd.data = torch.clamp(x.data + eta, clip_min, clip_max)
    return x_pgd




def PGD_Wrapper(model, device, X, y, args, logger, sub_dir_path, start_batch, end_batch, checkpoint_path = None, distance="linf", epsilon=8/255, attack_type=None, batch_size = 8, workers=4, **kwargs):
    """
        Evaluate on atuo-attack adversarial validation set inputs.
    """
    _dataset = TensorDataset(X[:], y[:])
    data_loader = DataLoader(_dataset, batch_size=batch_size, shuffle=None, sampler=None, num_workers=workers, pin_memory=True)
    
    # switch to evaluation mode
    model.eval()
    normalize = lambda x: x
    
    batch_sizes = []
    accuracies = []
    rob_accuracies = []
    
    if checkpoint_path != None:
        checkpoint = torch.load(checkpoint_path)
        start_batch = checkpoint['start_batch']
        batch_sizes = checkpoint['batch_sizes']
        accuracies = checkpoint['acc']
        rob_accuracies = checkpoint['rob_acc']
    
    print('Starting iteration: ', start_batch)
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            torch.cuda.empty_cache()
            if i >= start_batch and i <= end_batch:
                _start = time.time()    
                images, target = data[0].to(device), data[1].to(device)

                # clean images
                output = model(images)
                acc = accuracy(output, target, topk=(1,))
                accuracies.append(acc[0])
                batch_sizes.append(images.size(0))
                
                # Adv images
                images.requires_grad = True
                x_adv = pgd_whitebox(model,
                                    images,
                                    target,
                                    device,
                                    epsilon,
                                    num_steps = 50,
                                    step_size = 0.0078,
                                    clip_min = 0,
                                    clip_max = 1,
                                    is_random=True,
                                    distance=distance,
                                    normalize=normalize)
                output = model(x_adv)
                acc_adv = accuracy(output, target, topk=(1,))
                rob_accuracies.append(acc_adv[0])
                # Save checkpoint
                result = calculate_avg_acc(batch_sizes, accuracies, rob_accuracies)
                checkpoint = {
                                'start_batch': i+1,
                                'batch_sizes': batch_sizes,
                                'acc': accuracies,
                                'rob_acc': rob_accuracies,
                                "Natural Accuracy until now": result[0].item(),
                                "Robust Accuracy until now": result[1].item()
                            }
                torch.save(checkpoint, os.path.join(sub_dir_path, "checkpoint.pth.tar"))
        
                if (i - start_batch) % args.print_freq == 0:
                    logger.info("Iteration: " + str(i) + " ----------- Natural Accuracy: " + str(result[0].item()) + " -------- Robust Accuracy: " + str(result[1].item()))
                print("------------ One iteration time:", time.time() - _start)
                        
    result = calculate_avg_acc(batch_sizes, accuracies, rob_accuracies)
    return {"Natural Accuracy": result[0].item(), "Robust Accuracy": result[1].item()}
