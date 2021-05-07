import os
import sys
import traceback
import types
import gc

import torch
import torch.nn as nn
import torch.optim as optim

from diora.net.diora import DioraTreeLSTM
from diora.net.diora import DioraMLP
from diora.net.diora import DioraMLPShared

from diora.logging.configuration import get_logger

def override_inside_hook(var):
    def func(self, level, h, c, s):
        length = self.length
        B = self.batch_size
        L = length - level

        assert s.shape[0] == B
        assert s.shape[1] == L
        assert s.shape[2] == level
        assert s.shape[3] == 1
        assert len(s.shape) == 4
        smax = s.max(2, keepdim=True)[0]
        s = s - smax
        # TODO try with p

        for pos in range(L):
            # print("Saving {} {}".format(level,pos))
            self.saved_scalars[level][pos] = s[:, pos, :]
        # print(len(self.saved_scalars))
    var.inside_hook = types.MethodType(func, var)

class ReconstructionLoss(nn.Module):
    name = 'reconstruct_loss'

    def __init__(self, embeddings, input_size, size, margin=1, k_neg=3, cuda=False):
        super(ReconstructionLoss, self).__init__()
        self.k_neg = k_neg
        self.margin = margin

        self.embeddings = embeddings
        self.mat = nn.Parameter(torch.FloatTensor(size, input_size))
        self._cuda = cuda
        self.reset_parameters()

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def loss_hook(self, sentences, neg_samples, inputs):
        pass

    def forward(self, sentences, neg_samples, diora, info):
        batch_size, length = sentences.shape
        input_size = self.embeddings.weight.shape[1]
        size = diora.outside_h.shape[-1]
        k = self.k_neg

        emb_pos = self.embeddings(sentences)
        emb_neg = self.embeddings(neg_samples)

        # Calculate scores.

        ## The predicted vector.
        cell = diora.outside_h[:, :length].view(batch_size, length, 1, -1)

        ## The projected samples.
        proj_pos = torch.matmul(emb_pos, torch.t(self.mat))
        proj_neg = torch.matmul(emb_neg, torch.t(self.mat))

        ## The score.
        xp = torch.einsum('abc,abxc->abx', proj_pos, cell)
        xn = torch.einsum('ec,abxc->abe', proj_neg, cell)
        score = torch.cat([xp, xn], 2)

        # Calculate loss.
        lossfn = nn.MultiMarginLoss(margin=self.margin)
        inputs = score.view(batch_size * length, k + 1)
        device = torch.cuda.current_device() if self._cuda else None
        outputs = torch.full((inputs.shape[0],), 0, dtype=torch.int64, device=device)

        self.loss_hook(sentences, neg_samples, inputs)

        loss = lossfn(inputs, outputs)

        ret = dict(reconstruction_loss=loss)

        return loss, ret


class ReconstructionSoftmaxLoss(nn.Module):
    name = 'reconstruct_softmax_loss'

    def __init__(self, embeddings, input_size, size, margin=1, k_neg=3, cuda=False):
        super(ReconstructionSoftmaxLoss, self).__init__()
        self.k_neg = k_neg
        self.margin = margin
        self.input_size = input_size

        self.embeddings = embeddings
        self.mat = nn.Parameter(torch.FloatTensor(size, input_size))
        self._cuda = cuda
        self.reset_parameters()

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def loss_hook(self, sentences, neg_samples, inputs):
        pass

    def forward(self, sentences, neg_samples, diora, info):
        batch_size, length = sentences.shape
        input_size = self.input_size
        size = diora.outside_h.shape[-1]
        k = self.k_neg

        emb_pos = self.embeddings(sentences)
        emb_neg = self.embeddings(neg_samples.unsqueeze(0))

        # Calculate scores.

        ## The predicted vector.
        cell = diora.outside_h[:, :length].view(batch_size, length, 1, -1)

        ## The projected samples.
        proj_pos = torch.matmul(emb_pos, torch.t(self.mat))
        proj_neg = torch.matmul(emb_neg, torch.t(self.mat))

        ## The score.
        xp = torch.einsum('abc,abxc->abx', proj_pos, cell)
        xn = torch.einsum('zec,abxc->abe', proj_neg, cell)
        score = torch.cat([xp, xn], 2)

        # Calculate loss.
        lossfn = nn.CrossEntropyLoss()
        inputs = score.view(batch_size * length, k + 1)
        device = torch.cuda.current_device() if self._cuda else None
        outputs = torch.full((inputs.shape[0],), 0, dtype=torch.int64, device=device)

        self.loss_hook(sentences, neg_samples, inputs)

        loss = lossfn(inputs, outputs)

        ret = dict(reconstruction_softmax_loss=loss)

        return loss, ret


# Our rule based loss. The parameter ce_loss controls whether ranking loss or cross entropy loss is used. 
class RuleBasedLoss(nn.Module):
    name = 'rule_based_loss'

    def __init__(self, embeddings, input_size, size, margin=1, k_neg=3, cuda=False, ce_loss=False):
        super(RuleBasedLoss, self).__init__()
        self.k_neg = k_neg
        self.margin = margin
        self.input_size = input_size

        # self.embeddings = embeddings  # 3346
        self.mat = nn.Embedding(2502,1,padding_idx=2500)  # TODO put number of rules
        # These will eventually become rule weights
        self._cuda = cuda
        self.ce_loss = ce_loss
        self.reset_parameters()

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def loss_hook(self, sentences, neg_samples, inputs):
        pass

    def forward(self, sentences, rule_labels, diora, info):
        batch_size, length = sentences.shape
        input_size = self.input_size
        size = diora.outside_h.shape[-1]
        k = self.k_neg
        loss = 0.
        rule_lab_idx = 0
        rules = rule_labels['rules']
        mask  = rule_labels['mask']
        rule_indices = rule_labels['indices']
        # print("----------\n",diora.saved_scalars)
        for level in range(1, length):
            L = length - level
            N = level

            for pos in range(L):
                # Assumes that the bottom-left most leaf is in the first constituent.
                # print(diora.saved_scalars[level][pos].size())
                # print(len(rules[rule_lab_idx]),[rules[rule_lab_idx][x].size() for x in range(N)])
                # print(rules[rule_lab_idx].size(),rule_indices[rule_lab_idx].size())
                spbatch = torch.softmax(diora.saved_scalars[level][pos],1).squeeze(dim=-1)  # Is this necessary?


                rule_lab  = torch.zeros_like(spbatch)
                diff_mask = torch.zeros_like(spbatch)

                if torch.max(rules[rule_lab_idx]).item() > 2500:
                    print(rules[rule_lab_idx])
                masked_rules = self.mat(rules[rule_lab_idx]).squeeze(dim=-1)*mask[rule_lab_idx]

                rule_lab = torch.abs(1*rule_lab.scatter_add(1,rule_indices[rule_lab_idx],masked_rules))  # This has N entries with some rules # TODO vectorize somehow using scatter add
                rule_lab = torch.exp(rule_lab) / (torch.exp(rule_lab).sum(dim=-1).view(-1,1)+1e-10)

                if self.ce_loss:
                    loss = loss - 1.5*(torch.abs(rule_lab) * torch.log(spbatch+1e-10)).sum() + 1e-2* torch.abs(rule_lab).sum()
                else:
                    # Get the signs of pairwise difference between r_i
                    rule_lab_diff  = torch.triu(torch.tanh(50*(rule_lab.unsqueeze(1) - rule_lab.unsqueeze(2))))
                    # Get the signs of pairwise difference between e_i
                    actual_diff    = torch.triu(torch.tanh(50*(spbatch.unsqueeze(1) - spbatch.unsqueeze(2))))
                    
                    # Construct mask for positions which are not triggering any rule 
                    diff_mask      = diff_mask.scatter_add(1,rule_indices[rule_lab_idx],mask[rule_lab_idx])
                    diff_mask      = torch.triu(torch.sign(diff_mask.unsqueeze(1) * diff_mask.unsqueeze(2)))
                    
                    loss = loss + 1e-2*(((rule_lab_diff - actual_diff)*diff_mask)**2).sum() 
                rule_lab_idx += 1
        # for i in range()
        ret = dict(rule_based_loss=loss)

        return loss, ret

def get_loss_funcs(options, batch_iterator=None, embedding_layer=None):
    input_dim = embedding_layer.weight.shape[1]
    size = options.hidden_dim
    k_neg = options.k_neg
    margin = options.margin
    cuda = options.cuda

    loss_funcs = []

    # Reconstruction Loss
    if options.reconstruct_mode == 'margin':
        reconstruction_loss_fn = ReconstructionLoss(embedding_layer,
            margin=margin, k_neg=k_neg, input_size=input_dim, size=size, cuda=cuda)
    elif options.reconstruct_mode == 'softmax':
        reconstruction_loss_fn = ReconstructionSoftmaxLoss(embedding_layer,
            margin=margin, k_neg=k_neg, input_size=input_dim, size=size, cuda=cuda)

    loss_funcs.append(reconstruction_loss_fn)

    if options.rule_based:
        loss_funcs.append(RuleBasedLoss(embedding_layer,
            margin=margin, k_neg=k_neg, input_size=input_dim, size=size, cuda=cuda))

    return loss_funcs


class Embed(nn.Module):
    def __init__(self, embeddings, input_size, size):
        super(Embed, self).__init__()
        self.input_size = input_size
        self.size = size
        self.embeddings = embeddings
        self.mat = nn.Parameter(torch.FloatTensor(size, input_size))
        self.reset_parameters()

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def forward(self, x):
        batch_size, length = x.shape
        e = self.embeddings(x.view(-1))
        t = torch.mm(e, self.mat.t()).view(batch_size, length, -1)
        return t


class Net(nn.Module):
    def __init__(self, embed, diora, loss_funcs=[], epoch_curriculum=1000):
        super(Net, self).__init__()
        size = diora.size

        self.embed = embed
        self.diora = diora
        self.loss_func_names = [m.name for m in loss_funcs]
        self.epoch_curriculum = epoch_curriculum
        for m in loss_funcs:
            setattr(self, m.name, m)

        self.reset_parameters()

    def reset_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        for i, param in enumerate(params):
            param.data.normal_()

    def compute_loss(self, batch, neg_samples, rules,info,epochs):
        ret, loss = {}, []

        # Loss
        diora = self.diora.get_chart_wrapper()
        for func_name in self.loss_func_names:
            # if 
            func = getattr(self, func_name)
            if func_name == 'rule_based_loss':
                # print("---------------------\n",batch)
                neg_samples_ = rules
                lamda = epochs*0.1
            else:
                neg_samples_ = neg_samples
                lamda = 1  
            subloss, desc = func(batch, neg_samples_, diora, info)
            loss.append(lamda*subloss.view(1, 1))
            for k, v in desc.items():
                ret[k] = v

        loss = torch.cat(loss, 1)

        return ret, loss

    def forward(self, batch, neg_samples=None,rules=None, compute_loss=True, info=None, epochs=-1):
        # Embed
        embed = self.embed(batch)

        # Run DIORA
        self.diora(embed)

        # Compute Loss
        # if epochs > self.epoch_curriculum:
        #   self.loss_func_names = ['rule_based_loss','reconstruct_softmax_loss']
        # else:
        #   self.loss_func_names = ['reconstruct_softmax_loss']

        if compute_loss:
            ret, loss = self.compute_loss(batch, neg_samples,rules, info=info, epochs=epochs)
        else:
            ret, loss = {}, torch.full((1, 1), 1, dtype=torch.float32,
                device=embed.device)

        # Results
        ret['total_loss'] = loss

        return ret


class Trainer(object):
    def __init__(self, net, k_neg=None, ngpus=None, cuda=None, curriculum=False):
        super(Trainer, self).__init__()
        self.net = net
        self.optimizer = None
        self.optimizer_cls = None
        self.optimizer_kwargs = None
        self.cuda = cuda
        self.ngpus = ngpus
        self.curriculum = curriculum

        self.parallel_model = None

        print("Trainer initialized with {} gpus.".format(ngpus))

    def freeze_diora(self):
        for p in self.net.diora.parameters():
            p.requires_grad = False

    def parameter_norm(self, requires_grad=True, diora=False):
        net = self.net.diora if diora else self.net
        total_norm = 0
        for p in net.parameters():
            if requires_grad and not p.requires_grad:
                continue
            total_norm += p.norm().item()
        return total_norm

    def init_optimizer(self, optimizer_cls, optimizer_kwargs):
        if optimizer_cls is None:
            optimizer_cls = self.optimizer_cls
        if optimizer_kwargs is None:
            optimizer_kwargs = self.optimizer_kwargs
        params = [p for p in self.net.parameters() if p.requires_grad]
        self.optimizer = optimizer_cls(params, **optimizer_kwargs)

    @staticmethod
    def get_single_net(net):
        if isinstance(net, torch.nn.parallel.DistributedDataParallel):
            return net.module
        return net

    def save_model(self, model_file):
        state_dict = self.net.state_dict()

        todelete = []

        for k in state_dict.keys():
            if 'embeddings' in k:
                todelete.append(k)

        for k in todelete:
            del state_dict[k]

        torch.save({
            'state_dict': state_dict,
        }, model_file)

    @staticmethod
    def load_model(net, model_file):
        save_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict_toload = save_dict['state_dict']
        state_dict_net = Trainer.get_single_net(net).state_dict()

        # Bug related to multi-gpu
        keys = list(state_dict_toload.keys())
        prefix = 'module.'
        for k in keys:
            if k.startswith(prefix):
                newk = k[len(prefix):]
                state_dict_toload[newk] = state_dict_toload[k]
                del state_dict_toload[k]

        # Remove extra keys.
        keys = list(state_dict_toload.keys())
        for k in keys:

            if k not in state_dict_net:
                print('deleting {}'.format(k))
                del state_dict_toload[k]

        # Hack to support embeddings.
        for k in state_dict_net.keys():
            if 'embeddings' in k:
                state_dict_toload[k] = state_dict_net[k]

            if 'rule_based_loss' in k:
                if k not in save_dict.keys():
                    state_dict_toload[k] = state_dict_net[k]
                # else:
                #   state_dict_toload[k] = 

        Trainer.get_single_net(net).load_state_dict(state_dict_toload)

    def run_net(self, batch_map, compute_loss=True, multigpu=False, epochs=-1):
        batch = batch_map['sentences']
        neg_samples = batch_map.get('neg_samples', None)
        info = self.prepare_info(batch_map)
        rules_and_mask = {"rules":batch_map['rules'], "mask": batch_map['rules_mask'], "indices": batch_map['rule_indices']}
        out = self.net(batch, neg_samples=neg_samples, rules=rules_and_mask,compute_loss=compute_loss, info=info, epochs=epochs)
        return out

    def gradient_update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        params = [p for p in self.net.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        self.optimizer.step()

    def prepare_result(self, batch_map, model_output):
        result = {}
        result['batch_size'] = batch_map['batch_size']
        result['length'] = batch_map['length']
        for k, v in model_output.items():
            if 'loss' in k:
                result[k] = v.mean(dim=0).sum().item()
        return result

    def prepare_info(self, batch_map):
        return {}

    def step(self, *args, **kwargs):
        try:
            return self._step(*args, **kwargs)
        except Exception as err:
            batch_map = args[0]
            print('Failed with shape: {}'.format(batch_map['sentences'].shape))
            if self.ngpus > 1:
                print(traceback.format_exc())
                print('The step failed. Running multigpu cleanup.')
                os.system("ps -elf | grep [p]ython | grep adrozdov | grep " + self.experiment_name + " | tr -s ' ' | cut -f 4 -d ' ' | xargs -I {} kill -9 {}")
                sys.exit()
            else:
                raise err

    def _step(self, batch_map, train=True, compute_loss=True, epochs=-1):
        if train:
            self.net.train()
        else:
            self.net.eval()
        multigpu = self.ngpus > 1 and train
        if not self.curriculum:
            epochs = 10
        with torch.set_grad_enabled(train):
            model_output = self.run_net(batch_map, compute_loss=compute_loss, multigpu=multigpu , epochs=epochs)

        # Calculate average loss for multi-gpu and sum for backprop.
        total_loss = model_output['total_loss'].mean(dim=0).sum()

        if train:
            self.gradient_update(total_loss)

        result = self.prepare_result(batch_map,  model_output)
        return result
        # return {"batch_size":1,"length":1}

    def get_top_rules(self,model_file):
        rule_ten = torch.arange(0,3346).view(1,-1).int()
        # print(self.loss_func_names)
        # func = getattr(self.net, "rule_based_loss")
        save_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict_toload = save_dict['state_dict']
        rule_mat = state_dict_toload['rule_based_loss.mat.weight']
        # mat = nn.Embedding(3346,1,padding_idx=3345)
        # mat.weight.data.copy_(rule_mat.long())
        # mat.weight.data.copy

        # emb = torch.abs(mat(rule_ten.long()))
        return (torch.argsort(-1.*torch.abs(rule_mat).squeeze())[:200].detach().cpu().numpy())


def build_net(options, embeddings=None, batch_iterator=None, random_seed=None):

    logger = get_logger()

    lr = options.lr
    size = options.hidden_dim
    k_neg = options.k_neg
    margin = options.margin
    normalize = options.normalize
    input_dim = embeddings.shape[1]
    cuda = options.cuda
    rank = options.local_rank
    curriculum = options.curriculum
    ngpus = 1

    if cuda and options.multigpu:
        ngpus = torch.cuda.device_count()
        os.environ['MASTER_ADDR'] = options.master_addr
        os.environ['MASTER_PORT'] = options.master_port
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # Embed
    embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(embeddings), freeze=True)
    embed = Embed(embedding_layer, input_size=input_dim, size=size)

    # Diora
    if options.arch == 'treelstm':
        diora = DioraTreeLSTM(size, outside=True, normalize=normalize, compress=False)
    elif options.arch == 'mlp':
        diora = DioraMLP(size, outside=True, normalize=normalize, compress=False)
    elif options.arch == 'mlp-shared':
        diora = DioraMLPShared(size, outside=True, normalize=normalize, compress=False)

    # Loss
    loss_funcs = get_loss_funcs(options, batch_iterator, embedding_layer)

    if options.rule_based:
        print("Overriding")
        override_inside_hook(diora)
    # Net
    net = Net(embed, diora, loss_funcs=loss_funcs)

    # Load model.
    if options.load_model_path is not None:
        logger.info('Loading model: {}'.format(options.load_model_path))
        Trainer.load_model(net, options.load_model_path)

    # CUDA-support
    if cuda:
        if options.multigpu:
            torch.cuda.set_device(options.local_rank)
        net.cuda()
        diora.cuda()

    if cuda and options.multigpu:
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[rank], output_device=rank)

    # Trainer
    trainer = Trainer(net, k_neg=k_neg, ngpus=ngpus, cuda=cuda, curriculum=curriculum)
    trainer.rank = rank
    trainer.experiment_name = options.experiment_name # for multigpu cleanup
    trainer.init_optimizer(optim.Adam, dict(lr=lr, betas=(0.9, 0.999), eps=1e-8))

    return trainer
