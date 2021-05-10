# Mesh Generation Net: model loader
# author: ynie
# date: Feb, 2020

from models.registers import METHODS, MODULES, LOSSES
from models.network import BaseNetwork
import torch
from torch import nn

@METHODS.register_module
class MGNet(BaseNetwork):

    def __init__(self, cfg):
        '''
        load submodules for the network.
        :param config: customized configurations.
        '''
        super(BaseNetwork, self).__init__()
        self.cfg = cfg

        '''load network blocks'''
        for phase_name, net_spec in cfg.config['model'].items():
            method_name = net_spec['method']
            # load specific optimizer parameters
            optim_spec = self.load_optim_spec(cfg.config, net_spec)
            subnet = MODULES.get(method_name)(cfg, optim_spec)
            self.add_module(phase_name, subnet)

            '''load corresponding loss functions'''
            setattr(self, phase_name + '_loss', LOSSES.get(self.cfg.config['model'][phase_name]['loss'], 'Null')(
                self.cfg.config['model'][phase_name].get('weight', 1)))

        '''Multi-GPU setting'''
        self.mesh_reconstruction = nn.DataParallel(self.mesh_reconstruction)

        '''freeze submodules or not'''
        self.freeze_modules(cfg)

    def forward(self, data):

        est_data = self.mesh_reconstruction(
            data['img'], data['cls'], threshold=0.2, factor=1)

        return est_data

    def loss(self, est_data, gt_data):
        '''
        calculate loss of est_out given gt_out.
        '''
        loss = self.mesh_reconstruction_loss(est_data, gt_data, self.cfg.config['data']['tmn_subnetworks'],
                                             self.cfg.config['data']['face_samples'])
        total_loss = sum(loss.values())
        for key, item in loss.items():
            loss[key] = item.item()
        return {'total':total_loss, **loss}

    def freeze_modules(self, cfg):
        if cfg.config['mode'] == 'train':
            freeze_layers = cfg.config['train']['freeze']
            for layer in freeze_layers:
                if 'mesh_reconstruction_stage' in layer:
                    args = layer.split(',')
                    self.mesh_reconstruction.module.freeze_by_stage(int(args[1]), args[2:])
        super(MGNet, self).freeze_modules(cfg)
