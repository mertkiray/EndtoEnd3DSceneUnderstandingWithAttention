# SGCN
# author: chengzhang

import torch
import torch.nn as nn
from models.registers import MODULES
from configs.data_config import NYU40CLASSES, pix3d_n_classes
import torch.nn.functional as F
from net_utils.libs import get_bdb_form_from_corners, recover_points_to_world_sys, \
    get_rotation_matix_result, get_bdb_3d_result, recover_points_to_obj_sys
from torch_geometric.nn.conv import GATConv
import wandb


class Collection_Unit_wAttention(nn.Module):
    def __init__(self, dim_in,dim_out, hidden_size, layer_size, heads, final_head, dropout, final_dropout):
        super(Collection_Unit_wAttention, self).__init__()
        self.layer_size = layer_size

        self.convs = nn.ModuleList()
        if layer_size != 0:
            self.conv1 = GATConv(dim_in, hidden_size, heads=heads, dropout=dropout)
        else:
            self.conv1 = GATConv(dim_in, dim_out, heads=heads, concat=False, dropout=dropout)

        for layer in range(layer_size):
            if layer != layer_size - 1:
                self.convs.append(
                    GATConv(hidden_size * heads, hidden_size, heads=heads, dropout=dropout)
                )
            else:
                self.convs.append(
                    GATConv(hidden_size * heads, dim_out, heads=final_head, concat=False, dropout=final_dropout)
                )

    def forward(self, graph, attention_base):
        ## attention forward
        # result = self.conv1(graph, attention_base)
        if self.layer_size != 0:
            x = F.elu(self.conv1(graph, attention_base) + graph)
        else:
            x = self.conv1(graph, attention_base) + graph
        # x = self.conv2(x_1, attention_base) + x_1

        for i, layer in enumerate(self.convs):
            if i != len(self.convs)-1:
                x = F.elu(layer(x, attention_base) + x)
            else:
                x = layer(x, attention_base) + x
        return x
        # print(f'result shape: {result.shape}')
        # result1 = F.elu(result)
        # result = self.conv2(result1, attention_base)

@MODULES.register_module
class ATGCNN(nn.Module):
    def __init__(self, cfg, optim_spec=None):
        super(ATGCNN, self).__init__()

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec

        '''configs and params'''
        self.cfg = cfg
        self.lo_features = cfg.config['model']['output_adjust']['lo_features']
        self.obj_features = cfg.config['model']['output_adjust']['obj_features']
        self.rel_features = cfg.config['model']['output_adjust']['rel_features']
        feature_dim = cfg.config['model']['output_adjust']['feature_dim']
        self.feat_update_step = cfg.config['model']['output_adjust']['feat_update_step']
        self.res_output = cfg.config['model']['output_adjust'].get('res_output', False)
        self.feat_update_group = cfg.config['model']['output_adjust'].get('feat_update_group', 1)
        self.res_group = cfg.config['model']['output_adjust'].get('res_group', False)
        self.heads = cfg.config['model']['output_adjust']['heads']
        self.final_head = cfg.config['model']['output_adjust']['final_head']
        self.hidden_size = cfg.config['model']['output_adjust']['hidden_size']
        self.layer_size = cfg.config['model']['output_adjust']['layer_size']
        self.dropout = cfg.config['model']['output_adjust']['dropout']
        self.final_dropout = cfg.config['model']['output_adjust']['final_dropout']


        print(f'heads: {self.heads}')
        print(f'final_head: {self.final_head}')
        print(f'hidden_size: {self.hidden_size}')
        print(f'layer_size: {self.layer_size}')
        print(f'dropout: {self.dropout}')
        print(f'final_dropout: {self.final_dropout}')
        print(f'feature_dim: {feature_dim}')
        print(f'feat_update_step: {self.feat_update_step}')
        # print(f'wandb config: {wandb.config}')

        self.feature_length = {
            'size_cls': len(NYU40CLASSES), 'cls_codes': pix3d_n_classes,
            'bdb2D_pos': 4, 'g_features': 32, 'mgn_afeature': 1024, 'K': 3,
            'pitch_reg_result': 2, 'roll_reg_result': 2, 'pitch_cls_result': 2,
            'roll_cls_result': 2, 'lo_ori_reg_result': 2, 'lo_ori_cls_result': 2,
            'lo_centroid_result': 3, 'lo_coeffs_result': 3, 'lo_afeatures': 2048,
            'size_reg_result': 3, 'ori_reg_result': 6, 'ori_cls_result': 6,
            'centroid_reg_result': 6, 'centroid_cls_result': 6, 'offset_2D_result': 2,
            'odn_afeature': 2048, 'odn_rfeatures': 2048, 'odn_arfeatures': 2048,
            'ldif_afeature': cfg.config['model']['mesh_reconstruction'].get('bottleneck_size', None),
            'analytic_code': cfg.config['model']['mesh_reconstruction'].get('analytic_code_len', None),
            'blob_center': (cfg.config['model']['mesh_reconstruction'].get('element_count', 0)
                           + cfg.config['model']['mesh_reconstruction'].get('sym_element_count', 0)) * 3,
            'ldif_phy': (cfg.config['model']['mesh_reconstruction'].get('element_count', 0)
                           + cfg.config['model']['mesh_reconstruction'].get('sym_element_count', 0)) // 2,
            'structured_implicit_vector': cfg.config['model']['mesh_reconstruction'].get('structured_implicit_vector_len', None)
        }
        obj_features_len = sum([self.feature_length[k] for k in self.obj_features])
        rel_features_len = sum([self.feature_length[k] for k in self.rel_features]) * 2
        lo_features_len = sum([self.feature_length[k] for k in self.lo_features])

        bin = cfg.dataset_config.bins
        self.OBJ_ORI_BIN = len(bin['ori_bin'])
        self.OBJ_CENTER_BIN = len(bin['centroid_bin'])
        self.PITCH_BIN = len(bin['pitch_bin'])
        self.ROLL_BIN = len(bin['roll_bin'])
        self.LO_ORI_BIN = len(bin['layout_ori_bin'])

        '''modules'''
        # feature embedding (from graph-rcnn)
        self.obj_embedding = nn.Sequential(
            nn.Linear(obj_features_len, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, feature_dim),
        )
        self.rel_embedding = nn.Sequential(
            nn.Linear(rel_features_len, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, feature_dim),
        )
        self.lo_embedding = nn.Sequential(
            nn.Linear(lo_features_len, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, feature_dim),
        )

        # graph message passing (from graph-rcnn)
        if self.feat_update_step > 0:
            # self.gcn_collect_feat = nn.ModuleList([
            #     _GraphConvolutionLayer_Collect(feature_dim, feature_dim) for i in range(self.feat_update_group)])
            self.gcn_collect_feat = Collection_Unit_wAttention(feature_dim, feature_dim, self.hidden_size, self.layer_size,
                                                               self.heads, self.final_head, self.dropout, self.final_dropout)

        # feature to output (from Total3D object_detection)
        # branch to predict the size
        self.fc1 = nn.Linear(feature_dim, feature_dim // 2)
        self.fc2 = nn.Linear(feature_dim // 2, 3)

        # branch to predict the orientation
        self.fc3 = nn.Linear(feature_dim, feature_dim // 2)
        self.fc4 = nn.Linear(feature_dim // 2, self.OBJ_ORI_BIN * 2)

        # branch to predict the centroid
        self.fc5 = nn.Linear(feature_dim, feature_dim // 2)
        self.fc_centroid = nn.Linear(feature_dim // 2, self.OBJ_CENTER_BIN * 2)

        # branch to predict the 2D offset
        self.fc_off_1 = nn.Linear(feature_dim, feature_dim // 2)
        self.fc_off_2 = nn.Linear(feature_dim // 2, 2)

        # feature to output (from Total3D layout_estimation)
        self.fc_1 = nn.Linear(feature_dim, feature_dim // 2)
        self.fc_2 = nn.Linear(feature_dim // 2, (self.PITCH_BIN + self.ROLL_BIN) * 2)
        # fc for layout
        self.fc_layout = nn.Linear(feature_dim, feature_dim)
        # for layout orientation
        self.fc_3 = nn.Linear(feature_dim, feature_dim // 2)
        self.fc_4 = nn.Linear(feature_dim // 2, self.LO_ORI_BIN * 2)
        # for layout centroid and coefficients
        self.fc_5 = nn.Linear(feature_dim, feature_dim // 2)
        self.fc_6 = nn.Linear(feature_dim // 2, 6)

        self.relu_1 = nn.LeakyReLU(0.2)
        self.dropout_1 = nn.Dropout(p=0.5)

        # initiate weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()

    def _K2feature(self, K):
        camKs = K.reshape(K.shape[0], -1)
        camKs = camKs.index_select(1, torch.tensor([0, 2, 4, 5], device=camKs.device))
        camKs = camKs[:, :3] / camKs[:, 3:]
        return camKs

    def _get_bdb3D_form(self, data):
        # camera orientation for evaluation
        cam_R_out = get_rotation_matix_result(self.cfg.bins_tensor,
                                              torch.argmax(data['pitch_cls_result'], 1),
                                              data['pitch_reg_result'],
                                              torch.argmax(data['roll_cls_result'], 1),
                                              data['roll_reg_result'])

        # projected center
        P_result = torch.stack(((data['bdb2D_pos'][:, 0] + data['bdb2D_pos'][:, 2]) / 2 - (
                data['bdb2D_pos'][:, 2] - data['bdb2D_pos'][:, 0]) * data['offset_2D_result'][:, 0],
                                (data['bdb2D_pos'][:, 1] + data['bdb2D_pos'][:, 3]) / 2 - (
                                        data['bdb2D_pos'][:, 3] - data['bdb2D_pos'][:, 1]) * data['offset_2D_result'][:, 1]), 1)

        # retrieved 3D bounding box
        bdb3D_result, _ = get_bdb_3d_result(self.cfg.bins_tensor,
                                            torch.argmax(data['ori_cls_result'], 1),
                                            data['ori_reg_result'],
                                            torch.argmax(data['centroid_cls_result'], 1),
                                            data['centroid_reg_result'],
                                            data['size_cls'],
                                            data['size_reg_result'],
                                            P_result,
                                            data['K'],
                                            cam_R_out,
                                            data['split'])
        bdb3D_form = get_bdb_form_from_corners(bdb3D_result)

        return bdb3D_form

    def _get_object_features(self, data, type):
        features = []
        keys = self.obj_features if type == 'obj' else self.rel_features
        for k in keys:
            if k in ['size_cls', 'cls_codes', 'size_reg_result', 'ori_reg_result', 'ori_cls_result',
                     'centroid_reg_result', 'centroid_cls_result', 'offset_2D_result',
                     'ldif_afeature', 'mgn_afeature', 'odn_afeature', 'odn_rfeatures', 'odn_arfeatures']:
                v = data[k]
            elif k == 'g_features':
                assert type == 'rel'
                v = data[k]
            elif k == 'bdb2D_pos':
                v = data[k].clone()
                center_inds = data['K'][:, :2, 2]
                for center_ind, (start, end) in zip(center_inds, data['split']):
                    for i in range(start.item(), end.item()):
                        v[i][0] = (v[i][0] - center_ind[0]) / center_ind[0]
                        v[i][2] = (v[i][2] - center_ind[0]) / center_ind[0]
                        v[i][1] = (v[i][1] - center_ind[1]) / center_ind[1]
                        v[i][3] = (v[i][3] - center_ind[1]) / center_ind[1]
            elif k == 'K':
                camKs = self._K2feature(data[k])
                v = []
                for i, (start, end) in enumerate(data['split']):
                    v.append(camKs[i:i+1, :].expand(end-start, -1))
                v = torch.cat(v, 0)
            elif k in ['analytic_code', 'structured_implicit_vector', 'blob_center']:
                if k == 'analytic_code':
                    v = data['structured_implicit'].analytic_code
                elif k == 'structured_implicit_vector':
                    v = data['structured_implicit'].vector
                elif k == 'blob_center':
                    # get world_sys points from blob centers
                    bdb3D_form = self._get_bdb3D_form(data)
                    centers = data['structured_implicit'].all_centers.clone()
                    centers[:, :, 2] *= -1
                    v = recover_points_to_world_sys(bdb3D_form, centers, data['obj_center'], data['obj_coef'])
                v = v.reshape([v.shape[0], -1])
            elif k == 'ldif_phy':
                assert type == 'rel'
                bdb3D_form = self._get_bdb3D_form(data)
                structured_implicit = data['structured_implicit']
                ldif_center, ldif_coef = data['obj_center'], data['obj_coef']

                # get world_sys points from blob centers
                centers = data['structured_implicit'].all_centers.clone()
                centers[:, :, 2] *= -1
                obj_samples = recover_points_to_world_sys(bdb3D_form, centers, data['obj_center'], data['obj_coef'])
                element_count = centers.shape[1]

                # put points to other objects' coor
                max_sample_points = (data['split'][:, 1] - data['split'][:, 0]).max() * element_count
                other_obj_samples = torch.zeros([len(obj_samples), max_sample_points, 3], device=centers.device)
                for start, end in data['split']:
                    other_obj_sample = obj_samples[start:end].reshape(1, -1, 3).expand(end - start, -1, -1)
                    other_obj_samples[start:end, :other_obj_sample.shape[1]] = other_obj_sample
                other_obj_samples = recover_points_to_obj_sys(bdb3D_form, other_obj_samples, ldif_center, ldif_coef)
                other_obj_samples[:, :, 2] *= -1

                # get sdf of blob centers from other objects
                est_sdf = data['mgn'](
                    samples=other_obj_samples,
                    structured_implicit=structured_implicit.dict(),
                    apply_class_transfer=False,
                )['global_decisions'] + 0.07

                # reshape into relation features
                v = [est_sdf[start:end, :(end - start) * element_count].reshape(-1, element_count)
                     for start, end in data['split']]
                v = torch.cat(v)

            else:
                raise NotImplementedError

            if type == 'obj' or k in ('g_features', 'ldif_phy'):
                features.append(v)
            else:
                features_rel = []
                for start, end in data['split']:
                    features_rel.append(torch.stack(
                        [torch.cat([loc1, loc2], -1)
                         for loc1 in v[start:end]
                         for loc2 in v[start:end]]))
                features.append(torch.cat(features_rel, 0))
        return torch.cat(features, -1)

    def _get_layout_features(self, data):
        features = []
        keys = self.lo_features
        for k in keys:
            if k in ['pitch_reg_result', 'roll_reg_result', 'pitch_cls_result',
                     'roll_cls_result', 'lo_ori_reg_result', 'lo_ori_cls_result',
                     'lo_centroid_result', 'lo_coeffs_result', 'lo_afeatures']:
                v = data[k]
            elif k == 'K':
                v = self._K2feature(data[k])
            else:
                raise NotImplementedError
            features.append(v)
        return torch.cat(features, -1)

    def _get_map(self, data):
        device = data['g_features'].device
        # torch.set_printoptions(profile="full")
        split = data['split']
        # print(split)
        obj_num = split[-1][-1] + split.shape[0]  # number of objects and layouts
        obj_obj_map = torch.zeros([obj_num, obj_num])  # mapping of obj/lo vertices with connections
        rel_inds = []  # indexes of vertices connected by relation nodes
        rel_masks = []  # mask of relation features for obj/lo vertices connected by relation nodes
        obj_masks = torch.zeros(obj_num, dtype=torch.bool)  # mask of object vertices
        lo_masks = torch.zeros(obj_num, dtype=torch.bool)  # mask of layout vertices
        for lo_index, (start, end) in enumerate(split):
            start = start + lo_index  # each subgraph has Ni object vertices and 1 layout vertex
            end = end + lo_index + 1  # consider layout vertex, Ni + 1 vertices in total
            obj_obj_map[start:end, start:end] = 1  # each subgraph is a complete graph with self circle
            obj_ind = torch.arange(start, end, dtype=torch.long)
            subj_ind_i, obj_ind_i = torch.meshgrid(obj_ind, obj_ind)  # indexes of each vertex in the subgraph
            rel_ind_i = torch.stack([subj_ind_i.reshape(-1), obj_ind_i.reshape(-1)], -1)
            # print(rel_ind_i)
            rel_mask_i = rel_ind_i[:, 0] != rel_ind_i[:, 1]  # vertices connected by relation nodes should be different
            # print(f'REL MASK I: {rel_mask_i}')
            rel_inds.append(rel_ind_i[rel_mask_i])
            rel_masks.append(rel_mask_i)
            obj_masks[start:end - 1] = True  # for each subgraph, first Ni vertices are objects
            lo_masks[end - 1] = True  # for each subgraph, last 1 vertex is layout

        rel_inds = torch.cat(rel_inds, 0)
        rel_masks = torch.cat(rel_masks, 0)

        subj_pred_map = torch.zeros(obj_num, rel_inds.shape[0])  # [sum(Ni + 1), sum((Ni + 1) ** 2)]
        obj_pred_map = torch.zeros(obj_num, rel_inds.shape[0])
        # map from subject (an object or layout vertex) to predicate (a relation vertex)
        subj_pred_map.scatter_(0, (rel_inds[:, 0].view(1, -1)), 1)
        # map from object (an object or layout vertex) to predicate (a relation vertex)
        obj_pred_map.scatter_(0, (rel_inds[:, 1].view(1, -1)), 1)


        # print(f'rel_inds shape: {rel_inds.shape}')
        # print(f'rel_inds: {rel_inds}')
        # print(f'rel_inds 0: {rel_inds[:, 0]}')
        # print('=============================')
        # print(f'rel_inds 1: {rel_inds[:, 1]}')
        # print('=============================')
        # print(f'rel_mask: {rel_masks}')
        # print(f'rel_mask shape: {rel_masks.shape}')


        # print(f'subj_pred_map shape: {subj_pred_map.shape}')
        # print(f'subj_pred_map: {subj_pred_map}')
        #
        # print(f'obj_pred_map shape: {obj_pred_map.shape}')
        # print(f'obj_pred_map: {obj_pred_map}')

        # adj_map = map_init(obj_num + rel_inds.shape[0], obj_num + rel_inds.shape[0])
        # adj_map = map_init(5, 5)

        # print(f'obj num: {obj_num}')
        # print(f'rel_inds shape num: {rel_inds.shape[0]}')

        # print(adj_map)
        # exit(0)

        adj_map = self.edge_index_init(rel_inds, obj_num)


        return rel_masks.to(device), obj_masks.to(device), lo_masks.to(device), \
               obj_obj_map.to(device), subj_pred_map.to(device), obj_pred_map.to(device), adj_map.to(device)

    def edge_index_init(self, rel_inds, obj_num):
        # Connect all obj_nums togethers like if obj count 9 -> 2x81 fully connected
        obj_rel_keys = []
        obj_rel_map = []
        for i in range(obj_num):
            for j in range(obj_num):
                obj_rel_keys.append(i)
                obj_rel_map.append(j)
        #Add pred nodes between object nodes
        for i, rel_ind in enumerate(rel_inds):
            rel_index = i + obj_num
            for obj_node in rel_ind:
                obj_rel_keys.append(rel_index.item())
                obj_rel_map.append(obj_node.item())

                obj_rel_keys.append(obj_node.item())
                obj_rel_map.append(rel_index.item())

        rel_torch = torch.tensor([obj_rel_keys, obj_rel_map])
        return rel_torch

    def forward(self, output):
        maps = self._get_map(output)
        if maps is None:
            return {}

        ## attention maps
        rel_masks, obj_masks, lo_masks, obj_obj_map, subj_pred_map, obj_pred_map, adj_map = maps

        # print(subj_pred_map)

        x_obj, x_pred = self._get_object_features(output, 'obj'), self._get_object_features(output, 'rel')
        x_lo = self._get_layout_features(output)
        x_obj, x_pred = self.obj_embedding(x_obj), self.rel_embedding(x_pred)
        x_lo = self.lo_embedding(x_lo)

        x_obj_lo = [] # representation of object and layout vertices
        x_pred_objlo = [] # representation of relation vertices connecting obj/lo vertices
        rel_pair = output['rel_pair_counts']
        for lo_index, (start, end) in enumerate(output['split']):
            # print('1')
            x_obj_lo.append(x_obj[start:end]) # for each subgraph, first Ni vertices are objects
            x_obj_lo.append(x_lo[lo_index:lo_index+1]) # for each subgraph, last 1 vertex is layout
            x_pred_objlo.append(
                x_pred[rel_pair[lo_index]:rel_pair[lo_index + 1]].reshape(end - start, end - start, -1))
            x_pred_objlo[-1] = F.pad(x_pred_objlo[-1].permute(2,0,1), [0, 1, 0, 1], "constant", 0.001).permute(1,2,0)
            x_pred_objlo[-1] = x_pred_objlo[-1].reshape((end - start + 1) ** 2, -1)
        x_obj = torch.cat(x_obj_lo) # from here, for compatibility with graph-rcnn, x_obj corresponds to obj/lo vertices
        x_pred = torch.cat(x_pred_objlo)
        x_pred = x_pred[rel_masks]
        '''feature level agcn'''
        # obj_feats = [x_obj]
        # pred_feats = [x_pred]
        # print(f'x_obj shape: {x_obj.shape}')
        # print(f'x_pred shape: {x_pred.shape}')


        # collect = torch.mm(subj_pred_map.t(), x_obj)  # Nobj x Nrel Nrel x dim
        # print(f'collect shape: {collect.shape}')
        # print(f'collect shape: {collect}')

        all_feats = torch.cat((x_obj, x_pred), dim=0)

        for t in range(self.feat_update_step):
            all_feats = self.gcn_collect_feat(all_feats, adj_map)

        #obj_feats_wolo = obj_feats[-1][obj_masks]
        # print(obj_masks)
        # print('xxx')
        # print(num_objects)
        obj_feats_wolo = all_feats[:obj_masks.shape[0]]
        # print(obj_feats_wolo.shape)
        obj_feats_wolo = obj_feats_wolo[obj_masks]
        # print(obj_feats_wolo.shape)
        # branch to predict the size
        size = self.fc1(obj_feats_wolo)
        size = self.relu_1(size)
        size = self.dropout_1(size)
        size = self.fc2(size)

        # branch to predict the orientation
        ori = self.fc3(obj_feats_wolo)
        ori = self.relu_1(ori)
        ori = self.dropout_1(ori)
        ori = self.fc4(ori)
        ori = ori.view(-1, self.OBJ_ORI_BIN, 2)
        ori_reg = ori[:, :, 0]
        ori_cls = ori[:, :, 1]

        # branch to predict the centroid
        centroid = self.fc5(obj_feats_wolo)
        centroid = self.relu_1(centroid)
        centroid = self.dropout_1(centroid)
        centroid = self.fc_centroid(centroid)
        centroid = centroid.view(-1, self.OBJ_CENTER_BIN, 2)
        centroid_cls = centroid[:, :, 0]
        centroid_reg = centroid[:, :, 1]

        # branch to predict the 2D offset
        offset = self.fc_off_1(obj_feats_wolo)
        offset = self.relu_1(offset)
        offset = self.dropout_1(offset)
        offset = self.fc_off_2(offset)

        #obj_feats_lo = obj_feats[-1][lo_masks]
        obj_feats_lo = all_feats[:lo_masks.shape[0]]
        # branch for camera parameters
        obj_feats_lo = obj_feats_lo[lo_masks]
        cam = self.fc_1(obj_feats_lo)
        cam = self.relu_1(cam)
        cam = self.dropout_1(cam)
        cam = self.fc_2(cam)
        pitch_reg = cam[:, 0: self.PITCH_BIN]
        pitch_cls = cam[:, self.PITCH_BIN: self.PITCH_BIN * 2]
        roll_reg = cam[:, self.PITCH_BIN * 2: self.PITCH_BIN * 2 + self.ROLL_BIN]
        roll_cls = cam[:, self.PITCH_BIN * 2 + self.ROLL_BIN: self.PITCH_BIN * 2 + self.ROLL_BIN * 2]

        # branch for layout orientation, centroid and coefficients
        lo = self.fc_layout(obj_feats_lo)
        lo = self.relu_1(lo)
        lo = self.dropout_1(lo)
        # branch for layout orientation
        lo_ori = self.fc_3(lo)
        lo_ori = self.relu_1(lo_ori)
        lo_ori = self.dropout_1(lo_ori)
        lo_ori = self.fc_4(lo_ori)
        lo_ori_reg = lo_ori[:, :self.LO_ORI_BIN]
        lo_ori_cls = lo_ori[:, self.LO_ORI_BIN:]

        # branch for layout centroid and coefficients
        lo_ct = self.fc_5(lo)
        lo_ct = self.relu_1(lo_ct)
        lo_ct = self.dropout_1(lo_ct)
        lo_ct = self.fc_6(lo_ct)
        lo_centroid = lo_ct[:, :3]
        lo_coeffs = lo_ct[:, 3:]

        if self.res_output:
            size += output['size_reg_result']
            ori_reg += output['ori_reg_result']
            ori_cls += output['ori_cls_result']
            centroid_reg += output['centroid_reg_result']
            centroid_cls += output['centroid_cls_result']
            offset += output['offset_2D_result']

            pitch_reg += output['pitch_reg_result']
            pitch_cls += output['pitch_cls_result']
            roll_reg += output['roll_reg_result']
            roll_cls += output['roll_cls_result']
            lo_ori_reg += output['lo_ori_reg_result']
            lo_ori_cls += output['lo_ori_cls_result']
            lo_centroid += output['lo_centroid_result']
            lo_coeffs += output['lo_coeffs_result']

        return {'size_reg_result': size, 'ori_reg_result': ori_reg,
                'ori_cls_result': ori_cls, 'centroid_reg_result': centroid_reg,
                'centroid_cls_result': centroid_cls, 'offset_2D_result': offset,
                'pitch_reg_result': pitch_reg, 'pitch_cls_result': pitch_cls, 'roll_reg_result': roll_reg,
                'roll_cls_result': roll_cls, 'lo_ori_reg_result': lo_ori_reg, 'lo_ori_cls_result': lo_ori_cls,
                'lo_centroid_result': lo_centroid, 'lo_coeffs_result': lo_coeffs}
