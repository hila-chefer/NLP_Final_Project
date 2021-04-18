import numpy as np
import torch
import cv2

class HeadPrune:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_expl(self, input, index=None):
        output = self.model(input)['scores']
        kwargs = {"alpha": 1}

        ##### gradient propagation
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        ####### LRP propagation
        self.model.relprop(torch.tensor(one_hot_vector).to(output.device), **kwargs)

        grad_scores = []
        cam_scores = []
        blocks = self.model.model.bert.encoder.layer
        for blk in blocks:
            #### grads for all attention maps- h x (t+i) x (t+i)
            grad_score_per_head = blk.attention.self.get_attn_gradients()[0]
            #### lrp for all attention maps- h x (t+i) x (t+i)
            cam_score_per_head = blk.attention.self.get_attn_cam()[0]
            #### attention map
            attn = blk.attention.self.get_attn()[0]

            grad_score_per_head = grad_score_per_head * attn

            grad_score_per_head = grad_score_per_head.mean(dim=[1,2])
            cam_score_per_head = cam_score_per_head.mean(dim=[1,2])

            grad_scores.append(grad_score_per_head)
            cam_scores.append(cam_score_per_head)

        return grad_scores, cam_scores


class LayerPrune:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_expl(self, input, index=None):
        output = self.model(input)['scores']
        kwargs = {"alpha": 1}

        ##### gradient propagation
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        ####### LRP propagation
        self.model.relprop(torch.tensor(one_hot_vector).to(output.device), **kwargs)

        grad_scores = []
        cam_scores = []
        blocks = self.model.model.bert.encoder.layer
        for blk in blocks:
            #### grads for all attention maps- h x (t+i) x (t+i)
            grad_score_per_layer = blk.attention.self.get_attn_gradients()[0]
            #### lrp for all attention maps- h x (t+i) x (t+i)
            cam_score_per_layer = blk.attention.self.get_attn_cam()[0]
            #### attention map
            attn = blk.attention.self.get_attn()[0]

            grad_score_per_layer = grad_score_per_layer * attn

            grad_score_per_layer = grad_score_per_layer.mean(dim=[0,1,2]).item()
            cam_score_per_layer = cam_score_per_layer.mean(dim=[0,1,2]).item()

            grad_scores.append(grad_score_per_layer)
            cam_scores.append(cam_score_per_layer)

        return grad_scores, cam_scores
