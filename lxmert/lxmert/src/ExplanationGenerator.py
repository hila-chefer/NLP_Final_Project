import numpy as np
import torch

class HeadPrune:
    def __init__(self, model_usage):
        self.model_usage = model_usage

    def generate_ours(self, input):
        kwargs = {"alpha": 1}
        output = self.model_usage.forward(input).question_answering_score
        model = self.model_usage.model

        index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        model.zero_grad()
        one_hot.backward(retain_graph=True)
        model.relprop(torch.tensor(one_hot_vector).to(output.device), **kwargs)

        grad_scores_text = []
        cam_scores_text = []

        grad_scores_image = []
        cam_scores_image = []

        def calc_head_scores(grad_score_per_head, cam_score_per_head, attn, is_text_layer):
            grad_score_per_head = grad_score_per_head.reshape(-1, grad_score_per_head.shape[-2], grad_score_per_head.shape[-1])
            cam_score_per_head = cam_score_per_head.reshape(-1, cam_score_per_head.shape[-2], cam_score_per_head.shape[-1])
            attn = attn.reshape(-1, attn.shape[-2],attn.shape[-1])

            ############ to use grad * attn
            grad_score_per_head = grad_score_per_head * attn
            ############ to use grad * attn

            grad_score_per_head = grad_score_per_head.mean(dim=[1, 2])
            cam_score_per_head = cam_score_per_head.mean(dim=[1, 2])

            if is_text_layer:
                grad_scores_text.append(grad_score_per_head.flatten())
                cam_scores_text.append(cam_score_per_head.flatten())
            else:
                grad_scores_image.append(grad_score_per_head.flatten())
                cam_scores_image.append(cam_score_per_head.flatten())

        # language self attention
        blocks = model.lxmert.encoder.layer

        for blk in blocks:
            #### grads for all attention maps- h x (t+i) x (t+i)
            grad_score_per_head = blk.attention.self.get_attn_gradients().detach()
            #### lrp for all attention maps- h x (t+i) x (t+i)
            cam_score_per_head = blk.attention.self.get_attn_cam().detach()
            #### attention map
            attn = blk.attention.self.get_attn().detach()

            calc_head_scores(grad_score_per_head, cam_score_per_head, attn, is_text_layer=True)

        # image self attention
        blocks = model.lxmert.encoder.r_layers
        for blk in blocks:
            #### grads for all attention maps- h x (t+i) x (t+i)
            grad_score_per_head = blk.attention.self.get_attn_gradients().detach()
            #### lrp for all attention maps- h x (t+i) x (t+i)
            cam_score_per_head = blk.attention.self.get_attn_cam().detach()
            #### attention map
            attn = blk.attention.self.get_attn().detach()

            calc_head_scores(grad_score_per_head, cam_score_per_head, attn, is_text_layer=False)

        # cross attn layers
        blocks = model.lxmert.encoder.x_layers
        for i, blk in enumerate(blocks):
            # in the last cross attention module, only the text cross modal
            # attention has an impact on the CLS token, since it's the first
            # token in the language tokens
            if i == len(blocks) - 1:
                break
            # cross attn- first for language then for image

            ###### for language- cross attn
            #### grads for all attention maps- h x (t+i) x (t+i)
            grad_score_per_head = blk.visual_attention.att.get_attn_gradients().detach()
            #### lrp for all attention maps- h x (t+i) x (t+i)
            cam_score_per_head = blk.visual_attention.att.get_attn_cam().detach()
            #### attention map
            attn = blk.visual_attention.att.get_attn().detach()

            calc_head_scores(grad_score_per_head, cam_score_per_head, attn, is_text_layer=True)

            ###### for image- cross attn
            #### grads for all attention maps- h x (t+i) x (t+i)
            grad_score_per_head = blk.visual_attention_copy.att.get_attn_gradients().detach()
            #### lrp for all attention maps- h x (t+i) x (t+i)
            cam_score_per_head = blk.visual_attention_copy.att.get_attn_cam().detach()
            #### attention map
            attn = blk.visual_attention_copy.att.get_attn().detach()

            calc_head_scores(grad_score_per_head, cam_score_per_head, attn, is_text_layer=False)

            ###### for language- self attn
            #### grads for all attention maps- h x (t+i) x (t+i)
            grad_score_per_head = blk.lang_self_att.self.get_attn_gradients().detach()
            #### lrp for all attention maps- h x (t+i) x (t+i)
            cam_score_per_head = blk.lang_self_att.self.get_attn_cam().detach()
            #### attention map
            attn = blk.lang_self_att.self.get_attn().detach()

            calc_head_scores(grad_score_per_head, cam_score_per_head, attn, is_text_layer=True)

            ###### for image- self attn
            #### grads for all attention maps- h x (t+i) x (t+i)
            grad_score_per_head = blk.visn_self_att.self.get_attn_gradients().detach()
            #### lrp for all attention maps- h x (t+i) x (t+i)
            cam_score_per_head = blk.visn_self_att.self.get_attn_cam().detach()
            #### attention map
            attn = blk.visn_self_att.self.get_attn().detach()

            calc_head_scores(grad_score_per_head, cam_score_per_head, attn, is_text_layer=False)


        # take care of last cross attention layer- only text
        blk = model.lxmert.encoder.x_layers[-1]
        ###### for language- cross attn
        #### grads for all attention maps- h x (t+i) x (t+i)
        grad_score_per_head = blk.visual_attention.att.get_attn_gradients().detach()
        #### lrp for all attention maps- h x (t+i) x (t+i)
        cam_score_per_head = blk.visual_attention.att.get_attn_cam().detach()
        #### attention map
        attn = blk.visual_attention.att.get_attn().detach()

        calc_head_scores(grad_score_per_head, cam_score_per_head, attn, is_text_layer=True)

        ###### for language- self attn
        #### grads for all attention maps- h x (t+i) x (t+i)
        grad_score_per_head = blk.lang_self_att.self.get_attn_gradients().detach()
        #### lrp for all attention maps- h x (t+i) x (t+i)
        cam_score_per_head = blk.lang_self_att.self.get_attn_cam().detach()
        #### attention map
        attn = blk.lang_self_att.self.get_attn().detach()

        calc_head_scores(grad_score_per_head, cam_score_per_head, attn, is_text_layer=True)

        return grad_scores_text, grad_scores_image, cam_scores_text, cam_scores_image

class LayerPrune:
    def __init__(self, model_usage):
        self.model_usage = model_usage

    def generate_ours(self, input):
        kwargs = {"alpha": 1}
        output = self.model_usage.forward(input).question_answering_score
        model = self.model_usage.model

        index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        model.zero_grad()
        one_hot.backward(retain_graph=True)
        model.relprop(torch.tensor(one_hot_vector).to(output.device), **kwargs)

        grad_scores_text = []
        cam_scores_text = []

        grad_scores_image = []
        cam_scores_image = []

        def calc_layer_scores(grad_score_per_head, cam_score_per_head, attn, is_text_layer):
            grad_score_per_head = grad_score_per_head.reshape(-1, grad_score_per_head.shape[-2], grad_score_per_head.shape[-1])
            cam_score_per_head = cam_score_per_head.reshape(-1, cam_score_per_head.shape[-2], cam_score_per_head.shape[-1])
            attn = attn.reshape(-1, attn.shape[-2],attn.shape[-1])

            ############ to use grad * attn
            grad_score_per_head = grad_score_per_head * attn
            ############ to use grad * attn

            grad_score_per_layer = grad_score_per_head.mean(dim=[0, 1, 2])
            cam_score_per_layer = cam_score_per_head.mean(dim=[0, 1, 2])

            if is_text_layer:
                grad_scores_text.append(grad_score_per_layer.item())
                cam_scores_text.append(cam_score_per_layer.item())
            else:
                grad_scores_image.append(grad_score_per_layer.item())
                cam_scores_image.append(cam_score_per_layer.item())

        # language self attention
        blocks = model.lxmert.encoder.layer

        for blk in blocks:
            #### grads for all attention maps- h x (t+i) x (t+i)
            grad_score_per_head = blk.attention.self.get_attn_gradients().detach()
            #### lrp for all attention maps- h x (t+i) x (t+i)
            cam_score_per_head = blk.attention.self.get_attn_cam().detach()
            #### attention map
            attn = blk.attention.self.get_attn().detach()

            calc_layer_scores(grad_score_per_head, cam_score_per_head, attn, is_text_layer=True)

        # image self attention
        blocks = model.lxmert.encoder.r_layers
        for blk in blocks:
            #### grads for all attention maps- h x (t+i) x (t+i)
            grad_score_per_head = blk.attention.self.get_attn_gradients().detach()
            #### lrp for all attention maps- h x (t+i) x (t+i)
            cam_score_per_head = blk.attention.self.get_attn_cam().detach()
            #### attention map
            attn = blk.attention.self.get_attn().detach()

            calc_layer_scores(grad_score_per_head, cam_score_per_head, attn, is_text_layer=False)

        # cross attn layers
        blocks = model.lxmert.encoder.x_layers
        for i, blk in enumerate(blocks):
            # in the last cross attention module, only the text cross modal
            # attention has an impact on the CLS token, since it's the first
            # token in the language tokens
            if i == len(blocks) - 1:
                break
            # cross attn- first for language then for image

            ###### for language- cross attn
            #### grads for all attention maps- h x (t+i) x (t+i)
            grad_score_per_head = blk.visual_attention.att.get_attn_gradients().detach()
            #### lrp for all attention maps- h x (t+i) x (t+i)
            cam_score_per_head = blk.visual_attention.att.get_attn_cam().detach()
            #### attention map
            attn = blk.visual_attention.att.get_attn().detach()

            calc_layer_scores(grad_score_per_head, cam_score_per_head, attn, is_text_layer=True)

            ###### for image- cross attn
            #### grads for all attention maps- h x (t+i) x (t+i)
            grad_score_per_head = blk.visual_attention_copy.att.get_attn_gradients().detach()
            #### lrp for all attention maps- h x (t+i) x (t+i)
            cam_score_per_head = blk.visual_attention_copy.att.get_attn_cam().detach()
            #### attention map
            attn = blk.visual_attention_copy.att.get_attn().detach()

            calc_layer_scores(grad_score_per_head, cam_score_per_head, attn, is_text_layer=False)

            ###### for language- self attn
            #### grads for all attention maps- h x (t+i) x (t+i)
            grad_score_per_head = blk.lang_self_att.self.get_attn_gradients().detach()
            #### lrp for all attention maps- h x (t+i) x (t+i)
            cam_score_per_head = blk.lang_self_att.self.get_attn_cam().detach()
            #### attention map
            attn = blk.lang_self_att.self.get_attn().detach()

            calc_layer_scores(grad_score_per_head, cam_score_per_head, attn, is_text_layer=True)

            ###### for image- self attn
            #### grads for all attention maps- h x (t+i) x (t+i)
            grad_score_per_head = blk.visn_self_att.self.get_attn_gradients().detach()
            #### lrp for all attention maps- h x (t+i) x (t+i)
            cam_score_per_head = blk.visn_self_att.self.get_attn_cam().detach()
            #### attention map
            attn = blk.visn_self_att.self.get_attn().detach()

            calc_layer_scores(grad_score_per_head, cam_score_per_head, attn, is_text_layer=False)


        # take care of last cross attention layer- only text
        blk = model.lxmert.encoder.x_layers[-1]
        ###### for language- cross attn
        #### grads for all attention maps- h x (t+i) x (t+i)
        grad_score_per_head = blk.visual_attention.att.get_attn_gradients().detach()
        #### lrp for all attention maps- h x (t+i) x (t+i)
        cam_score_per_head = blk.visual_attention.att.get_attn_cam().detach()
        #### attention map
        attn = blk.visual_attention.att.get_attn().detach()

        calc_layer_scores(grad_score_per_head, cam_score_per_head, attn, is_text_layer=True)

        ###### for language- self attn
        #### grads for all attention maps- h x (t+i) x (t+i)
        grad_score_per_head = blk.lang_self_att.self.get_attn_gradients().detach()
        #### lrp for all attention maps- h x (t+i) x (t+i)
        cam_score_per_head = blk.lang_self_att.self.get_attn_cam().detach()
        #### attention map
        attn = blk.lang_self_att.self.get_attn().detach()

        calc_layer_scores(grad_score_per_head, cam_score_per_head, attn, is_text_layer=True)

        return grad_scores_text, grad_scores_image, cam_scores_text, cam_scores_image
