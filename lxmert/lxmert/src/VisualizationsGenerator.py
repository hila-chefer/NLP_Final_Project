import numpy as np
import torch
import cv2

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    eye = torch.eye(num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].matmul(joint_attention)
    return joint_attention

def clean_word(word_list):
    new_word_list = []
    for word in word_list:
        for latex_sensitive in ["\\", "%", "&", "^", "#", "_",  "{", "}"]:
            if latex_sensitive in word:
                word = word.replace(latex_sensitive, '\\'+latex_sensitive)
        new_word_list.append(word)
    return new_word_list

def generate(text_list, attention_list, latex_file, color='red'):
    attention_list = attention_list[:len(text_list)]
    if attention_list.max() == attention_list.min():
        attention_list = torch.zeros_like(attention_list)
    else:
        attention_list[0] = attention_list[1:].min()
        attention_list = 100 * (attention_list - attention_list.min()) / (attention_list.max() - attention_list.min())
    attention_list[attention_list < 1] = 0
    attention_list = attention_list.tolist()
    text_list = [text_list[i].replace('$', '') for i in range(len(text_list))]
    word_num = len(text_list)
    text_list = clean_word(text_list)
    with open(latex_file,'w') as f:
        f.write(r'''\documentclass[varwidth=150mm]{standalone}
\special{papersize=210mm,297mm}
\usepackage{color}
\usepackage{tcolorbox}
\usepackage{CJK}
\usepackage{adjustbox}
\tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
\begin{document}
\begin{CJK*}{UTF8}{gbsn}'''+'\n')
        string = r'''{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{'''+"\n"
        for idx in range(word_num):
            if '\#\#' in text_list[idx]:
                token = text_list[idx].replace('\#\#', '')
                string += "\\colorbox{%s!%s}{" % (color, attention_list[idx]) + "\\strut " + token + "}"
            else:
                string += " " + "\\colorbox{%s!%s}{" % (color, attention_list[idx]) + "\\strut " + text_list[idx] + "}"
        string += "\n}}}"
        f.write(string+'\n')
        f.write(r'''\end{CJK*}
\end{document}''')

def save_visual_results(modelUsage, token_scores, image_scores, method_name):
    expl_dir = './'

    bbox_scores = image_scores
    _, top_bboxes_indices = bbox_scores.topk(k=1, dim=-1)
    image_file_path = modelUsage.image_file_path
    img = cv2.imread(image_file_path)
    cv2.imwrite(
        expl_dir + modelUsage.image_id + '_orig.jpg', img)

    for index in top_bboxes_indices:
        [x, y, w, h] = modelUsage.bboxes[0][index]

        alpha = 0.7
        img_out = (255 - img) * alpha + img
        img_out[int(y):int(h), int(x):int(w)] = img[int(y):int(h), int(x):int(w)]
        img_out = img_out.clip(max=255, min=0).astype(np.uint8)

        cv2.rectangle(img_out, (int(x), int(y)), ((int(w), int(h))), (0, 0, 0), 5)

    image_file_path = modelUsage.image_file_path
    img = cv2.imread(image_file_path)
    mask = torch.zeros(img.shape[0], img.shape[1])
    for index in range(len(bbox_scores)):
        [x, y, w, h] = modelUsage.bboxes[0][index]
        curr_score_tensor = mask[int(y):int(h), int(x):int(w)]
        new_score_tensor = torch.ones_like(curr_score_tensor)*bbox_scores[index].item()
        mask[int(y):int(h), int(x):int(w)] = torch.max(new_score_tensor,mask[int(y):int(h), int(x):int(w)])
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = mask.unsqueeze_(-1)
    mask = mask.expand(img.shape)
    img = img * mask.cpu().data.numpy()
    cv2.imwrite(
        expl_dir + modelUsage.image_id + '_{0}.jpg'.format(method_name), img)

    generate(modelUsage.question_tokens, token_scores, expl_dir + modelUsage.image_id + '_{0}.tex'.format(method_name))

# rule 5 from paper
def avg_heads(cam, grad, head_prune=None):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    if head_prune is not None:
        head_prune = head_prune.reshape(-1, 1, 1)
        cam = cam * head_prune
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam

# rules 6 + 7 from paper
def apply_self_attention_rules(R_ss, R_sq, cam_ss):
    R_sq_addition = torch.matmul(cam_ss, R_sq)
    R_sq_addition[torch.isnan(R_sq_addition)] = 0
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    R_ss_addition[torch.isnan(R_ss_addition)] = 0
    return R_ss_addition, R_sq_addition

# rules 10 + 11 from paper
def apply_mm_attention_rules(R_ss, R_qq, R_qs, cam_sq, apply_normalization=True, apply_self_in_rule_10=True):
    R_ss_normalized = R_ss
    R_qq_normalized = R_qq
    if apply_normalization:
        R_ss_normalized = handle_residual(R_ss)
        R_qq_normalized = handle_residual(R_qq)
    R_sq_addition = torch.matmul(R_ss_normalized.t(), torch.matmul(cam_sq, R_qq_normalized))
    R_sq_addition[torch.isnan(R_sq_addition)] = 0
    if not apply_self_in_rule_10:
        R_sq_addition = cam_sq
    R_ss_addition = torch.matmul(cam_sq, R_qs)
    R_ss_addition[torch.isnan(R_ss_addition)] = 0
    return R_sq_addition, R_ss_addition

# normalization- eq. 8+9
def handle_residual(orig_self_attention):
    self_attention = orig_self_attention.clone()
    diag_idx = range(self_attention.shape[-1])
    # computing R hat
    self_attention -= torch.eye(self_attention.shape[-1]).to(self_attention.device)
    assert self_attention[diag_idx, diag_idx].min() >= 0
    # normalizing R hat
    self_attention = self_attention / self_attention.sum(dim=-1, keepdim=True)
    self_attention += torch.eye(self_attention.shape[-1]).to(self_attention.device)
    return self_attention

class GeneratorOurs:
    def __init__(self, model_usage, save_visualization=False):
        self.model_usage = model_usage
        self.save_visualization = save_visualization

    def handle_self_attention_lang(self, blocks, head_prune=None):
        for i,blk in enumerate(blocks):
            grad = blk.attention.self.get_attn_gradients().detach()
            cam = blk.attention.self.get_attn().detach()
            cam = avg_heads(cam, grad, head_prune[i])
            R_t_t_add, R_t_i_add = apply_self_attention_rules(self.R_t_t, self.R_t_i, cam)
            self.R_t_t += R_t_t_add
            self.R_t_i += R_t_i_add

    def handle_self_attention_image(self, blocks, head_prune=None):
        for i, blk in enumerate(blocks):
            grad = blk.attention.self.get_attn_gradients().detach()
            cam = blk.attention.self.get_attn().detach()
            cam = avg_heads(cam, grad, head_prune[i])
            R_i_i_add, R_i_t_add = apply_self_attention_rules(self.R_i_i, self.R_i_t, cam)
            self.R_i_i += R_i_i_add
            self.R_i_t += R_i_t_add

    def handle_co_attn_self_lang(self, block, head_prune=None):
        grad = block.lang_self_att.self.get_attn_gradients().detach()
        cam = block.lang_self_att.self.get_attn().detach()
        cam = avg_heads(cam, grad, head_prune)
        R_t_t_add, R_t_i_add = apply_self_attention_rules(self.R_t_t, self.R_t_i, cam)
        self.R_t_t += R_t_t_add
        self.R_t_i += R_t_i_add

    def handle_co_attn_self_image(self, block, head_prune=None):
        grad = block.visn_self_att.self.get_attn_gradients().detach()
        cam = block.visn_self_att.self.get_attn().detach()
        cam = avg_heads(cam, grad, head_prune)
        R_i_i_add, R_i_t_add = apply_self_attention_rules(self.R_i_i, self.R_i_t, cam)
        self.R_i_i += R_i_i_add
        self.R_i_t += R_i_t_add

    def handle_co_attn_lang(self, block, head_prune=None):
        cam_t_i = block.visual_attention.att.get_attn().detach()
        grad_t_i = block.visual_attention.att.get_attn_gradients().detach()
        cam_t_i = avg_heads(cam_t_i, grad_t_i, head_prune)
        R_t_i_addition, R_t_t_addition = apply_mm_attention_rules(self.R_t_t, self.R_i_i, self.R_i_t, cam_t_i)
        return R_t_i_addition, R_t_t_addition

    def handle_co_attn_image(self, block, head_prune=None):
        cam_i_t = block.visual_attention_copy.att.get_attn().detach()
        grad_i_t = block.visual_attention_copy.att.get_attn_gradients().detach()
        cam_i_t = avg_heads(cam_i_t, grad_i_t, head_prune)
        R_i_t_addition, R_i_i_addition = apply_mm_attention_rules(self.R_i_i, self.R_t_t, self.R_t_i, cam_i_t)
        return R_i_t_addition, R_i_i_addition

    def generate_ours(self, input, index=None, method_name="ours", head_prune_image=None, head_prune_text=None):
        output = self.model_usage.forward(input).question_answering_score
        model = self.model_usage.model

        # initialize relevancy matrices
        text_tokens = self.model_usage.text_len
        image_bboxes = self.model_usage.image_boxes_len

        # text self attention matrix
        self.R_t_t = torch.eye(text_tokens, text_tokens).to(model.device)
        # image self attention matrix
        self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(model.device)
        # impact of images on text
        self.R_t_i = torch.zeros(text_tokens, image_bboxes).to(model.device)
        # impact of text on images
        self.R_i_t = torch.zeros(image_bboxes, text_tokens).to(model.device)

        index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        model.zero_grad()
        one_hot.backward(retain_graph=True)

        # language self attention
        blocks = model.lxmert.encoder.layer
        self.handle_self_attention_lang(blocks, head_prune_text[:len(blocks)])

        layer_index_text = len(blocks)

        # image self attention
        blocks = model.lxmert.encoder.r_layers
        self.handle_self_attention_image(blocks, head_prune_image[:len(blocks)])

        layer_index_image = len(blocks)


        # cross attn layers
        blocks = model.lxmert.encoder.x_layers
        for i, blk in enumerate(blocks):
            # in the last cross attention module, only the text cross modal
            # attention has an impact on the CLS token, since it's the first
            # token in the language tokens
            if i == len(blocks) - 1:
                break
            # cross attn- first for language then for image
            R_t_i_addition, R_t_t_addition = self.handle_co_attn_lang(blk, head_prune_text[layer_index_text])
            R_i_t_addition, R_i_i_addition = self.handle_co_attn_image(blk, head_prune_image[layer_index_image])

            layer_index_text += 1
            layer_index_image += 1

            self.R_t_i += R_t_i_addition
            self.R_t_t += R_t_t_addition
            self.R_i_t += R_i_t_addition
            self.R_i_i += R_i_i_addition

            # language self attention
            self.handle_co_attn_self_lang(blk, head_prune_text[layer_index_text])

            # image self attention
            self.handle_co_attn_self_image(blk, head_prune_image[layer_index_image])

            layer_index_text += 1
            layer_index_image += 1


        # take care of last cross attention layer- only text
        blk = model.lxmert.encoder.x_layers[-1]
        # cross attn- first for language then for image
        R_t_i_addition, R_t_t_addition = self.handle_co_attn_lang(blk, head_prune_text[layer_index_text])
        layer_index_text += 1
        self.R_t_i += R_t_i_addition
        self.R_t_t += R_t_t_addition

        # language self attention
        self.handle_co_attn_self_lang(blk, head_prune_text[layer_index_text])

        # disregard the [CLS] token itself
        self.R_t_t[0,0] = 0
        if self.save_visualization:
            save_visual_results(self.model_usage, self.R_t_t[0], self.R_t_i[0], method_name=method_name)
        return self.R_t_t, self.R_t_i
