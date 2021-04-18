import numpy as np
import torch
import cv2

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
        cls_position = attention_list.shape[0]-2
        attention_list[cls_position] = attention_list.max()
        attention_list[cls_position] = attention_list.min()
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

def save_visual_results(input, cls_per_token_score,method_name='', COCO_path=None):
    input_mask = input['input_mask']
    expl_dir = './'

    bbox_scores = cls_per_token_score[0, input_mask.sum(1):]
    _, top_5_bboxes_indices = bbox_scores.topk(k=5, dim=-1)
    image_file_path = COCO_path + input['image_info_0']['feature_path'][0] + '.jpg'
    img = cv2.imread(image_file_path)
    for index in top_5_bboxes_indices:
        [x, y, w, h] = input['image_info_0']['bbox'][0][index]
        cv2.rectangle(img, (int(x), int(y)), ((int(w), int(h))), (255, 0, 0), 5)

    img = cv2.imread(image_file_path)
    mask = torch.zeros(img.shape[0], img.shape[1])
    bbox_scores = (bbox_scores - bbox_scores.min()) / (bbox_scores.max() - bbox_scores.min())
    for index in range(len(bbox_scores)):
        [x, y, w, h] = input['image_info_0']['bbox'][0][index]
        curr_score_tensor = mask[int(y):int(h), int(x):int(w)]
        new_score_tensor = torch.ones_like(curr_score_tensor) * bbox_scores[index].item()
        mask[int(y):int(h), int(x):int(w)] = torch.max(curr_score_tensor, new_score_tensor)
    mask = mask.unsqueeze_(-1)
    mask = mask.expand(img.shape)
    img = img * mask.cpu().data.numpy()
    cv2.imwrite(
        expl_dir + input['image_info_0']['feature_path'][0] + '_{0}.jpg'.format(method_name), img)


    token_scores = cls_per_token_score[0, : input_mask.sum(1)]
    generate(input['tokens'][0], token_scores, expl_dir + input['image_info_0']['feature_path'][0] + '_{0}.tex'.format(method_name))


class SelfAttentionGenerator:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_ours(self, input, index=None, save_visualization=False, head_prune=None, prune_step=None, COCO_path=None):
        output = self.model(input)['scores']
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        blocks = self.model.model.bert.encoder.layer
        num_tokens = blocks[0].attention.self.get_attn().shape[-1]
        R = torch.eye(num_tokens, num_tokens).to(blocks[0].attention.self.get_attn().device)
        for i,blk in enumerate(blocks):
            grad = blk.attention.self.get_attn_gradients()
            cam = blk.attention.self.get_attn()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
            if head_prune is not None:
                curr_prune = head_prune[i].reshape(-1, 1, 1).to(cam.device)
                cam = cam * curr_prune
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            R += torch.matmul(cam, R)
        input_mask = input['input_mask']
        cls_index = input_mask.sum(1) - 2
        cls_per_token_score = R[cls_index]
        cls_per_token_score[:, cls_index] = 0

        if save_visualization:
            save_visual_results(input, cls_per_token_score, method_name=prune_step, COCO_path=COCO_path)

        return cls_per_token_score
