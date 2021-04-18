from lxmert.lxmert.src.tasks import vqa_data
from lxmert.lxmert.src.modeling_frcnn import GeneralizedRCNN
import lxmert.lxmert.src.vqa_utils as utils
from lxmert.lxmert.src.processing_image import Preprocess
from transformers import LxmertTokenizer
from lxmert.lxmert.src.huggingface_lxmert import LxmertForQuestionAnswering
from lxmert.lxmert.src.lxmert_lrp import LxmertForQuestionAnswering as LxmertForQuestionAnsweringLRP
from tqdm import tqdm
from lxmert.lxmert.src.VisualizationsGenerator import GeneratorOurs as VisGen
from lxmert.lxmert.src.ExplanationGenerator import HeadPrune, LayerPrune
import random
from lxmert.lxmert.src.param import args
import torch
import cv2
import os
from PIL import Image
import psutil


OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"
VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"

class ModelUsage:
    def __init__(self, COCO_VAL_PATH):
        self.COCO_VAL_PATH = COCO_VAL_PATH
        self.vqa_answers = utils.get_data(VQA_URL)

        # load models and model components
        self.frcnn_cfg = utils.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.frcnn_cfg.MODEL.DEVICE = "cuda"

        self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.frcnn_cfg)

        self.image_preprocess = Preprocess(self.frcnn_cfg)

        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

        self.lxmert_vqa = LxmertForQuestionAnsweringLRP.from_pretrained("unc-nlp/lxmert-vqa-uncased").to("cuda")
        self.lxmert_vqa_no_lrp = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-vqa-uncased").to("cuda")

        self.lxmert_vqa.eval()
        self.lxmert_vqa_no_lrp.eval()
        self.model = self.lxmert_vqa

        self.vqa_dataset = vqa_data.VQADataset(splits="valid")

    def forward(self, item):
        image_file_path = self.COCO_VAL_PATH + item['img_id'] + '.jpg'
        self.image_file_path = image_file_path
        self.image_id = item['img_id']
        # run frcnn
        images, sizes, scales_yx = self.image_preprocess(image_file_path)
        output_dict = self.frcnn(
            images,
            sizes,
            scales_yx=scales_yx,
            padding="max_detections",
            max_detections= self.frcnn_cfg.max_detections,
            return_tensors="pt"
        )
        inputs = self.lxmert_tokenizer(
            item['sent'],
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        self.question_tokens = self.lxmert_tokenizer.convert_ids_to_tokens(inputs.input_ids.flatten())
        self.text_len = len(self.question_tokens)
        # Very important that the boxes are normalized
        normalized_boxes = output_dict.get("normalized_boxes")
        features = output_dict.get("roi_features")
        self.image_boxes_len = features.shape[1]
        self.bboxes = output_dict.get("boxes")
        self.output = self.lxmert_vqa(
            input_ids=inputs.input_ids.to("cuda"),
            attention_mask=inputs.attention_mask.to("cuda"),
            visual_feats=features.to("cuda"),
            visual_pos=normalized_boxes.to("cuda"),
            token_type_ids=inputs.token_type_ids.to("cuda"),
            return_dict=True,
            output_attentions=False,
        )
        return self.output

    def forward_prune(self, item, text_prune, image_prune):
        image_file_path = self.COCO_VAL_PATH + item['img_id'] + '.jpg'
        # run frcnn
        images, sizes, scales_yx = self.image_preprocess(image_file_path)
        output_dict = self.frcnn(
            images,
            sizes,
            scales_yx=scales_yx,
            padding="max_detections",
            max_detections= self.frcnn_cfg.max_detections,
            return_tensors="pt"
        )
        inputs = self.lxmert_tokenizer(
            item['sent'],
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        # Very important that the boxes are normalized
        normalized_boxes = output_dict.get("normalized_boxes")
        features = output_dict.get("roi_features")
        output = self.lxmert_vqa_no_lrp(
            input_ids=inputs.input_ids.to("cuda"),
            attention_mask=inputs.attention_mask.to("cuda"),
            visual_feats=features.to("cuda"),
            visual_pos=normalized_boxes.to("cuda"),
            token_type_ids=inputs.token_type_ids.to("cuda"),
            return_dict=True,
            output_attentions=False,
            text_head_prune=text_prune,
            image_head_prune=image_prune,
        )

        answer = self.vqa_answers[output.question_answering_score.argmax()]
        return answer

def main():
    model = ModelUsage(args.COCO_path)
    vis_gen = VisGen(model, save_visualization=True)
    head_prune = HeadPrune(model)
    vqa_dataset = vqa_data.VQADataset(splits="valid")

    items = vqa_dataset.data
    random.seed(args.seed)
    r = list(range(len(items)))
    random.shuffle(r)
    pert_steps = [0, 0.4, 0.6, 0.9]
    for i in r[:args.num_samples]:
        scores_text, scores_image, _, _ = head_prune.generate_ours(items[i])

        num_text_layers = len(scores_text)
        num_image_layers = len(scores_image)
        num_text_heads = scores_text[0].shape[0]
        num_image_heads = scores_image[0].shape[0]
        tot_num = num_text_layers * num_text_heads + num_image_layers * num_image_heads

        scores_text = torch.stack(scores_text)
        scores_image = torch.stack(scores_image)
        joint_scores = torch.cat([scores_text, scores_image]).to("cuda")
        print(f"Question: {model.question_tokens}")

        for step_idx, step in enumerate(pert_steps):
            # find top step heads
            curr_num = int((1 - step) * tot_num)
            joint_scores = joint_scores.flatten()
            _, top_heads = joint_scores.topk(k=curr_num, dim=-1)
            heads_indicator = torch.zeros_like(joint_scores)
            heads_indicator[top_heads] = 1
            heads_indicator = heads_indicator.reshape(num_text_layers + num_image_layers, num_image_heads)
            heads_indicator_text = heads_indicator[:num_text_layers, :]
            heads_indicator_image = heads_indicator[num_text_layers:, :]
            vis_gen.generate_ours(items[i], head_prune_text=heads_indicator_text, head_prune_image=heads_indicator_image, method_name="prune_{0}".format(step*100))
            vqa_answer_txt = model.forward_prune(items[i], heads_indicator_text, heads_indicator_image)
            print(f"Answer is '{vqa_answer_txt}'")

if __name__ == "__main__":
    main()
