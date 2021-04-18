from lxmert.lxmert.src.tasks import vqa_data
from lxmert.lxmert.src.modeling_frcnn import GeneralizedRCNN
import lxmert.lxmert.src.vqa_utils as utils
from lxmert.lxmert.src.processing_image import Preprocess
from transformers import LxmertTokenizer
from lxmert.lxmert.src.huggingface_lxmert import LxmertForQuestionAnswering
from lxmert.lxmert.src.lxmert_lrp import LxmertForQuestionAnswering as LxmertForQuestionAnsweringLRP
from tqdm import tqdm
from lxmert.lxmert.src.ExplanationGenerator import HeadPrune, LayerPrune
import random
from lxmert.lxmert.src.param import args
import torch

OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"
VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"

class ModelPert:
    def __init__(self, COCO_val_path, use_lrp=False):
        self.COCO_VAL_PATH = COCO_val_path
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

        self.pert_steps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.pert_acc = [0] * len(self.pert_steps)

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

    def perturbation(self, item, scores_text, scores_image, is_positive_pert=False, heads=True):
        image_file_path = self.COCO_VAL_PATH + item['img_id'] + '.jpg'
        # run frcnn
        images, sizes, scales_yx = self.image_preprocess(image_file_path)
        output_dict = self.frcnn(
            images,
            sizes,
            scales_yx=scales_yx,
            padding="max_detections",
            max_detections=self.frcnn_cfg.max_detections,
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

        num_text_layers = len(scores_text)
        num_image_layers = len(scores_image)

        # create a 1D tensor of all scores
        # the initial shape of the scores differs from heads to layers
        if heads == True:
            num_text_heads = scores_text[0].shape[0]
            num_image_heads = scores_image[0].shape[0]
            tot_num = num_text_layers * num_text_heads + num_image_layers * num_image_heads

            scores_text = torch.stack(scores_text)
            scores_image = torch.stack(scores_image)

            joint_scores = torch.cat([scores_text, scores_image]).to("cuda")
        else:
            tot_num = num_text_layers + num_image_layers

            joint_scores = torch.tensor(scores_text + scores_image) #In layers-pruning these are two lists so we use '+'

        if is_positive_pert: # if positive pert then flip scores
            joint_scores = joint_scores * (-1)

        with torch.no_grad():
            for step_idx, step in enumerate(self.pert_steps):
                # find top step heads
                curr_num = int((1 - step) * tot_num)
                joint_scores = joint_scores.flatten()
                _, top_heads = joint_scores.topk(k=curr_num, dim=-1)
                heads_indicator = torch.zeros_like(joint_scores)
                heads_indicator[top_heads] = 1

                # reshape the binary vector, in layers num_image_heads is irrelevant so we use 1
                if heads == True:
                    heads_indicator = heads_indicator.reshape(num_text_layers + num_image_layers, num_image_heads)
                else:
                    heads_indicator = heads_indicator.reshape(num_text_layers + num_image_layers, 1)

                # split binary vector to text heads and image heads
                heads_indicator_text = heads_indicator[:num_text_layers, :]
                heads_indicator_image = heads_indicator[num_text_layers:, :]

                output = self.lxmert_vqa_no_lrp(
                    input_ids=inputs.input_ids.to("cuda"),
                    attention_mask=inputs.attention_mask.to("cuda"),
                    visual_feats=features.to("cuda"),
                    visual_pos=normalized_boxes.to("cuda"),
                    token_type_ids=inputs.token_type_ids.to("cuda"),
                    return_dict=True,
                    output_attentions=False,
                    text_head_prune=heads_indicator_text,
                    image_head_prune=heads_indicator_image,
                )

                answer = self.vqa_answers[output.question_answering_score.argmax()]
                accuracy = item["label"].get(answer, 0)
                self.pert_acc[step_idx] += accuracy

        return self.pert_acc

def main(args):
    model_pert = ModelPert(args.COCO_path, use_lrp=True)
    if args.prune_type == "head": # is head pruning or layer pruning
        gen = HeadPrune(model_pert)
    else:
        gen = LayerPrune(model_pert)
    vqa_dataset = vqa_data.VQADataset(splits="valid")
    method_name = args.method

    items = vqa_dataset.data
    random.seed(args.seed)
    r = list(range(len(items)))
    random.shuffle(r)
    pert_samples_indices = r[:args.num_samples]
    iterator = tqdm([vqa_dataset.data[i] for i in pert_samples_indices])

    test_type = "positive" if args.is_positive_pert is True else "negative"
    modality = "text" if args.is_text_pert else "image"
    print("running {0} {1} prune pert test for {2} modality with method {3}".format(test_type, args.prune_type, modality, args.method))

    for index, item in enumerate(iterator):
        grad_scores_text, grad_scores_image, cam_scores_text, cam_scores_image = gen.generate_ours(item)
        scores_text = grad_scores_text if method_name == "ours" else cam_scores_text
        scores_image = grad_scores_image if method_name == "ours" else cam_scores_image
        curr_pert_result = model_pert.perturbation(item, scores_text, scores_image, args.is_positive_pert, heads=(args.prune_type == "head"))

        curr_pert_result = [round(res / (index + 1) * 100, 2) for res in curr_pert_result]
        iterator.set_description("Acc: {}".format(curr_pert_result))

if __name__ == "__main__":
    main(args)
