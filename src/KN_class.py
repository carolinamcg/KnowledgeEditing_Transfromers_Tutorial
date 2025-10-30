import torch
import torch.nn.functional as F
import time
import os
import numpy as np
import json
import jsonlines
import random
from collections import Counter

from utils import example2feature, pos_str2list, pos_list2str, stat

EPS = 1e-10


class AttributionScores:
    def __init__(
        self,
        model,
        tokenizer,
        max_seq_length,
        device,
        get_ig_gold,
        get_base,
        get_ig_pred,
        get_pred,
        batch_size=20,
        num_batch=1,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.device = device
        self.get_ig_gold = get_ig_gold
        self.get_base = get_base
        self.get_ig_pred = get_ig_pred
        self.get_pred = get_pred
        self.batch_size = batch_size
        self.num_batch = num_batch
        self.num_approx_steps = batch_size * num_batch

    def get_model_inputs(self, eval_example):
        eval_features, tokens_info = example2feature(
            eval_example, self.max_seq_length, self.tokenizer
        )
        # convert features to long type tensors
        baseline_ids, input_ids, input_mask, segment_ids = (
            eval_features["baseline_ids"],
            eval_features["input_ids"],
            eval_features["input_mask"],
            eval_features["segment_ids"],
        )
        baseline_ids = (
            torch.tensor(baseline_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        )
        input_ids = (
            torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        )
        input_mask = (
            torch.tensor(input_mask, dtype=torch.long).unsqueeze(0).to(self.device)
        )
        segment_ids = (
            torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        )
        return baseline_ids, input_ids, input_mask, segment_ids, tokens_info

    def inference(self, input_ids, input_mask, segment_ids, tgt_pos, tgt_label_id=None):
        # target layer doesn't matter here, we just wan't to get the pred label
        _, logits = self.model(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
            tgt_pos=tgt_pos,
            tgt_layer=0,
        )  # (1, n_vocab)
        base_pred_prob = F.softmax(logits, dim=-1)  # (1, n_vocab)
        if tgt_label_id is None:
            return base_pred_prob
        else:
            ori_tgt_prob = base_pred_prob[
                0, tgt_label_id
            ]  # scalar = out prob of the tgt_label for the tgt__pos/masked token
            ori_tgt_rank = (
                logits.squeeze().argsort(descending=True).argsort()[tgt_label_id] + 1
            )  # position that tgt_label_id occupies in the rank of high to lost logits/output probs
            ori_pred_prob, ori_pred_label_id = base_pred_prob[0].max(dim=-1)
            return (
                ori_tgt_prob,
                ori_tgt_rank,
                ori_pred_prob,
                ori_pred_label_id,
                base_pred_prob,
            )

    def predict_example(self, eval_example):
        _, input_ids, input_mask, segment_ids, tokens_info = self.get_model_inputs(
            eval_example
        )

        # record [MASK]'s position
        tgt_pos = tokens_info["tokens"].index(
            "[MASK]"
        )  # target pos is the index of the masked token

        base_pred_prob = self.inference(
            input_ids, input_mask, segment_ids, tgt_pos
        )  # (1, n_vocab)
        # pred_label_id = int(torch.argmax(base_pred_prob[0, :])) #because its the logits only for the target pos
        pred_prob, pred_label_id = base_pred_prob[0].max(dim=-1)
        pred_label = self.tokenizer.convert_ids_to_tokens(
            pred_label_id.item()
        )  # predicted token
        return pred_label_id, pred_label, pred_prob

    def convert_to_triplet_ig(self, ig_list):
        ig_triplet = []
        ig = np.array(ig_list)  # 12, 3072
        max_ig = ig.max()
        for i in range(ig.shape[0]):  # for each layer
            for j in range(ig.shape[1]):  # for each neuron
                if ig[i][j] >= max_ig * 0.1:
                    ig_triplet.append(
                        [i, j, ig[i][j]]
                    )  # just keep the neurons that have attribution scores > 0.1 * max
        return ig_triplet

    def scaled_input(self, emb, batch_size, num_batch):
        """
        For the  Riemann approximation with num_points approximated steps (20 in the article -> batch_size=20, num_batch=1)
        step[0] = original w values (wi) / num_points (m)
        res = list of all gradually changed weights, increasing (1/20)*wi at each index
        """
        # emb: (1, ffn_size)
        baseline = torch.zeros_like(emb)  # (1, ffn_size)

        num_points = batch_size * num_batch
        step = (emb - baseline) / num_points  # (1, ffn_size)

        res = torch.cat(
            [torch.add(baseline, step * i) for i in range(num_points)], dim=0
        )  # (num_points, ffn_size)
        # Left-sided Riemans Approximation: i=[0, 19], instead of [1, 20]
        # Incongruent with what they say in the artcile, but I'll allow it, to consider the impact of setting all
        # these neurons to 0 :)
        return res, step[0]

    def compute_attribution_scores(
        self,
        scaled_weights,
        weights_step,
        input_ids,
        input_mask,
        segment_ids,
        tgt_pos,
        tgt_layer,
        tgt_label,
    ):
        ig = None
        for batch_idx in range(self.num_batch):  # self.num_batch=1 -> batch_idx=0
            batch_weights = scaled_weights[
                batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
            ]  # scaled_weights[0:1*20]
            _, grad = self.model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                tgt_pos=tgt_pos,
                tgt_layer=tgt_layer,
                tmp_score=batch_weights,
                tgt_label=tgt_label,
            )  # (batch, n_vocab), (batch, ffn_size) -> batch=20=num_points/approximate steps
            grad = grad.sum(
                dim=0
            )  # (ffn_size) -> summing gradients of all approximate steps
            ig = grad if ig is None else torch.add(ig, grad)  # (ffn_size)
        ig = (
            ig * weights_step
        )  # (ffn_size) = attribution scores for this target/masked token, wrt to model's prediction
        return ig

    def get_attr_scores(
        self,
        relations,
        eval_bag_list_perrel,
        output_dir,
        output_prefix,
        target_layers_idxs=None,
        th_entity_pairs=10,
        th_prompts=60,
    ):
        """
        For each relation in relations, feed the different entity pairs and prompts to the pre-trained model (self.model)
        and get attribution scores for its target_layers_idxs's FFN neurons.

        param relations (list): wanted relations as named in the evaluation dataset file.
        param eval_bag_list_perrel (dict): each key is a relation and teh corresponding a value is a list with
                        n head-tail entity pairs, where each of these pairs has m different prompts.
        param output_dir (str):  path to save the computed attribution scores for each relation.
        param output_prefix (str):  prefix to give to the save files' names.
        param target_layers_idxs (list): indexes of the layers for which to compute the attribution scores
                                        (default: None -> all model's layers)
        param  th_entity_pairs (int): max number of entity pairs per relation (default: None -> all)
        param  th_prompts (int): max number of distinct prompts per entity pair (default: None -> all)
        """

        target_layers_idxs = (
            self.model.bert.config.num_hidden_layers
            if target_layers_idxs is None
            else target_layers_idxs
        )

        used_bags = {}  # to save the relations and eval examples that we actually used

        for relation in relations:
            eval_bag_list = eval_bag_list_perrel[relation]
            used_bags[relation] = []

            tic = time.perf_counter()  # record running time
            with jsonlines.open(
                os.path.join(
                    output_dir, output_prefix + "-" + relation + ".rlt" + ".jsonl"
                ),
                "w",
            ) as fw:
                for bag_idx, eval_bag in enumerate(eval_bag_list):
                    # go through each pair of head-tail entities in that relation
                    if th_entity_pairs is not None and bag_idx >= th_entity_pairs:
                        print(f"Stopped computing entity pairs at {bag_idx}")
                        break

                    used_bags[relation].append([])  # append a list for the entity pair

                    res_dict_bag = (
                        []
                    )  # appends, for each pair, a list [token_info, rest_dict]
                    # each row corresponds to the each pair. The second column has teh attribution scores for each pair

                    for ex_idx, eval_example in enumerate(eval_bag):
                        # go throug each prompt of that pair, already with the tail as [MASK]
                        if th_prompts is not None and ex_idx >= th_prompts:
                            print(f"Stopped computing examples at {ex_idx}")
                            break

                        (
                            _,
                            input_ids,
                            input_mask,
                            segment_ids,
                            tokens_info,
                        ) = self.get_model_inputs(eval_example)

                        # record [MASK]'s position
                        tgt_pos = tokens_info["tokens"].index(
                            "[MASK]"
                        )  # target pos is the index of the masked token

                        # record various results
                        res_dict = {
                            "pred": [],  # final predict logits (1, n_vocab)
                            "ig_pred": [],  # len=12; attribution scores for the masked token per layers (1, ffn_size)
                            "ig_gold": [],  # same as 'ig_pred', but with gradients computed from gt_label's logit
                            "base": [],
                        }

                        base_pred_prob = self.inference(
                            input_ids, input_mask, segment_ids, tgt_pos
                        )  # (1, n_vocab)
                        if (
                            self.get_pred
                        ):  # SAVE original pred probabilities distribution
                            res_dict["pred"].append(base_pred_prob.tolist())
                        pred_label = int(
                            torch.argmax(base_pred_prob[0, :])
                        )  # because its the logits only for the target pos
                        tokens_info["pred_obj"] = self.tokenizer.convert_ids_to_tokens(
                            pred_label
                        )  # predicted token
                        gold_label = self.tokenizer.convert_tokens_to_ids(
                            tokens_info["gold_obj"]
                        )  # token idx of the GT label

                        # append the prompt/eval example in the respective entity pair's list
                        new_example = []
                        new_example.extend(eval_example)
                        new_example.append(tokens_info["pred_obj"])
                        used_bags[relation][bag_idx].append(new_example)

                        # COMPUTE attr scores
                        for tgt_layer in range(target_layers_idxs):
                            ## GET scaled layer "weights" wi to compute the Riemann Approximation
                            ffn_weights, _ = self.model(
                                input_ids=input_ids,
                                attention_mask=input_mask,
                                token_type_ids=segment_ids,
                                tgt_pos=tgt_pos,
                                tgt_layer=tgt_layer,
                            )  # (1, ffn_size), (1, n_vocab)
                            # ffn_weights = output of GELU(Linear(x)) of tgt_layer's FFN module
                            scaled_weights, weights_step = self.scaled_input(
                                ffn_weights, self.batch_size, self.num_batch
                            )  # (num_points, ffn_size), (ffn_size)
                            scaled_weights.requires_grad_(True)

                            # integrated grad at the pred label for each layer
                            if self.get_ig_pred:
                                ig_pred = self.compute_attribution_scores(
                                    scaled_weights,
                                    weights_step,
                                    input_ids,
                                    input_mask,
                                    segment_ids,
                                    tgt_pos,
                                    tgt_layer,
                                    pred_label,
                                )
                                res_dict["ig_pred"].append(ig_pred.tolist())

                            # integrated grad at the gold label for each layer
                            # exactly the same process as before, but with the target label being the gt one
                            # this just changes the computation of the gradients
                            if self.get_ig_gold:
                                ig_gold = self.compute_attribution_scores(
                                    scaled_weights,
                                    weights_step,
                                    input_ids,
                                    input_mask,
                                    segment_ids,
                                    tgt_pos,
                                    tgt_layer,
                                    gold_label,
                                )
                                res_dict["ig_gold"].append(ig_gold.tolist())

                            # base ffn_weights for each layer
                            if self.get_base:
                                res_dict["base"].append(
                                    ffn_weights.squeeze().tolist()
                                )  # appends the original pretrained layer's ffn_weights

                        # RIGHT NOW IN res_dict:
                        #'ig_pred': [], # len=12; attribution scores for the masked token per layers (1, ffn_size)
                        #'ig_gold': [], # same as 'ig_pred', but with gradients computed from gt_label's logit
                        # 'base': [], len=12; each contains the original pretrained ffn_weights for that layer
                        if self.get_ig_gold:
                            res_dict["ig_gold"] = self.convert_to_triplet_ig(
                                res_dict["ig_gold"]
                            )
                        if self.get_base:
                            res_dict["base"] = self.convert_to_triplet_ig(
                                res_dict["base"]
                            )
                        # RIGHT NOW IN res_dict:
                        #'ig_pred': [], # len=12; triplets (layer_id, neuron_is, attribution score), only for the relevant neurons
                        #'ig_gold': [], # same as 'ig_pred', but with gradients computed from gt_label's logit
                        # 'base': [], #same as the previous, but these are the original weights, not attribution scores computed w/ gradients
                        res_dict_bag.append([tokens_info, res_dict])

                    if len(res_dict_bag) > 1:
                        fw.write(res_dict_bag)  # SAVING IN THE OUTPUT FOLDER

            # record running time
            toc = time.perf_counter()
            print(
                f"***** Relation: {relation} evaluated. Costing time: {toc - tic:0.4f} seconds *****"
            )

        with jsonlines.open(
            os.path.join(output_dir, output_prefix + "data_used_bags.json"), "w"
        ) as fb:
            fb.write(used_bags)
        fb.close()


class KN(AttributionScores):
    def __init__(
        self,
        model,
        tokenizer,
        max_seq_length,
        device,
        get_ig_gold,
        get_base,
        get_ig_pred,
        get_pred,
        batch_size=20,
        num_batch=1,
        mode_ratio_rel=0.1,
        threshold_ratio=[0.2, 0.5],
    ) -> None:
        super().__init__(
            model,
            tokenizer,
            max_seq_length,
            device,
            get_ig_gold,
            get_base,
            get_ig_pred,
            get_pred,
            batch_size,
            num_batch,
        )

        self.threshold_ratio = threshold_ratio
        # list of th for neuron values computed by
        # attr scores and for the base original ones ("ffn_weights")
        self.mode_ratio_rel = mode_ratio_rel

    def re_filter(self, metric_triplets, metric):
        """
        For each prompt, retain the neurons with attribution scores greater
        than the attribution threshold t, obtaining the coarse
        set of knowledge neurons.
        """
        metric_max = -999
        if metric == "base":
            th = self.threshold_ratio[1]
        else:
            th = self.threshold_ratio[0]
        for i in range(len(metric_triplets)):
            metric_max = max(
                metric_max, metric_triplets[i][2]
            )  # get max attribution score for this prompt input
        metric_triplets = [
            triplet for triplet in metric_triplets if triplet[2] >= metric_max * th
        ]  # retain only the
        # knowledge neurons that have scores > that 0.2*max
        return metric_triplets

    def parse_kn(self, pos_cnt, tot_num, mode_ratio, min_threshold=0):
        """
        Considering all the coarse sets together, retain the knowledge neurons
        shared by more than mode_ratio% prompts.

        tot_num = num of prompts
        min_threshold = min % of prompts that can be considered
        """
        mode_threshold = tot_num * mode_ratio
        mode_threshold = max(mode_threshold, min_threshold)
        kn_bag = []
        for pos_str, cnt in pos_cnt.items():
            if cnt >= mode_threshold:
                kn_bag.append(
                    pos_str2list(pos_str)
                )  # retains the positions of the knowledge neurons shared by mode_threshold% prompts
                # Possible problem,: might retain all different positions, if mode_threshold <= 1
        return kn_bag

    def analysis_file(self, rlts_dir, filename, mode_ratio_bag=0.7, metric="ig_gold"):
        rel = filename.split(".")[0].split("-")[-1]  # realation ID, e.g. P101
        print(
            f"===========> parsing important position in {rel}-{metric}..., mode_ratio_bag={mode_ratio_bag}"
        )

        rlts_bag_list = []
        with open(os.path.join(rlts_dir, filename), "r") as fr:
            for rlts_bag in jsonlines.Reader(fr):
                rlts_bag_list.append(rlts_bag)

        ave_kn_num = 0  # final average number of knowledge neurons (kn) per head-tail entities in this relation

        # REFINING KNOWLEDGE NEURONS (removing false positives)

        kn_bag_list = []  # list of kn positions for this relation
        # get imp pos by bag_ig: commmon KNs per HEAD-TAIL PAIR
        for bag_idx, rlts_bag in enumerate(
            rlts_bag_list
        ):  # go through each pair of head-tail entities in this relation
            pos_cnt_bag = Counter()
            for rlt in rlts_bag:  # go throug each prompt of that pair
                res_dict = rlt[1]  # rl[0] = token_info
                metric_triplets = self.re_filter(
                    res_dict[metric], metric
                )  # filtered triplets: only the ones with high enough attribution score
                for metric_triplet in metric_triplets:
                    pos_cnt_bag.update(
                        [pos_list2str(metric_triplet[:2])]
                    )  # counts the occurence of layer_id@neuron_id
            kn_bag = self.parse_kn(
                pos_cnt_bag, len(rlts_bag), mode_ratio_bag, 3
            )  # retains the common kns across all prompts of that heal-tail pair
            ave_kn_num += len(kn_bag)
            kn_bag_list.append(kn_bag)

        ave_kn_num /= len(rlts_bag_list)  # total no of kn / no of entity pairs

        # get imp pos by rel_ig: commmon KNs in THE WHOLE RELATION's SET
        pos_cnt_rel = Counter()
        for kn_bag in kn_bag_list:  # for each head-tail entities pair
            for kn in kn_bag:  # kn positions retained for each head-tail entities pair
                pos_cnt_rel.update([pos_list2str(kn)])
        kn_rel = self.parse_kn(
            pos_cnt_rel, len(kn_bag_list), self.mode_ratio_rel
        )  # retains the common kns across different entity pairs
        # mode_ratio_rel=0.1

        return ave_kn_num, kn_bag_list, kn_rel

    def get_kn(
        self,
        rlts_dir,
        filename,
        kn_dir,
        rel,
        prefix="",
        mode_ratio_bag=0.7,
        metric="ig_gold",
    ):
        # threshold_ratio = attribution th
        # mode_ratio_bag = initial value for the refining th
        for _ in range(6):
            ave_kn_num, kn_bag_list, kn_rel = self.analysis_file(
                rlts_dir, filename, mode_ratio_bag=mode_ratio_bag, metric=metric
            )
            # we increase or decrease mode_ratio_bag by 0.05 at a time until the average number of knowledge
            # neurons lies in [2, 5].
            if ave_kn_num < 2:
                mode_ratio_bag -= 0.05
            if ave_kn_num > 5:
                mode_ratio_bag += 0.05
            if ave_kn_num >= 2 and ave_kn_num <= 5:
                break

        stat(kn_bag_list, f"{prefix}kn_bag", rel)
        stat(kn_rel, f"{prefix}kn_rel", rel)
        with open(
            os.path.join(kn_dir, f"{prefix}kn_bag-{rel}.json"), "w"
        ) as fw:  # kn positions per entity pair in the relation
            json.dump(kn_bag_list, fw, indent=2)
        with open(
            os.path.join(kn_dir, f"{prefix}kn_rel-{rel}.json"), "w"
        ) as fw:  # kn positions for the whole relation's set
            json.dump(kn_rel, fw, indent=2)

        return kn_bag_list, kn_rel

    def get_kn_from_files(self, rlts_dir, kn_dir, mode_ratio_bag=0.7, metric="ig_gold"):
        kn_pos_perrel = {}
        for filename in os.listdir(rlts_dir):
            if filename.endswith(
                ".rlt.jsonl"
            ):  # file with the attribution scores savedin 1_analyse_mlm.py
                rel = filename.split(".")[0].split("-")[-1]
                kn_bag_list, kn_rel = self.get_kn(
                    rlts_dir,
                    filename,
                    kn_dir,
                    rel,
                    mode_ratio_bag=mode_ratio_bag,
                    metric=metric,
                )
                ## REPEAT EXACTLY THE SAME PROCESS but for the base ffn_weights stored in res_dict['base']
                base_kn_bag_list, base_kn_rel = self.get_kn(
                    rlts_dir,
                    filename,
                    kn_dir,
                    rel,
                    prefix="base_",
                    mode_ratio_bag=mode_ratio_bag,
                    metric="base",
                )

                kn_pos_perrel[rel] = {}
                kn_pos_perrel[rel]["kn_bag_list"] = kn_bag_list
                kn_pos_perrel[rel]["kn_rel"] = kn_rel
                kn_pos_perrel[rel]["base_kn_bag_list"] = base_kn_bag_list
                kn_pos_perrel[rel]["base_kn_rel"] = base_kn_rel

        return kn_pos_perrel

    def forward(
        self,
        relations,
        eval_bag_list_perrel,
        output_dir,
        output_prefix,
        target_layers_idxs=None,
        th_entity_pairs=10,
        th_prompts=60,
        mode_ratio_bag=0.7,
    ):
        self.get_attr_scores(
            relations,
            eval_bag_list_perrel,
            output_dir,
            output_prefix,
            target_layers_idxs,
            th_entity_pairs,
            th_prompts,
        )  # get the json files with all atribution scores
        # for the target relations and layers

        kn_dir = os.path.join(output_dir, "kn/")
        os.makedirs(kn_dir, exist_ok=True)
        kn_pos_perrel = self.get_kn_from_files(
            output_dir, kn_dir, mode_ratio_bag=mode_ratio_bag, metric="ig_gold"
        )
        return kn_pos_perrel


class EditKnowledge(KN):
    def __init__(
        self,
        model,
        tokenizer,
        max_seq_length,
        device,
        get_ig_gold,
        get_base,
        get_ig_pred,
        get_pred,
        batch_size=20,
        num_batch=1,
        mode_ratio_rel=0.1,
        threshold_ratio=0.2,
        norm_lambda1=1,
        norm_lambda2=8,
        mode_ratio_rel_edit=0.3,
    ) -> None:
        super().__init__(
            model,
            tokenizer,
            max_seq_length,
            device,
            get_ig_gold,
            get_base,
            get_ig_pred,
            get_pred,
            batch_size,
            num_batch,
            mode_ratio_rel,
            threshold_ratio,
        )

        self.norm_lambda1 = norm_lambda1
        self.norm_lambda2 = norm_lambda2
        self.mode_ratio_rel_edit = mode_ratio_rel_edit

        self.results = {
            "success_updated": 0,
            "success_updated_5": 0,
            "ori_changed": 0,
            "tgt_prob_inc": 0,
            "tgt_prob_inc_ratio": 0,
            "tgt_ori_rank": 0,
            "tgt_new_rank": 0,
            "tgt_rank_inc": 0,
            "ori_inter_log_ppl": [],
            "ori_inter_log_ppl_pred": [],
            "ori_inner_log_ppl": [],
            "ori_inner_log_ppl_pred": [],
            "new_inter_log_ppl": [],
            "new_inter_log_ppl_pred": [],
            "new_inner_log_ppl": [],
            "new_inner_log_ppl_pred": [],
        }

        self.average_results = {
            "success_updated": [],
            "success_updated_5": [],
            "ori_changed": [],
            "tgt_prob_inc": [],
            "tgt_prob_inc_ratio": [],
            "tgt_ori_rank": [],
            "tgt_new_rank": [],
            "tgt_rank_inc": [],
            "other_ppl_inc": [],
            "ori_inter_log_ppl": [],
            "ori_inter_log_ppl_pred": [],
            "ori_inner_log_ppl": [],
            "ori_inner_log_ppl_pred": [],
            "new_inter_log_ppl": [],
            "new_inter_log_ppl_pred": [],
            "new_inner_log_ppl": [],
            "new_inner_log_ppl_pred": [],
        }

    def reset_results(self):
        self.results = {
            "success_updated": 0,
            "success_updated_5": 0,
            "ori_changed": 0,
            "tgt_prob_inc": 0,
            "tgt_prob_inc_ratio": 0,
            "tgt_ori_rank": 0,
            "tgt_new_rank": 0,
            "tgt_rank_inc": 0,
            "ori_inter_log_ppl": [],
            "ori_inner_log_ppl": [],
            "new_inter_log_ppl": [],
            "new_inner_log_ppl": [],
        }

    def reset_average_results(self):
        self.average_results = {
            "success_updated": [],
            "success_updated_5": [],
            "ori_changed": [],
            "tgt_prob_inc": [],
            "tgt_prob_inc_ratio": [],
            "tgt_ori_rank": [],
            "tgt_new_rank": [],
            "tgt_rank_inc": [],
            "other_ppl_inc": [],
            "ori_inter_log_ppl": [],
            "ori_inter_log_ppl_pred": [],
            "ori_inner_log_ppl": [],
            "ori_inner_log_ppl_pred": [],
            "new_inter_log_ppl": [],
            "new_inter_log_ppl_pred": [],
            "new_inner_log_ppl": [],
            "new_inner_log_ppl_pred": [],
        }

    def select_kn(self, bag_idx, kn_bag_list):
        """
        Returns a filtered list of kn for the entity in question (new_kn_bag), where the kn
        that are common across different pairs in the relation are removed. So, it returns only
        the kn positions that are unique to this enity pair.

        param kn_bag_list (ist): list of kn for each entity pair in the whole relation (across all pairs)
        param bag_idx (int): index of the enitiy pair in question
        """
        kn_bag = kn_bag_list[bag_idx]  # previously detected kn for that entity pair
        kn_counter = Counter()
        for kn in kn_bag:
            kn_counter.update([pos_list2str(kn)])  # count the no of ocurrences
            # of different kn positions for that entity pair

        for i, tmp_kn_bag in enumerate(
            kn_bag_list
        ):  # for each entity pair in the relation
            if i != bag_idx:
                for kn in tmp_kn_bag:
                    str_kn = pos_list2str(kn)
                    if str_kn not in kn_counter:
                        continue
                    kn_counter.update(
                        [pos_list2str(kn)]
                    )  # for the kn positions of this entity pair,
                    # counts also their occurence kn in other pairs of the same relation
        new_kn_bag = []
        for k, v in kn_counter.items():
            if (
                v > len(kn_bag_list) * self.mode_ratio_rel_edit
            ):  # happens in more that 30% of all entity pairs (paper's original: 10%)
                # for the purpose of teh seminar, I use less entties per rel. So, I increased this %
                # to increase this th. Otherwise some specific kn would not be included in new_kn_bag
                continue
            new_kn_bag.append(pos_str2list(k))
        # random kn
        # for i in range(len(new_kn_bag)):
        #     new_kn_bag[i] = [random.randint(0, 11), random.randint(0, 3071)]
        return new_kn_bag  # these are the kn that are SPECIFIC to this entity pair and not to the whole relational statement

    def recover_neurons_knowledge(
        self, kn_bag, ori_pred_emb, tgt_emb, lambda_list_1, lambda_list_2
    ):
        with torch.no_grad():
            for i, (layer, pos) in enumerate(kn_bag):
                # changing the weights respective COLUMN (VALUE) of that kn (768, 1)
                self.model.bert.encoder.layer[layer].output.dense.weight[:, pos] += (
                    ori_pred_emb * lambda_list_1[i]
                )
                self.model.bert.encoder.layer[layer].output.dense.weight[:, pos] -= (
                    tgt_emb * lambda_list_2[i]
                )

    def edit_neurons_knowledge(
        self, kn_bag, ori_pred_emb, tgt_emb, lambda_list_1, lambda_list_2
    ):
        print(" ++++++++++++++++++ EDITING KN ++++++++++++++++++")
        with torch.no_grad():
            for i, (layer, pos) in enumerate(kn_bag):
                print(
                    f" kn at layer {layer}, position {pos}, average weights value {self.model.bert.encoder.layer[layer].output.dense.weight[:, pos].mean().item()}"
                )
                # changing the weights respective COLUMN (VALUE) of that kn (768, 1)
                self.model.bert.encoder.layer[layer].output.dense.weight[:, pos] -= (
                    ori_pred_emb * lambda_list_1[i]
                )
                self.model.bert.encoder.layer[layer].output.dense.weight[:, pos] += (
                    tgt_emb * lambda_list_2[i]
                )
                print(
                    f"  ----> New edited  average weights value: {self.model.bert.encoder.layer[layer].output.dense.weight[:, pos].mean().item()}"
                )

    def get_edit_neurons_shits(self, kn_bag, ori_pred_label_id, tgt_label_id):
        ori_pred_emb = self.model.bert.embeddings.word_embeddings.weight[
            ori_pred_label_id.item()
        ]  # (768)
        tgt_emb = self.model.bert.embeddings.word_embeddings.weight[tgt_label_id]
        print(f"-- kn_num: {len(kn_bag)}")
        lambda_list_1 = []
        lambda_list_2 = []
        ori_pred_emb_norm = torch.norm(ori_pred_emb)
        tgt_emb_norm = torch.norm(tgt_emb)
        for layer, pos in kn_bag:
            # output.dense.weight shape = (768, 3072)
            value_norm = torch.norm(
                self.model.bert.encoder.layer[layer].output.dense.weight[:, pos]
            )  # kn value norm
            # (768, 1) -> norm -> value_norm (1)
            # output.dense.weight is the weight matrix if the Liner layer after the 1st half of FFN module
            # so, it's the matrix that multiplies with the so called "ffn_weights" where kn are identified
            # by the matmul operation, the kn pos of the tgt_token multiply with the exact same neuron position
            # in each row of output.dense.weight -> that kn ACTIVATES A COLUMN (VALUE) in output.dense.weight
            lambda_list_1.append(value_norm / ori_pred_emb_norm * self.norm_lambda1)
            lambda_list_2.append(value_norm / tgt_emb_norm * self.norm_lambda2)
        return ori_pred_emb, tgt_emb, lambda_list_1, lambda_list_2

    def compute_inner_log_ppl(self, eval_bag):
        log_ppl_list_gold, log_ppl_list_pred = [], []
        for idx, eval_example in enumerate(
            eval_bag
        ):  # for each prompt in that entity pair
            # convert features to long type tensors
            _, input_ids, input_mask, segment_ids, tokens_info = self.get_model_inputs(
                eval_example
            )
            # record [MASK]'s position
            tgt_pos = tokens_info["tokens"].index("[MASK]")
            gold_id = self.tokenizer.convert_tokens_to_ids(tokens_info["gold_obj"])
            ori_pred_id = self.tokenizer.convert_tokens_to_ids(tokens_info["pred_obj"])
            assert (
                ori_pred_id is not None
            ), "Predicted label in the saved eval example is None"

            base_pred_prob = self.inference(
                input_ids, input_mask, segment_ids, tgt_pos
            )  # (1, n_vocab)
            gold_prob = base_pred_prob[0][gold_id]  # prob of the GT label
            gold_inter_log_ppl = np.log(1.0 / (gold_prob.item() + EPS))
            log_ppl_list_gold.append(gold_inter_log_ppl)

            pred_prob = base_pred_prob[0][ori_pred_id]  # prob of the GT label
            pred_inter_log_ppl = np.log(1.0 / (pred_prob.item() + EPS))
            log_ppl_list_pred.append(pred_inter_log_ppl)

        return log_ppl_list_gold, log_ppl_list_pred

    def calculate_intra_inter_rel_ppl(
        self,
        eval_bag_list_perrel,
        rel,
        bag_idx,
        inner_rand_bag_idx_list=None,
        inter_rand_rel_list=None,
        inter_rand_bag_idx_list=None,
    ):
        # eval_bag_list_perrel = dict with the input prompts examples per relation

        # intra relation
        inner_log_ppl_list_gold, inner_log_ppl_list_pred = [], []
        eval_bag_list = eval_bag_list_perrel[rel]
        if inner_rand_bag_idx_list is None:
            inner_rand_bag_idx_list = []
            ## Choose 5 other entity pairs 1= from the previously selected (bag_idx)
            for i in range(5):
                rand_bag_idx = random.randint(0, len(eval_bag_list) - 1)
                while rand_bag_idx == bag_idx:
                    rand_bag_idx = random.randint(0, len(eval_bag_list) - 1)
                inner_rand_bag_idx_list.append(rand_bag_idx)

        for rand_bag_idx in inner_rand_bag_idx_list:
            eval_bag = eval_bag_list[
                rand_bag_idx
            ]  # example prompts for this entity pairs
            log_ppl_list_gold, log_ppl_list_pred = self.compute_inner_log_ppl(eval_bag)
            inner_log_ppl_list_gold.extend(log_ppl_list_gold)
            inner_log_ppl_list_pred.extend(log_ppl_list_pred)

        ## REPEAT THE SAME EXACT PPL COMPUTATION FOR DIFFERENT RELATIONS
        # inter relation
        inter_log_ppl_list_gold, inter_log_ppl_list_pred = [], []
        rels = list(eval_bag_list_perrel.keys())  # list of relations
        if inter_rand_rel_list is None:
            inter_rand_rel_list, inter_rand_bag_idx_list = [], []
            ## Choose 5 different relations from the current one and 1 entity pair per each
            for i in range(5):
                rand_rel = random.choice(rels)
                while rand_rel == rel:
                    rand_rel = random.choice(rels)
                rand_bag_idx = random.randint(
                    0, len(eval_bag_list_perrel[rand_rel]) - 1
                )  # choose 1 entitiy pair for that rand_rel
                inter_rand_rel_list.append(rand_rel)
                inter_rand_bag_idx_list.append(rand_bag_idx)

        for rand_rel, rand_bag_idx in zip(inter_rand_rel_list, inter_rand_bag_idx_list):
            eval_bag = eval_bag_list_perrel[rand_rel][
                rand_bag_idx
            ]  # entity pair for rand_rel
            log_ppl_list_gold_it, log_ppl_list_pred_it = self.compute_inner_log_ppl(
                eval_bag
            )
            inter_log_ppl_list_gold.extend(log_ppl_list_gold_it)
            inter_log_ppl_list_pred.extend(log_ppl_list_pred_it)

        del (
            log_ppl_list_gold,
            log_ppl_list_gold_it,
            log_ppl_list_pred,
            log_ppl_list_pred_it,
        )

        return (
            np.mean(inner_log_ppl_list_gold),
            np.mean(inner_log_ppl_list_pred),
            np.mean(inter_log_ppl_list_gold),
            np.mean(inter_log_ppl_list_pred),
            inner_rand_bag_idx_list,
            inter_rand_rel_list,
            inter_rand_bag_idx_list,
        )

    def edit(
        self,
        eval_bag_list_perrel,
        rel,
        bag_idx,
        tgt_ent,
        kn_bag_list=None,
        kn_dir="results/kn/",
    ):
        eval_bag = eval_bag_list_perrel[rel][bag_idx]

        if kn_bag_list is None:
            with open(os.path.join(kn_dir, f"kn_bag-{rel}.json"), "r") as fr:
                kn_bag_list = json.load(fr)

        kn_bag = self.select_kn(
            bag_idx, kn_bag_list
        )  # kn specific to bag_idx (an entity pair of rel)
        print(f"Unique kn for relation {rel}- entity {bag_idx}: {kn_bag}")

        if len(kn_bag) < 1:
            return "no_kn"

        rd_idx = (
            random.randint(1, len(eval_bag)) - 1
        )  # randomly choose a prompt inside the entity apir
        eval_example = eval_bag[rd_idx]
        _, input_ids, input_mask, segment_ids, tokens_info = self.get_model_inputs(
            eval_example
        )
        # record [MASK]'s position
        tgt_pos = tokens_info["tokens"].index("[MASK]")
        # record tgt_ent's label: the new tgt label we want to predict
        tgt_label_id = self.tokenizer.convert_tokens_to_ids(tgt_ent)

        (
            ori_tgt_prob,
            ori_tgt_rank,
            ori_pred_prob,
            ori_pred_label_id,
            _,
        ) = self.inference(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            tgt_pos=tgt_pos,
            tgt_label_id=tgt_label_id,
        )
        ori_pred_label = self.tokenizer.convert_ids_to_tokens(
            ori_pred_label_id.item()
        )  # predicted token

        if (
            ori_pred_label != eval_example[1]
        ):  # eval_example[1] = GT/gold label for this example/prompt
            print("Model is predicting the incorrect_answer for this example.")

        # ori pred label, ori pred prob, tgt label, ori tgt prob
        print(
            f"{rel}-{bag_idx}-{rd_idx}, # No of Kneurons: {len(kn_bag)},",
            "example:",
            eval_example[0],
            "gold:",
            eval_example[1],
        )
        print(
            "============================== ori ========================================="
        )
        print(
            f"ori pred label: {ori_pred_label}, ori pred label prob: {ori_pred_prob:.8}"
        )
        print(f"tgt label: {tgt_ent}, tgt prob: {ori_tgt_prob:.8}")

        ###  ============================== EDIT knowledge ==============================
        (
            ori_pred_emb,
            tgt_emb,
            lambda_list_1,
            lambda_list_2,
        ) = self.get_edit_neurons_shits(kn_bag, ori_pred_label_id, tgt_label_id)
        self.edit_neurons_knowledge(
            kn_bag, ori_pred_emb, tgt_emb, lambda_list_1, lambda_list_2
        )

        # inference with the new weights to chek if the edit worked
        (
            new_tgt_prob,
            new_tgt_rank,
            new_pred_prob,
            new_pred_label_id,
            pred_probs,
        ) = self.inference(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            tgt_pos=tgt_pos,
            tgt_label_id=tgt_label_id,
        )
        new_ori_pred_prob = pred_probs[0, ori_pred_label_id]
        new_pred_label = self.tokenizer.convert_ids_to_tokens(new_pred_label_id.item())

        # new pred label, new pred prob, tgt label, new tgt prob
        print(
            "============================== edited ========================================="
        )
        print(
            f"ori pred label: {ori_pred_label}, new ori pred label prob: {new_ori_pred_prob:.8}"
        )
        print(
            f"new pred label: {new_pred_label}, new pred label prob: {new_pred_prob:.8}"
        )
        print(f"tgt label: {tgt_ent}, tgt prob: {new_tgt_prob:.8}")

        # SAVE RESULTS
        if new_pred_label == tgt_ent:
            print("Successfully edited knowledge!")
            self.results["success_updated"] = 1
        if new_tgt_rank.item() <= 5:
            self.results[
                "success_updated_5"
            ] = 1  # if the wanted tgt was one of the 5 tokens with higher probability
        if new_pred_label != ori_pred_label:
            self.results[
                "ori_changed"
            ] = 1  # if the predicted label changes with the edition
        self.results["tgt_prob_inc"] = (
            new_tgt_prob.item() - ori_tgt_prob.item()
        )  # change in tgt's output prob
        self.results["tgt_prob_inc_ratio"] = (
            new_tgt_prob.item() - ori_tgt_prob.item()
        ) / ori_tgt_prob.item()
        self.results["tgt_ori_rank"] = ori_tgt_rank.item()  # previous target rank
        self.results["tgt_new_rank"] = new_tgt_rank.item()
        self.results["tgt_rank_inc"] = (
            ori_tgt_rank.item() - new_tgt_rank.item()
        )  # change in tgt rank: if > 0, tgt got higher in the rank with the edit

        ## Compute inner log PPL for other relations and entity pairs to check if editing this knowledge messe up
        ## with other knowledges (it shouldn't)
        (
            new_inner_log_ppl_mean,
            new_inner_log_ppl_mean_pred,
            new_inter_log_ppl_mean,
            new_inter_log_ppl_mean_pred,
            inner_rand_bag_idx_list,
            inter_rand_rel_list,
            inter_rand_bag_idx_list,
        ) = self.calculate_intra_inter_rel_ppl(eval_bag_list_perrel, rel, bag_idx)
        self.results["new_inner_log_ppl"] = new_inner_log_ppl_mean
        self.results["new_inter_log_ppl"] = new_inter_log_ppl_mean
        self.results["new_inner_log_ppl_pred"] = new_inner_log_ppl_mean_pred
        self.results["new_inter_log_ppl_pred"] = new_inter_log_ppl_mean_pred

        # ======================== Recover knowledge and calculate PPL to compare ======================================

        self.recover_neurons_knowledge(
            kn_bag, ori_pred_emb, tgt_emb, lambda_list_1, lambda_list_2
        )
        (
            inner_log_ppl_mean,
            inner_log_ppl_mean_pred,
            inter_log_ppl_mean,
            inter_log_ppl_mean_pred,
            _,
            _,
            _,
        ) = self.calculate_intra_inter_rel_ppl(
            eval_bag_list_perrel,
            rel,
            bag_idx,
            inner_rand_bag_idx_list,
            inter_rand_rel_list,
            inter_rand_bag_idx_list,
        )
        self.results["ori_inner_log_ppl"] = inner_log_ppl_mean
        self.results["ori_inter_log_ppl"] = inter_log_ppl_mean
        self.results["ori_inner_log_ppl_pred"] = inner_log_ppl_mean_pred
        self.results["ori_inter_log_ppl_pred"] = inter_log_ppl_mean_pred

        return self.results, kn_bag, ori_pred_emb, tgt_emb, lambda_list_1, lambda_list_2

    def edit_several_relations(
        self,
        eval_bag_list_perrel,
        kn_pos_perrel=None,
        kn_dir="results/kn/",
        max_rels=10,
    ):
        # eval_bag_list_perrel should be compuyted from "data_used_bags.json" file
        # so relations and entities which kns we still haven't computed are not included here
        rels = list(eval_bag_list_perrel.keys())

        print(
            "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" * 2
        )
        ## Edits knowledge and evaluates both the success of the edition and if it didn't affect other knowledge
        ## in other relation examples with different GT labels
        for rel_sample_cnt, rel in enumerate(
            rels
        ):  # Does this process for 10 != relations
            for i in range(10000):
                sampled_bag_idx = random.randint(
                    0, len(eval_bag_list_perrel[rel]) - 1
                )  # choose one entity pair randomly
                while True:
                    # randomly chooses another entity pair with a MASKED entitiy != from the previous pair's one
                    replaced_tail = random.choice(eval_bag_list_perrel[rel])[0][1]
                    if (
                        replaced_tail
                        != eval_bag_list_perrel[rel][sampled_bag_idx][0][1]
                    ):
                        break
                if kn_pos_perrel is not None:
                    kn_bag_list = kn_pos_perrel[rel]["kn_bag_list"]
                else:
                    kn_bag_list = None
                results = self.edit(
                    eval_bag_list_perrel,
                    rel,
                    sampled_bag_idx,
                    replaced_tail,
                    kn_bag_list=kn_bag_list,
                    kn_dir=kn_dir,
                )
                # replaced_tail=tgt_ent

                if type(results) == str:
                    continue

                for k in self.average_results.keys():
                    self.average_results[k].append(results(k))

                self.reset_results()

                print(
                    f"Results for relation {rel}-entity {sampled_bag_idx} and new targeted label {replaced_tail}: \n{results}"
                )
                print(
                    "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
                    * 2
                )

                if rel_sample_cnt == max_rels:
                    break

        ## Compute average EDIT results across rel_sample_cnt relations (1 entitie's example edited per relation)
        ori_MRR = (1 / np.array(self.average_results["tgt_ori_rank"])).mean()
        new_MRR = (1 / np.array(self.average_results["tgt_new_rank"])).mean()
        for k in self.average_results.keys():
            self.average_results[k] = np.array(self.average_results[k]).mean()

        ## PPL
        self.average_results["ori_inner_ppl"] = np.exp(
            -self.average_results["ori_inner_log_ppl"]
        )  # Lower values of perplexity (np.exp(...) ≃ 1) -> better performances
        self.average_results["new_inner_ppl"] = np.exp(
            -self.average_results["new_inner_log_ppl"]
        )
        self.average_results["ori_inner_ppl_pred"] = np.exp(
            -self.average_results["ori_inner_log_ppl_pred"]
        )  # Lower values of perplexity (np.exp(...) ≃ 1) -> better performances
        self.average_results["new_inner_ppl_pred"] = np.exp(
            -self.average_results["new_inner_log_ppl_pred"]
        )
        self.average_results["ori_inter_ppl"] = np.exp(
            -self.average_results["ori_inter_log_ppl"]
        )
        self.average_results["new_inter_ppl"] = np.exp(
            -self.average_results["new_inter_log_ppl"]
        )
        self.average_results["ori_inter_ppl_pred"] = np.exp(
            -self.average_results["ori_inter_log_ppl_pred"]
        )
        self.average_results["new_inter_ppl_pred"] = np.exp(
            -self.average_results["new_inter_log_ppl_pred"]
        )
        print(
            "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" * 2
        )
        print(
            "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" * 2
        )
        print(self.average_results)
        print(
            f'MR: {self.average_results["tgt_ori_rank"]:.8} -> {self.average_results["tgt_new_rank"]:.8} (up {self.average_results["tgt_new_rank"] - self.average_results["tgt_ori_rank"]:.8})'
        )
        print(f"MRR: {ori_MRR:.8} -> {new_MRR:.8} (up {new_MRR - ori_MRR:.8})")
        print(
            f"inner PPL: {self.average_results['ori_inner_ppl']:.8} -> {self.average_results['new_inner_ppl']:.8} (up {self.average_results['new_inner_ppl'] - self.average_results['ori_inner_ppl']:.8})"
        )
        print(
            f"inter PPL: {self.average_results['ori_inter_ppl']:.8} -> {self.average_results['new_inter_ppl']:.8} (up {self.average_results['new_inter_ppl'] - self.average_results['ori_inter_ppl']:.8})"
        )
        print(
            f"inner PPL (for prediction): {self.average_results['ori_inner_ppl_pred']:.8} -> {self.average_results['new_inner_ppl_pred']:.8} (up {self.average_results['new_inner_ppl_pred'] - self.average_results['ori_inner_ppl_pred']:.8})"
        )
        print(
            f"inter PPL (for prediction): {self.average_results['ori_inter_ppl_pred']:.8} -> {self.average_results['new_inter_ppl_pred']:.8} (up {self.average_results['new_inter_ppl_pred'] - self.average_results['ori_inter_ppl_pred']:.8})"
        )

        return self.average_results
