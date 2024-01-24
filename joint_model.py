import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class JointModel(nn.Module):
    def __init__(
            self, 
            gen_model, 
            retr_model, 
            gen_config, 
            retr_config, 
            gen_tokenizer, 
            retr_tokenizer, 
            loss_weights=None, 
            eval_lmbda=0.5,
            nce_loss='binary', 
            dist_loss='kl_div'
    ):
        super().__init__()

        self.gen_model = gen_model
        self.retr_model = retr_model

        self.gen_config = gen_config
        self.retr_config = retr_config

        self.gen_tokenizer = gen_tokenizer
        self.retr_tokenizer = retr_tokenizer
        
        self.loss_weights = loss_weights if loss_weights is not None else [1., 1., 1.]
        self.eval_lmbda = eval_lmbda
        self.nce_loss = nce_loss
        self.dist_loss = dist_loss

    def forward(self, **batch):
        retr_bm25_inputs_input_ids = batch['retr_bm25_inputs_input_ids']
        retr_bm25_inputs_attention_mask = batch['retr_bm25_inputs_attention_mask']
        retr_bm25_inputs_token_type_ids = batch['retr_bm25_inputs_token_type_ids']
        retr_bm25_labels = batch['retr_bm25_labels'].squeeze(0)

        gen_decoder_ids = batch['gen_decoder_ids']


        gen_bm25_ctxs_input_ids = batch['gen_bm25_ctxs_input_ids']
        gen_bm25_ctxs_attention_mask = batch['gen_bm25_ctxs_attention_mask']

        # [retriever loss]: log Q(Z|X)
        retr_normalized_log_prob = None
        if self.loss_weights[1] == 0 and self.loss_weights[2] == 0:
            retr_loss = 0.
        else:
            if self.retr_config.model_type == 'bert':
                retr_outputs = self.retr_model(
                    input_ids=retr_bm25_inputs_input_ids,
                    attention_mask=retr_bm25_inputs_attention_mask,
                    token_type_ids=retr_bm25_inputs_token_type_ids,
                    return_dict=True)
            else:
                retr_outputs = self.retr_model(
                    input_ids=retr_bm25_inputs_input_ids,
                    attention_mask=retr_bm25_inputs_attention_mask,
                    return_dict=True)

            retr_logits = retr_outputs.logits
            retr_normalized_log_prob = F.log_softmax(retr_logits.squeeze(1), dim=0).unsqueeze(0)

            if self.nce_loss == 'binary':
                # binary NCE loss
                retr_loss = F.binary_cross_entropy_with_logits(
                    retr_logits.squeeze(1), 
                    retr_bm25_labels)
            elif self.nce_loss == 'rank':
                # ranking NCE loss (~softmax; but here we may have >1 pos so we use KLD)
                retr_normalized_labels = (retr_bm25_labels / retr_bm25_labels.sum()).unsqueeze(0)
                retr_loss = F.kl_div(retr_normalized_log_prob, retr_normalized_labels, reduction="batchmean", log_target=False)
            else:
                raise NotImplementedError

        # [generator loss]: log P(X|Z)
        if self.loss_weights[0] == 0 or not self.training:
            gen_loss = 0
        else:
            gen_gold_ctxs_input_ids = batch['gen_gold_ctxs_input_ids']
            gen_gold_ctxs_attention_mask = batch['gen_gold_ctxs_attention_mask']
            n_gold_ctxs = gen_gold_ctxs_input_ids.size(0)
            gen_decoder_ids_for_gold = gen_decoder_ids[:n_gold_ctxs]

            gen_outputs = self.gen_model(
                input_ids=gen_gold_ctxs_input_ids,
                attention_mask=gen_gold_ctxs_attention_mask,
                labels=gen_decoder_ids_for_gold,
                return_dict=True)
            gen_loss = gen_outputs.loss

        # [Distribution matching]:
        # gather logits from generator model then normalize to obtain target
        if self.loss_weights[2] == 0 and self.training:
            dist_loss = 0.
        else:
            gen_outputs_for_retr = self.gen_model(
                input_ids=gen_bm25_ctxs_input_ids,
                attention_mask=gen_bm25_ctxs_attention_mask,
                labels=gen_decoder_ids,
                return_dict=True)
            gen_logits_for_retr = gen_outputs_for_retr.logits

            decoder_ids = gen_decoder_ids.clone()
            decoder_ids[decoder_ids == -100] = self.gen_tokenizer.pad_token_id
            avg_nll_for_retr = -F.log_softmax(gen_logits_for_retr, dim=-1).gather(2, decoder_ids.unsqueeze(2)).squeeze(2).mean(dim=1)

            retr_normalized_log_prob_target = F.log_softmax(-avg_nll_for_retr, dim=0).unsqueeze(0)

            # use previously retrieved outputs to calculate loss
            if self.dist_loss == 'kl_div' and retr_normalized_log_prob is not None:
                # KL(Q||P)
                kl_qp = F.kl_div(retr_normalized_log_prob, retr_normalized_log_prob_target, reduction="batchmean", log_target=True)
                # KL(P||Q)
                kl_pq = F.kl_div(retr_normalized_log_prob_target, retr_normalized_log_prob, reduction="batchmean", log_target=True)
                dist_loss = kl_qp + kl_pq
            elif self.dist_loss == 'js_div' and retr_normalized_log_prob is not None:
                mean_dist = torch.logaddexp(retr_normalized_log_prob, retr_normalized_log_prob_target) - np.log(2)
                # JS(Q,P) = 0.5 KL(Q||M) + 0.5 KL(P||M), M = 0.5(Q+P)
                kl_qm = F.kl_div(retr_normalized_log_prob, mean_dist, reduction="batchmean", log_target=True)
                kl_pq = F.kl_div(retr_normalized_log_prob_target, mean_dist, reduction="batchmean", log_target=True)
                dist_loss = 0.5 * (kl_qm + kl_pq)
            elif self.dist_loss == 'mse' and retr_normalized_log_prob is not None:
                dist_loss = ((retr_normalized_log_prob - retr_normalized_log_prob_target) ** 2).mean()
            else:
                dist_loss = F.binary_cross_entropy_with_logits(
                    -avg_nll_for_retr, retr_bm25_labels)

        dist_weight = self.loss_weights[-1] 
        loss = (self.loss_weights[0] * gen_loss + 
                self.loss_weights[1] * retr_loss + 
                dist_weight * dist_loss)
        if self.loss_weights[1] > 0 or dist_weight > 0:
            outputs = retr_outputs
        elif not self.training:
            outputs = gen_outputs_for_retr
        else:
            outputs = gen_outputs

        outputs['loss'] = loss

        if not self.training:
            if retr_normalized_log_prob is not None:
                output_logits = self.eval_lmbda * retr_normalized_log_prob.squeeze(0) - (1 - self.eval_lmbda) * avg_nll_for_retr
            else:
                output_logits = -avg_nll_for_retr

            outputs['logits'] = output_logits.unsqueeze(0)

        return outputs # if return_dict else (loss, gen_outputs.logits)

