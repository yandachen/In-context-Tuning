import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM


class VerbalizedModel(nn.Module):
    def __init__(self, model_name, task_format, tokenizer):
        super(VerbalizedModel, self).__init__()
        assert task_format in ['mlm', 'clm']
        if task_format == 'mlm':
            self.lm_model = AutoModelForMaskedLM.from_pretrained(model_name)
        elif task_format == 'clm':
            self.lm_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.task_format = task_format
        self.loss_fct = nn.CrossEntropyLoss() # classification loss
        self.tokenizer = tokenizer


    def forward(self, input_dict, verbalizer_word_ids, labels=None):
        """
        :param input_dict: {'input_ids': ..., 'attention_mask': ..., 'token_type_ids': ...}
        :param verbalizer_word_ids: list of integers (length equal to the number of classification classes),
               where the i-th number is the word id of the i-th verbalizer.
        :param labels: example labels
        """
        if self.task_format == 'mlm':
            output = self.lm_model(**input_dict)
            output_logits = output.logits

            # locate <mask> positions in the input_ids
            # check that there is one unique mask position.
            mask_pos = []
            for input_ids in input_dict['input_ids']:
                assert sum([input_id == self.tokenizer.mask_token_id for input_id in input_ids]) == 1
                mask_pos.append(input_ids.tolist().index(self.tokenizer.mask_token_id))

            # slice output_logits at the masked positions and verbalizer-ids.
            logits_verbalizers = []
            for example_idx in range(len(output_logits)):
                logits_verbalizers.append(torch.index_select(output_logits[example_idx][mask_pos[example_idx]],
                                                             dim=0, index=verbalizer_word_ids))
            output_logits = torch.vstack(logits_verbalizers)

        elif self.task_format == 'clm':
            output = self.model(**input_dict)
            output_logits = output.logits[:, -1, verbalizer_word_ids]

        else:
            raise NotImplementedError('self.task_format must be either "clm" or "mlm".')

        if labels is None:
            return output_logits # (batch size, num output vocabs)
        else:
            loss = [self.loss_fct(output_logits[example_idx].unsqueeze(dim=0), labels[example_idx: example_idx + 1])
                    for example_idx in range(len(output_logits))]
            loss = torch.mean(torch.hstack(loss))
            return loss, output_logits

