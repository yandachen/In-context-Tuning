import random


class Data_loader():

    def __init__(self, tokenizer, task_format, task2verbalizers, example_delimiter, device):
        self.tokenizer = tokenizer
        self.task_format = task_format
        assert self.task_format in ['mlm', 'clm']
        self.example_delimiter = example_delimiter
        self.device = device

        self.task2verbalizers = task2verbalizers
        self.task2verbalizer_worids = {}
        for task in task2verbalizers:
            self.task2verbalizer_worids[task] = [self.tokenizer._convert_token_to_id_with_added_voc(verbalizer)
                                                 for verbalizer in task2verbalizers[task]]
            verbalizer_wordids = []
            for verbalizer in task2verbalizers[task]:
                wordids = self.tokenizer(verbalizer, add_special_tokens=False)['input_ids']
                assert len(wordids) == 1 # current code assumes that each verbalizer is one token.
                verbalizer_wordids.append(wordids[0])
            self.task2verbalizer_worids[task] = verbalizer_wordids


    def sample_demonstrations(self, query_example, support_examples, num_demonstrations, allow_label_overlap):
        assert allow_label_overlap in [True, False]
        if allow_label_overlap:
            selectable_example_idx = [i for i in range(len(support_examples))
                                      if not self.check_input_same(support_examples[i], query_example)]
        else:
            selectable_example_idx = [i for i in range(len(support_examples))
                                      if not self.check_input_same(support_examples[i], query_example)
                                      and not support_examples[i]['<label>'] == query_example['<label>']]
        assert len(selectable_example_idx) <= len(support_examples)
        prefix_example_idxs = random.sample(selectable_example_idx, num_demonstrations)
        return prefix_example_idxs


    def check_input_same(self, example1, example2):
        assert example1.keys() == example2.keys()
        for key in example1:
            if key != '<label>' and example1[key] != example2[key]:
                return False
        return True


    def prepare_input(self, task, query_example, support_examples, num_demonstrations, template, allow_label_overlap):
        prefix_example_idxs = self.sample_demonstrations(query_example, support_examples, num_demonstrations, allow_label_overlap)
        prefix_examples = [support_examples[idx] for idx in prefix_example_idxs]
        input_text = self.encode_input_str(prefix_examples, query_example, template, self.task2verbalizers[task])
        return input_text


    def encode_example_with_template(self, template, example, verbalizers):
        templated_example = template[:]
        for key in example:
            if key != '<label>': # all input keys
                templated_example = templated_example.replace(key, example[key])
        if self.task_format == 'mlm':
            nolabel_templated_example = templated_example[:].replace('<label>', self.tokenizer.mask_token)
            withlabel_templated_example = templated_example[:].replace('<label>', verbalizers[example['<label>']])
            return nolabel_templated_example, withlabel_templated_example
        elif self.task_format == 'clm':
            assert template.endswith('<label>') # template for CLM decoding must produce labels at the end of the prompt.
            nolabel_templated_example = templated_example[:].replace('<label>', '')
            withlabel_templated_example = templated_example[:].replace('<label>', verbalizers[example['<label>']])
            return nolabel_templated_example, withlabel_templated_example


    def encode_input_str(self, prefix_examples, query_example, template, verbalizers):
        input_texts = []
        for example in prefix_examples:
            _, withlabel_templated_example = self.encode_example_with_template(template, example, verbalizers)
            input_texts.append(withlabel_templated_example)
        query_example_masked, _ = self.encode_example_with_template(template, query_example, verbalizers)
        input_text = self.example_delimiter.join(input_texts + [query_example_masked])
        input_ids = self.tokenizer.encode(input_text)
        assert len(input_ids) <= self.tokenizer.model_max_length
        return input_text