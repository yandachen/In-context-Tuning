from transformers import AutoTokenizer
from verbalized_model import VerbalizedModel
import torch
import os
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import random
import numpy as np
from data_loader import Data_loader
from tqdm import trange
import pickle as pkl


class ICT():

	def __init__(self, model_name, task_format, device, load_model_path=None):
		assert task_format in ['clm', 'mlm']

		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		if 'gpt2' in model_name:
			self.tokenizer.pad_token = self.tokenizer.eos_token
			self.tokenizer.padding_side = 'left'
		self.task_format = task_format

		self.model = VerbalizedModel(model_name=model_name, task_format=task_format, tokenizer=self.tokenizer)
		self.device = device

		if load_model_path is not None:
			self.model.load_state_dict(torch.load(load_model_path, map_location='cpu'))
		self.model.to(device)


	def meta_train(self, task2examples, task2templates, task2verbalizers,
				   num_demonstrations, example_delimiter, allow_label_overlap,
				   lr, num_warmup_steps, num_epochs, bsz, output_dir):

		# create dataloader
		data_loader = Data_loader(tokenizer=self.tokenizer, task_format=self.task_format,
								  task2verbalizers=task2verbalizers, example_delimiter=example_delimiter,
								  device=self.device)

		assert task2examples.keys() == task2templates.keys()
		if not os.path.exists(output_dir):
			os.makedirs(output_dir, exist_ok=True)

		trainable_parameters = []
		for param in self.model.named_parameters():
			assert param[1].requires_grad  # finetune all LM parameters
			trainable_parameters.append(param[1])

		optimizer = AdamW(params=trainable_parameters, lr=lr)
		optimizer.zero_grad()

		task_num_examples = [len(task2examples[task]) for task in task2examples]
		num_steps = sum(task_num_examples) * num_epochs  // bsz
		lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_steps)

		scaler = torch.cuda.amp.GradScaler() # fp16 training
		self.model.train()

		for epoch_idx in trange(num_epochs):
			# prepare training examples
			epoch_train_examples = []
			for task in task2examples:
				task_examples = []
				examples = task2examples[task]
				for query_example in examples:
					template = random.sample(task2templates[task], 1)[0]
					input_text = data_loader.prepare_input(task, query_example, examples, num_demonstrations,
														   template, allow_label_overlap)
					if len(self.tokenizer(input_text)['input_ids']) <= self.tokenizer.model_max_length:
						task_examples.append((task, input_text, query_example['<label>']))
				# divide task_examples into batches, so that each batch has all examples from the same task
				random.shuffle(task_examples)
				for idx in range(0, len(task_examples), bsz):
					epoch_train_examples.append(task_examples[idx: idx + bsz])

			random.shuffle(epoch_train_examples)
			train_loss = []
			# batch training
			for batch_example_idx in range(len(epoch_train_examples)):
				optimizer.zero_grad()
				batch_train_examples = epoch_train_examples[batch_example_idx]
				input_dict = self.tokenizer([example[1] for example in batch_train_examples],
											padding=True, return_tensors="pt").to(self.device)
				labels = [example[2] for example in batch_train_examples]
				task = batch_train_examples[0][0]
				with torch.cuda.amp.autocast():
					loss, logits = self.model.forward(input_dict,
													  torch.LongTensor(data_loader.task2verbalizer_worids[task]).to(self.device),
													  torch.LongTensor(labels).to(self.device))
				train_loss.append(loss.item())
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()
				lr_scheduler.step()

			if output_dir is not None:
				with open('%s/train.log' % output_dir, 'a') as f:
					f.write('Epoch %d: - train loss: %.4f\n' % (epoch_idx, np.average(train_loss)))
					f.flush()

		if output_dir is not None:
			torch.save(self.model.state_dict(), '%s/model.pkl' % output_dir)


	def meta_test(self, task2examples, task2templates, task2verbalizers,
				  num_demonstrations, example_delimiter, allow_label_overlap, num_prefix_selections,
				  bsz):
		# create dataloader
		data_loader = Data_loader(tokenizer=self.tokenizer, task_format=self.task_format,
								  task2verbalizers=task2verbalizers, example_delimiter=example_delimiter,
								  device=self.device)
		task2preds, task2scores = {}, {}
		assert task2examples.keys() == task2templates.keys()
		for task in task2examples:
			examples, templates = task2examples[task], task2templates[task]
			input_texts, labels = [], []
			for template in templates:
				for query_example in examples:
					for prefix_selection_idx in range(num_prefix_selections):
						input_text = data_loader.prepare_input(
							task, query_example, examples, num_demonstrations, template, allow_label_overlap)
						input_texts.append(input_text)
						labels.append(query_example['<label>'])

			# predict on input_texts
			self.model.eval()
			output_logits = []
			for example_idx in np.arange(0, len(input_texts), bsz):
				input_dict = self.tokenizer(input_texts[example_idx: example_idx + bsz],
											padding=True, return_tensors="pt").to(self.device)
				with torch.no_grad():
					with torch.cuda.amp.autocast():
						batch_output_logits = self.model.forward(input_dict,
																 torch.LongTensor(data_loader.task2verbalizer_worids[task]).to(self.device))
				output_logits += [logits.cpu().numpy() for logits in batch_output_logits]
			task2preds[task] = output_logits

			# compute score
			precision1, precision, mrr = [], [], []
			for logits, gt_label in zip(output_logits, labels):
				rank = np.count_nonzero(logits > logits[gt_label]) + 1
				assert rank >= 1
				mrr.append(1 / rank)
				precision1.append(1 if rank <= 1 else 0)
				precision.append(1 if rank <= 10 else 0)
			task2scores[task] = {'precision1': np.mean(precision1), 'precision10': np.mean(precision),
								 'mrr': np.mean(mrr)}
		return task2preds, task2scores



if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Meta-training / meta-testing of in-context tuning.')
	parser.add_argument('--mode', type=str, choices=['meta-train', 'meta-test'],
						help='whether you want to meta-train a model or meta-test a meta-trained model.')

	parser.add_argument('--model_name', type=str,
						help='name of the model.')
	parser.add_argument('--task_format', type=str, choices=['mlm', 'clm'],
						help='frame verbalizer classifcation as masked language modeling or causal language modeling.')
	parser.add_argument('--device', type=str,
						help='device used for this experiment..')
	parser.add_argument('--task2examples', type=str,
						help='path to the meta-training/meta-testing data file (a dictionary of task2examples).')
	parser.add_argument('--task2templates', type=str,
						help='path to the meta-training/meta-testing templates file (a dictionary of task2templates).')
	parser.add_argument('--task2verbalizers', type=str,
						help='path to the meta-training/meta-testing verbalizers file (a dictionary of task2verbalizers).')
	parser.add_argument('--num_demonstrations', type=int,
						help='number of few-shot demonstrations.')
	parser.add_argument('--example_delimiter', type=int,
						help='delimiter used to separate consecutive examples in the input text.')
	parser.add_argument('--allow_label_overlap', action='store_true',
						help='whether few-shot support examples are allowed to have overlapping labels with the query example.')
	parser.add_argument('--bsz', type=int,
						help='batch size for meta-training/meta-testing.')

	# arguments only used for meta-training
	parser.add_argument('--lr', type=float,
						help='learning rate for meta-training.')
	parser.add_argument('--num_warmup_steps', type=int,
						help='number of warmup steps of the learning rate scheduler.')
	parser.add_argument('--num_epochs', type=int,
						help='number of meta-training epochs.')
	parser.add_argument('--output_dir', type=str,
						help='output directory to store the meta-trained model and the meta-training log file.')

	# arguments only used for meta-testing
	parser.add_argument('--num_prefix_selections', type=int,
						help='number of demonstration sampling for each query example (result averaged across different sampled demonstrations).')
	parser.add_argument('--load_model_path', type=str,
						help='path to the meta-trained model to evaluate with.')

	args = parser.parse_args()
	assert args.mode in ['meta-train', 'meta-test']
	if args.mode == 'meta-train':
		assert None not in [args.model_name, args.task_format, args.device,
							args.task2examples, args.task2templates, args.task2verbalizers,
							args.num_demonstrations, args.example_delimiter, args.allow_label_overlap,
							args.lr, args.num_warmup_steps, args.num_epochs, args.bsz, args.output_dir]
		ict = ICT(args.model_name, args.task_format, args.device)
		task2examples = pkl.load(open(args.task2examples, 'rb'))
		task2templates = pkl.load(open(args.task2templates, 'rb'))
		task2verbalizers = pkl.load(open(args.task2verbalizers, 'rb'))
		ict.meta_train(task2examples, task2templates, task2verbalizers,
					   args.num_demonstrations, args.example_delimiter, args.allow_label_overlap,
					   args.lr, args.num_warmup_steps, args.num_epochs, args.bsz, args.output_dir)

	elif args.mode == 'meta-test':
		assert None not in [args.model_name, args.task_format, args.device, args.load_model_path,
							args.task2examples, args.task2templates, args.task2verbalizers,
							args.num_demonstrations, args.example_delimiter, args.allow_label_overlap,
							args.num_prefix_selections, args.bsz]
		ict = ICT(args.model_name, args.task_format, args.device, args.load_model_path)
		task2examples = pkl.load(open(args.task2examples, 'rb'))
		task2templates = pkl.load(open(args.task2templates, 'rb'))
		task2verbalizers = pkl.load(open(args.task2verbalizers, 'rb'))
		ict.meta_test(task2examples, task2templates, task2verbalizers,
					  args.num_demonstrations, args.example_delimiter,
					  args.allow_label_overlap, args.num_prefix_selections, args.bsz)
