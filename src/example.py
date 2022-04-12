import pickle as pkl
from ict import ICT


if __name__ == '__main__':
	fold_idx = 0
	num_demonstrations = 5
	model_name = 'bert-base-cased'

	# prepare data
	cv_split = pkl.load(open('../example_data/cross_validation_splits.pkl', 'rb'))
	rel2data = pkl.load(open('../example_data/data.pkl', 'rb'))
	rel2templates = pkl.load(open('../example_data/templates.pkl', 'rb'))
	train_tasks, val_tasks = cv_split[fold_idx]['train'], cv_split[fold_idx]['val']
	train_task2examples = {task: rel2data[task] for task in train_tasks}
	train_task2templates = {task: rel2templates[task] for task in train_tasks}
	val_task2examples = {task: rel2data[task] for task in val_tasks}
	val_task2templates = {task: rel2templates[task] for task in val_tasks}

	# load verbalizers
	verbalizers = []
	with open('../example_data/class_verbalizers.txt', 'r') as f:
		for line in f.readlines():
			word = line.strip()
			assert len(word) != 0  # nonempty
			verbalizers.append(word)

	output_dir = '../example/%dshot_fold%d/' % (num_demonstrations, fold_idx)
	num_epochs, lr = 15, 3e-6
	ict = ICT(model_name='bert-base-cased', task_format='mlm', device='cuda:1')
	# meta-train
	ict.meta_train(train_task2examples, train_task2templates, task2verbalizers={task: verbalizers for task in train_tasks},
				   num_demonstrations=5, example_delimiter=' ', allow_label_overlap=False,
				   lr=3e-6, num_warmup_steps=100, num_epochs=15, bsz=48, output_dir=output_dir)

	# meta-test on val
	val_task2preds, val_task2scores = ict.meta_test(val_task2examples, val_task2templates,
													task2verbalizers={task: verbalizers for task in val_tasks},
													num_demonstrations=5, example_delimiter=' ', allow_label_overlap=False,
													num_prefix_selections=20, bsz=48)
	print(val_task2scores)

	# meta-test on test
	test_tasks = cv_split[fold_idx]['test']
	test_task2examples = {task: rel2data[task] for task in test_tasks}
	test_task2templates = {task: rel2templates[task] for task in test_tasks}
	test_task2preds, test_task2scores = ict.meta_test(test_task2examples, test_task2templates,
													  task2verbalizers={task: verbalizers for task in test_tasks},
													  num_demonstrations=5, example_delimiter=' ', allow_label_overlap=False,
													  num_prefix_selections=20, bsz=48)
	print(test_task2scores)
