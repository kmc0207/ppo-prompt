from datasets import load_dataset
def load_sst2():
    from datasets import load_dataset
    train_sentences = load_dataset( 'sst2', split='train')
    train_labels = train_sentences['label']
    test_sentences = load_dataset( 'sst2', split='validation')
    test_labels = test_sentences['label']
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]
    return train_sentences, train_labels, test_sentences, test_labels

def load_qnli():
    from datasets import load_dataset
    train_sentences = load_dataset('glue', 'qnli', split='train')
    train_labels = train_sentences['label']
    test_sentences = load_dataset('glue', 'qnli', split='validation')
    test_labels = test_sentences['label']
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]
    for item in train_sentences:
        item['text'] = 'question : ' + item['question'] + '\n sentence : ' + item['sentence']
    for item in test_sentences:
        item['text'] = 'question : ' + item['question'] + '\n sentence : ' + item['sentence']
    return train_sentences, train_labels, test_sentences, test_labels

def load_mnli():
    from datasets import load_dataset
    train_sentences = load_dataset('glue', 'mnli', split='train')
    train_labels = train_sentences['label']
    test_sentences = load_dataset('glue', 'mnli', split='validation_matched')
    test_labels = test_sentences['label']
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]
    for item in train_sentences:
        item['text'] = 'premise : ' + item['premise'] + '\n hypothesis : ' + item['hypothesis']
    for item in test_sentences:
        item['text'] = 'premise : ' + item['premise'] + '\n hypothesis : ' + item['hypothesis']
    return train_sentences, train_labels, test_sentences, test_labels

def load_agnews():
    from datasets import load_dataset
    train_sentences = load_dataset('ag_news', split='train')
    train_labels = train_sentences['label']
    test_sentences = load_dataset('ag_news', split='test')
    test_labels = test_sentences['label']
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences] 
    return train_sentences, train_labels, test_sentences, test_labels

def load_yelp_polarity():
    from datasets import load_dataset
    train_sentences = load_dataset('yelp_polarity', split='train')
    train_labels = train_sentences['label']
    test_sentences = load_dataset('yelp_polarity', split='test')
    test_labels = test_sentences['label']
    str2int = train_sentences.features['label']._str2int
    int2str = inv_map = {v: k for k, v in str2int.items()}
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]
    return train_sentences, train_labels, test_sentences, test_labels, str2int, int2str


def load_snli():
    from datasets import load_dataset
    train_sentences = load_dataset('snli', split='train')
    train_labels = train_sentences['label']
    test_sentences = load_dataset('snli', split='validation')
    test_labels = test_sentences['label']
    str2int = train_sentences.features['label']._str2int
    int2str = inv_map = {v: k for k, v in str2int.items()}
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]
    for item in train_sentences:
        item['text'] = 'premise : ' + item['premise'] + '\n hypothesis : ' + item['hypothesis']
    for item in test_sentences:
        item['text'] = 'premise : ' + item['premise'] + '\n hypothesis : ' + item['hypothesis']
    return train_sentences, train_labels, test_sentences, test_labels, str2int, int2str

def load_rte():
    from datasets import load_dataset
    # file_dict = {'train': 'data/k-shot/RTE/16-13/train.tsv'}
    # train_sentences = load_dataset('csv', data_files=file_dict, split='train', delimiter='\t')
    train_sentences = load_dataset('super_glue', 'rte', split='train')
    train_labels = train_sentences['label']
    unique = {label: idx for idx, label in enumerate(set(train_labels))}
    train_labels = [unique[label] for label in train_sentences['label']]
    # file_dict = {'train': 'data/k-shot/RTE/16-13/test.tsv'}
    # test_sentences = load_dataset('csv', data_files=file_dict, split='train', delimiter='\t')
    test_sentences = load_dataset('super_glue', 'rte', split='validation')
    test_labels = test_sentences['label']
    unique = {label: idx for idx, label in enumerate(set(test_labels))}
    test_labels = [unique[label] for label in test_sentences['label']]
    # str2int = train_sentences.features['label']._str2int
    # int2str = inv_map = {v: k for k, v in str2int.items()}
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]
    for item in train_sentences:
        item['text'] = 'premise : ' + item['premise'] + '\n hypothesis : ' + item['hypothesis']
    for item in test_sentences:
        item['text'] = 'premise : ' + item['premise'] + '\n hypothesis : ' + item['hypothesis']
    return train_sentences, train_labels, test_sentences, test_labels


def load_mrpc():
    from datasets import load_dataset
    train_sentences = load_dataset('glue', 'mrpc', split='train')
    train_labels = train_sentences['label']
    test_sentences = load_dataset('glue', 'mrpc', split='validation')
    test_labels = test_sentences['label']
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]
    for item in train_sentences:
        item['text'] = 'sentence1 : ' + item['sentence1'] + '\n sentence2 : ' + item['sentence2']
    for item in test_sentences:
        item['text'] = 'sentence1 : ' + item['sentence1'] + '\n sentence2 : ' + item['sentence2']
    return train_sentences, train_labels, test_sentences, test_labels

def load_customer_review():
    from datasets import load_dataset
    file_dict = {'train': 'cr/16-42/train.tsv'}
    train_sentences = load_dataset('csv', data_files=file_dict, split='train', delimiter='\t')
    train_labels = train_sentences['label']
    file_dict = {'train': 'cr/test.tsv'}
    test_sentences = load_dataset('csv', data_files=file_dict, split='train', delimiter='\t')
    test_labels = test_sentences['label']
    # str2int = train_sentences.features['label']._str2int
    # int2str = inv_map = {v: k for k, v in str2int.items()}
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]
    return train_sentences, train_labels, test_sentences, test_labels



def load_all_dataset(dataset_name):
    if dataset_name == 'sst2':
        return load_sst2()
    elif dataset_name == 'qnli':
        return load_qnli()
    elif dataset_name == 'mnli':
        return load_mnli()
    elif dataset_name == 'agnews':
        return load_agnews()
    elif dataset_name == 'yelp_polarity':
        return load_yelp_polarity()
    elif dataset_name == 'rte':
        return load_rte()
    elif dataset_name == 'mrpc':
        return load_mrpc()
    #elif dataset_name == 'customer_review':
    #    return load_customer_review()
    elif dataset_name == 'snli':
        return load_snli()
    else:
        raise NotImplementedError
    
def dataset_names():
    return ['sst2','qnli','mnli','agnews','yelp_polarity','rte','mrpc','snli']

def dataset_dicts(dataset_name):
    if dataset_name == 'sst2':
        return {0 : 'Negative',1 : 'Positive'}
    elif dataset_name == 'qnli':
        return {0 : 'Entailment',1 : 'Not Entailment'}
    elif dataset_name == 'mnli':
        return {0 : 'Entailment',1 : 'Neutral',2 : 'Contradiction'}
    elif dataset_name == 'agnews':
        return {0 : 'World',1 : 'Sports',2 : 'Business',3 : 'Sci/Tech'}
    elif dataset_name == 'yelp_polarity':
        return {0 : 'Negative',1 : 'Positive'}
    elif dataset_name == 'rte':
        return {0 : 'Entailment',1 : 'Not Entailment'}
    elif dataset_name == 'mrpc':
        return {0 : 'Different',1 : 'Equivalent'}
    elif dataset_name == 'snli':
        return {0 : 'Entailment',1 : 'Neutral',2 : 'Contradiction'}    
    else:
        raise NotImplementedError
    