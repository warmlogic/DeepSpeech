# coding: utf-8
import os
import re
import codecs
import unicodedata
import collections

import numpy as np
import tensorflow as tf
import cPickle as pickle

from math import ceil
from nltk.tokenize.punkt import PunktSentenceTokenizer

from tensorflow.contrib.learn.python.learn.datasets import base

def wiki_raw_data(data_path=None, model_path=None, pickle_word_to_id=False):
  # Preprocess raw data into train, valididation, and test sets
  raw_file = _download_wiki_raw_data(data_path)
  processed_file = _process_raw_file(raw_file)
  train_file, valid_file, test_file = _split_processed_file(processed_file)

  # Build map from words to ids
  word_to_id = _build_vocab(train_file)

  # Conditionally pickle word_to_id
  if pickle_word_to_id:
    _pickle_word_to_id(model_path, word_to_id)

  # Convert train, validation, and tests texts to numbers
  train_data = _file_to_word_ids(train_file, word_to_id)
  valid_data = _file_to_word_ids(valid_file, word_to_id)
  test_data = _file_to_word_ids(test_file, word_to_id)

  return train_data, valid_data, test_data

def _download_wiki_raw_data(data_path):
    wiki_file_name = 'wiki.txt'
    wiki_url = 'https://s3.amazonaws.com/language-models/raw/en/wiki.txt'
    return base.maybe_download(wiki_file_name, data_path, wiki_url)

def _process_raw_file(raw_file):
  # Determine processed file name
  data_path = os.path.dirname(raw_file)
  processed_file = os.path.join(data_path, "processed.txt")
  # If processed_file exists return immediately
  if os.path.isfile(processed_file):
    return processed_file
  # Determine temp processed file name
  temp_processed_file = os.path.join(data_path, "temp-processed.txt")
  # Create Counter for word counts
  counter = collections.Counter()
  # Split raw_file into sentences, convert to lower case, replace numbers with N save temp results
  with codecs.open(raw_file, encoding='utf-8') as open_raw_file:
    with open(temp_processed_file, 'w+') as open_temp_processed_file:
      sentence_tokenizer = PunktSentenceTokenizer()
      for sentence in sentence_tokenizer.sentences_from_text(unicodedata.normalize('NFKD', open_raw_file.read()).encode('ascii', 'ignore')):
        sentence = sentence.lower()
        sentence = re.sub(r'[0-9]+', 'N', sentence)
        sentence = re.sub(r'\n', ' ', sentence)
        sentence = re.sub(r'-|—', ' ', sentence)
        sentence = re.sub(r'[^a-zN ]', '', sentence)
        counter.update(sentence.split())
        open_temp_processed_file.write(' ' + sentence + '\n')
  # Create dictionary of the 10K most commit words 
  word_to_id = dict(counter.most_common(10000))
  # Replace words not in the top N words with <unk> save final results
  with open(processed_file, 'w+') as open_processed_file:
    with open(temp_processed_file) as open_temp_processed_file:
      for line in open_temp_processed_file:
        words = []
        for word in line.split():
          if word in word_to_id:
            words.append(word)
          else:
            words.append('<unk>')
        open_processed_file.write(' ' + ' '.join(words) + '\n')
  # Remove the temporary file
  os.remove(temp_processed_file)
  return processed_file

def _split_processed_file(processed_file):
  # Determine output file names
  data_path = os.path.dirname(processed_file)
  train_file = os.path.join(data_path, "train.txt")
  valid_file = os.path.join(data_path, "valid.txt")
  test_file = os.path.join(data_path, "test.txt")

  # If train_file, valid_file, and test_file exist return immediately
  if os.path.isfile(train_file) and os.path.isfile(valid_file) and os.path.isfile(test_file):
    return train_file, valid_file, test_file

  # Determine the number of lines in processed_file
  num_lines = sum(1 for line in open(processed_file))

  # Deterine which lines are in the train (85%) data set
  train_lines = np.random.choice(a=xrange(num_lines), size=int(ceil(0.85 * num_lines)), replace=False)

  # Determine which lines are in the validation (5%) data set
  valid_lines = np.random.choice(a=list(set(xrange(num_lines)) - set(train_lines)), size=int(ceil(0.05 * num_lines)), replace=False)

  # All other lines must be in the test set

  # Split processed_file into train, valid, and test
  with open(train_file, 'w+') as open_train_file:
    with open(valid_file, 'w+') as open_valid_file:
      with open(test_file, 'w+') as open_test_file:
        with open(processed_file) as open_processed_file:
          for line_num, line in enumerate(open_processed_file):
            if line_num in train_lines:
              open_train_file.write(line)
            elif line_num in valid_lines:
              open_valid_file.write(line)
            else:
              open_test_file.write(line)

  return train_file, valid_file, test_file

def _build_vocab(filename):
  # Create Counter for word counts
  counter = collections.Counter()

  # Count words in filename
  with open(filename) as open_filename:
    for line in open_filename:
      parsed_line = _parse_line(line)
      counter.update(parsed_line)

  # Sort on word counts
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  # Get ordered words
  words, _ = list(zip(*count_pairs))

  # Map words to ids
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id

def _pickle_word_to_id(model_path, word_to_id):
  word_to_id_file = os.path.join(model_path, 'word_to_id.pkl')
  with open(word_to_id_file , 'wb' ) as open_word_to_id_file:
    pickle.dump(word_to_id, open_word_to_id_file)

def get_word_to_id(model_path):
  word_to_id_file = os.path.join(model_path, 'word_to_id.pkl')
  with open(word_to_id_file, 'rb' ) as open_word_to_id_file:
    word_to_id = pickle.load(open_word_to_id_file)
  return word_to_id

def _file_to_word_ids(filename, word_to_id):
  # Define word id list to return
  word_ids = []

  # Map words to id's
  with open(filename) as open_filename:
    for line in open_filename:
      word_ids.extend(text_to_word_ids(line, word_to_id))

  return word_ids

def text_to_word_ids(text, word_to_id):
  # Define word id list to return
  word_ids = []

  # Map words to id's
  for word in _parse_line(text):
    if word in word_to_id:
      word_ids.append(word_to_id[word])
    else:
      word_ids.append(word_to_id['<unk>'])

  return word_ids

def _parse_line(line):
  return (line + ' <eos>').split()

def wiki_iterator(raw_data, batch_size, num_steps):
  # Convert raw_data into an np array
  raw_data = np.array(raw_data, dtype=np.int32)

  # Create a batch of data
  data_len = len(raw_data)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

  # Calculate epoch size
  epoch_size = (batch_len - 1) // num_steps

  # Check for trivial epoch
  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  # Create an iterator
  for i in range(epoch_size):
    x = data[:, i*num_steps:(i+1)*num_steps]
    y = data[:, i*num_steps+1:(i+1)*num_steps+1]
    yield (x, y)
