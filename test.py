from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata2 as unicodedata
import string
import re
import random
import ast

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import collections
import math
import progressbar
import pprint

from seq2seq import MAX_LENGTH, variableFromSentence, EOS_token, SOS_token, use_cuda, EncoderRNN, AttnDecoderRNN
from data.eng_data import eng_data
from data.nld_data import nld_data

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]

class Lang(object):
    def __init__(self, initial_data):
        for key in initial_data:
            setattr(self, key, initial_data[key])

def showAttention(input_sentence, output_words, attentions, target_sentence=None):
    # Set up figure with colorbar
    fig = plt.figure()
    fig.canvas.set_window_title(target_sentence)
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def evaluateAndShowAttention(input_sentence, target_sentence=None):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions, target_sentence)

def computBrevityPenalty(c, r):
    if c <= r:
        return math.exp(1 - r/c)
    return 1

def createNGrams(sentence, N):
    ngrams = collections.defaultdict(list)
    for i in range(len(sentence)):
        for n in range(1, N + 1):
            if i+n < len(sentence) + 1:
                ngrams[n].append(" ".join(sentence[i: i+n]))
    return ngrams

def computeNGramPrecision(output_sentence, target_sentence, N):
    ngrams_output = createNGrams(output_sentence, N)
    ngrams_target = createNGrams(target_sentence, N)
    # ngrams_precision = {}
    ngrams_clipped_counts = {}
    ngrams_counts = {}
    for n in range(1, N + 1):
        # count maximum number of times a word occurs in any single reference translation
        counter_target = collections.Counter(ngrams_target[n])
        # clip the total count of each candidate word by its maximum reference count
        counter_output = collections.Counter(ngrams_output[n])
        clipped_counts = dict({(word, min(counter_output[word], counter_target[word])) for word in counter_output})
        # add these clipped counts up and divide by the total unclipped number of candidate words
        ngrams_clipped_counts[n] = sum(clipped_counts.values())
        ngrams_counts[n] = len(ngrams_output[n])
        # if ngrams_counts[n] == 0:
        #     ngrams_precision[n] = 0
        # else:
        #     ngrams_precision[n] = ngrams_clipped_counts[n] / ngrams_counts[n]
    return ngrams_clipped_counts, ngrams_counts

def computeBlueSentence(output_sentence, target_sentence, bp, N):
    '''
    http://www.aclweb.org/anthology/P02-1040.pdf
    output_sentence: translated sentance by mt
    target_sentence: correct translation
    N: maximal order n-gram
    '''
    # w = 1.0 / N # uniform weights
    # geometric_average = 0
    ngrams_clipped_counts, ngrams_counts = computeNGramPrecision(output_sentence, target_sentence, N)
    bleu = computeBlue(ngrams_clipped_counts, ngrams_counts, bp, N)
    # for precision in ngrams_precision:
    #     geometric_average += w * math.log(precision)
    # bleu = bp * math.exp(geometric_average)
    return bleu, ngrams_clipped_counts, ngrams_counts

def computeBlue(clipped, counts, bp, N):
    w = 1.0 / N # uniform weights
    geometric_average = 0
    for n in range(1, N + 1):
        if counts[n] == 0 or clipped[n] == 0:
            continue
        precision = clipped[n] / counts[n]
        geometric_average += w * math.log(precision)
    bleu = bp * math.exp(geometric_average)
    return bleu

def readTrainData(filename):
    N = 4
    output_corpus_length = 0
    target_corpus_length = 0
    translations = []
    with open(filename, 'r') as input_file:
        lines = input_file.readlines()
        print("Translating", str(len(lines)),  "sentences")
        with progressbar.ProgressBar(max_value=len(lines)) as bar:
            for i, line in enumerate(lines):
                eng, dutch = ast.literal_eval(line)
                eng = eng.split()
                # evaluateAndShowAttention(dutch, eng
                output_words, attentions = evaluate(encoder1, attn_decoder1, dutch)
                output_corpus_length += len(output_words)
                target_corpus_length += len(eng)
                translations.append((dutch, eng, output_words[:-1]))
                bar.update(i)
    bp = computBrevityPenalty(output_corpus_length, target_corpus_length)
    bleu_per_sentence = {}
    total_clipped_counts = collections.defaultdict(int)
    total_counts = collections.defaultdict(int)
    print("Computing BLEU score")
    with progressbar.ProgressBar(max_value=len(translations)) as bar:
        for i, translation in enumerate(translations):
            dutch, eng, output = translation
            bleu, ngrams_clipped_counts, ngrams_counts = computeBlueSentence(output, eng, bp, N)
            bleu_per_sentence[dutch] = [bleu, eng, output]
            for n in range(1, N + 1):
                total_clipped_counts[n] += ngrams_clipped_counts[n]
                total_counts[n] += ngrams_counts[n]
            bar.update(i)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(bleu_per_sentence)
    print("bleu on corpus:", computeBlue(total_clipped_counts, total_counts, bp, N))


if __name__ == "__main__":
    input_lang = Lang(nld_data)
    output_lang = Lang(eng_data)

    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
    encoder1.load_state_dict(torch.load('models_project6/encoder.pt', map_location=lambda storage, loc: storage))
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, 1, dropout_p=0.1)
    attn_decoder1.load_state_dict(torch.load('models_project6/decoder.pt', map_location=lambda storage, loc: storage))

    readTrainData("data/dutch-sentences.txt")
    # evaluateAndShowAttention("zij vertrekken morgenochtend uit japan")
