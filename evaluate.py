import numpy as np
import collections
import math
import pickle
import progressbar
import pprint
import unicodedata2 as unicodedata
import string
import re
import random
import ast
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from vector_entropy import vector_entropy
from same_order import same_order


def plot_entropy_bleu(entropy, beer, colors):
    bleu = [float(b) for b in beer]
    plt.scatter(entropy, beer)
    # Change name to diagonality if using diagonality
    plt.xlabel("Entropy")
    plt.ylabel("Beer")

    plt.show()

def showAttention(input_sentence, output_words, attentions, target_sentence=None):
    # Set up figure with colorbar
    fig = plt.figure()
    fig.canvas.set_window_title(target_sentence)
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    print(input_sentence.split(' '))
    print(output_words.split(' '))
    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' '), rotation=90)
    ax.set_yticklabels([''] + output_words.split(' '))
    ax.set_title(target_sentence)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

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
    return ngrams_clipped_counts, ngrams_counts

def computeBlueSentence(output_sentence, target_sentence, bp, N):
    '''
    http://www.aclweb.org/anthology/P02-1040.pdf
    output_sentence: translated sentance by mt
    target_sentence: correct translation
    N: maximal order n-gram
    '''
    ngrams_clipped_counts, ngrams_counts = computeNGramPrecision(output_sentence, target_sentence, N)
    bleu = computeBlue(ngrams_clipped_counts, ngrams_counts, bp, N)
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

def evalutateData(src, target, predicted, attentions):
    N = 4
    output_corpus_length = 0
    target_corpus_length = 0
    translations = []
    print("Processing", str(len(predicted)),  "sentences")
    with progressbar.ProgressBar(max_value=len(predicted)) as bar:
        for i in range(len(predicted)):
            eng = src[i].rstrip()
            dutch = target[i].rstrip()
            pred = predicted[i].lower().rstrip()
            attention = attentions[i]
            # showAttention(eng, pred, attention)
            output_corpus_length += len(pred)
            target_corpus_length += len(dutch)
            translations.append((eng, dutch, pred, attention))
            bar.update(i)
    bp = computBrevityPenalty(output_corpus_length, target_corpus_length)
    bleu_per_sentence = {}
    total_clipped_counts = collections.defaultdict(int)
    total_counts = collections.defaultdict(int)
    entropies = []
    bleus = []
    print("Computing BLEU score")
    with progressbar.ProgressBar(max_value=len(translations)) as bar:
        for i, translation in enumerate(translations):
            eng, dutch, output, attention = translation
            entropy = [vector_entropy(a) for a in attention]
            bleu, ngrams_clipped_counts, ngrams_counts = computeBlueSentence(output, dutch, bp, N)
            bleu_per_sentence[dutch] = [bleu, eng, output, np.mean(entropy)]
            # showAttention(eng, output, attention)
            entropies.append(np.mean(entropy))
            bleus.append(bleu)
            for n in range(1, N + 1):
                total_clipped_counts[n] += ngrams_clipped_counts[n]
                total_counts[n] += ngrams_counts[n]
            bar.update(i)
    pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(bleu_per_sentence)
    print("bleu on corpus:", computeBlue(total_clipped_counts, total_counts, bp, N))
    plot_entropy_bleu(entropies, bleus)

def testData(src, target, predicted, attentions, beers):
    entropies = []
    colors = []
    with progressbar.ProgressBar(max_value=len(predicted)) as bar:
        for i in range(len(predicted)):
            eng = src[i].rstrip()
            dutch = target[i].rstrip()
            pred = predicted[i].lower().rstrip()
            attention = attentions[i]
            colors.append(len(pred.split()))
            entropy = [vector_entropy(a) for a in attention]
            # Switch comment to use diagonality
            # entropies.append(same_order(attention))
            entropies.append(np.mean(entropy))
            # Uncomment to show attention matrices
            # showAttention(eng, pred, attention, dutch)
            bar.update(i)
    plot_entropy_bleu(entropies, beers, colors)

    # Show Pearson correlation
    beers = [float(b) for b in beers]
    print(np.corrcoef(np.array(entropies), np.array(beers)))

def readData(src_filename, target_filename, predicted_filename, attention_filename, beer_filename):
    with open(src_filename, 'r') as src_file:
        src = src_file.readlines()
    with open(target_filename, 'r') as target_file:
        target = target_file.readlines()
    with open(predicted_filename, 'r') as predicted_file:
        predicted = predicted_file.readlines()
    with open(beer_filename, 'r') as beer_file:
        beer = beer_file.readlines()
    attentions = pickle.load(open(attention_filename, "rb" ))
    return src, target, predicted, attentions, beer

if __name__ == "__main__":
    # TODO: read from user input
    eng, nld, predicted, attentions, beer = readData("./data/torch/eng.txt", "./data/torch/nld.txt", "./output/torch/old_data/old_data_lower.txt", "./output/torch/old_data/attn.pkl", "./data/torch/beer_lower.txt")
    # eng, nld, predicted, attentions, beer = readData("./data/torch/eng_short.txt", "./data/torch/nld_short.txt", "./output/torch/old_short/old_data_predict_short.txt", "./output/torch/old_short/attentions_short.pkl", "./data/torch/beer_short.txt")
    # eng, nld, predicted, attentions, beer = readData("./data/ted/valid_eng.txt", "./data/ted/valid_nld.txt", "./output/ted/valid_pred.txt", "./output/ted/valid_attn.pkl", "./data/ted/valid_beer.txt")
    # eng, nld, predicted, attentions, beer = readData("./data/ted/test_eng.txt", "./data/ted/test_nld.txt", "./output/ted/test_pred.txt", "./output/ted/test_attn.pkl", "./data/ted/test_beer.txt")

    testData(eng, nld, predicted, attentions, beer)
    # evalutateData(eng, nld, predicted, attentions)
