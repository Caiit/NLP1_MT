import zipfile
import unicodedata
import re
import xml.etree.ElementTree as ET
import nltk

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readData(filename, reverse=False):
    print("Reading data...")
    z = zipfile.ZipFile(filename, 'r')
    with z.open("nld.txt") as zip_file:
        lines = zip_file.readlines()

    nld = []
    eng = []
    for sentence in lines:
        en, nl = sentence.decode("utf-8").split('\t')
        eng.append(normalizeString(en))
        nld.append(normalizeString(nl))
    return eng, nld

def readTedData(filename):
    data = []
    tree = ET.parse(filename)
    root = tree.getroot()
    for seg in root.findall('.//seg'):
        tokens = ' '.join(nltk.word_tokenize(seg.text))
        data.append(normalizeString(tokens))
    return data

def saveDataAsDict(data, filename):
    with open(filename + ".py", 'w') as output_file:
        output_file.write(filename + " = " + str(vars(data)))


def saveData(data, filename):
    with open(filename, 'w') as output_file:
        for sentence in data:
            output_file.write(sentence + "\n")


if __name__ == "__main__":
    # eng, nld = readData("data/nld-eng.zip")
    # saveData(eng, "data/torch/eng.txt")
    # saveData(nld, "data/torch/nld.txt")

    eng = readTedData("data/ted/IWSLT17.TED.tst2010.en-nl.en.xml")
    saveData(eng, "data/ted/test_eng.txt")

    nld = readTedData("data/ted/IWSLT17.TED.tst2010.en-nl.nl.xml")
    saveData(nld, "data/ted/test_nld.txt")
