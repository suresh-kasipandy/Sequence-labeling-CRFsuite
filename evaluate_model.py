import pycrfsuite
from collections import namedtuple, Counter
import csv
import glob
import os


def get_utterances_from_file(dialog_csv_file):
    """Returns a list of DialogUtterances from an open file."""
    reader = csv.DictReader(dialog_csv_file)
    return [_dict_to_dialog_utterance(du_dict) for du_dict in reader]

def get_utterances_from_filename(dialog_csv_filename):
    """Returns a list of DialogUtterances from an unopened filename."""
    with open(dialog_csv_filename, "r") as dialog_csv_file:
        return get_utterances_from_file(dialog_csv_file)

def get_data(data_dir):
    """Generates lists of utterances from each dialog file.

    To get a list of all dialogs call list(get_data(data_dir)).
    data_dir - a dir with csv files containing dialogs"""
    dialog_filenames = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    for dialog_filename in dialog_filenames:
        yield get_utterances_from_filename(dialog_filename)

DialogUtterance = namedtuple("DialogUtterance", ("act_tag", "speaker", "pos", "text"))

PosTag = namedtuple("PosTag", ("token", "pos"))

def _dict_to_dialog_utterance(du_dict):
    """Private method for converting a dict to a DialogUtterance."""

    # Remove anything with
    for k, v in du_dict.items():
        if len(v.strip()) == 0:
            du_dict[k] = None

    # Extract tokens and POS tags
    if du_dict["pos"]:
        du_dict["pos"] = [
            PosTag(*token_pos_pair.split("/"))
            for token_pos_pair in du_dict["pos"].split()]
    return DialogUtterance(**du_dict)

asd=list(get_data("C:/Users/neku104/PycharmProjects/544HW3/test"))
testfeatures1=[]
testlabels1=[]
testfeatures2=[]
testlabels2=[]
for val in asd:
    spk1 = ''
    spk2 = ''
    i=0
    fe=[]
    lb=[]

    for item in val:
        fa=[]
        spk2=getattr(item, "speaker")
        if i==0:
            fa.append("F")
        if i>0:
            if spk1!=spk2:
                fa.append("SC")

        toks=getattr(item, "pos")

        if toks is not None:
            for x in toks:
                fa.append("TOKEN_"+getattr(x, "token"))
            for x in toks:
                fa.append("POS_"+getattr(x, "pos"))

        lb.append(getattr(item, "act_tag"))
        fe.append(fa)
        i+=1
        spk1=spk2
    testfeatures1.append(fe)
    testlabels1.append(lb)
for val in asd:
    spk1 = ''
    spk2 = ''
    count=0
    fe=[]
    lb=[]


    for item in val:
        fa=[]
        toklist = []
        poslist = []
        toklist2=[]
        poslist2=[]
        spk2=getattr(item, "speaker")
        if count==0:
            fa.append("F")
        if count>0:
            if spk1!=spk2:
                fa.append("SC")

        toks=getattr(item, "pos")

        if toks is not None:
            for x in toks:
                fa.append("TOKEN_"+getattr(x, "token"))
                toklist.append(getattr(x, "token"))



            token_bigrams = zip(toklist, toklist[1:])  # Retrieve Bigrams of TOKENS
            for i in token_bigrams:
                bigram = "/".join(i)
                fa.append("TOKEN_" + bigram)

            token_trigrams = zip(toklist, toklist[1:], toklist[2:])  # Retrieve Trigrams of TOKENS
            for i in token_trigrams:
                trigram = "/".join(i)
                fa.append("TOKEN_" + trigram)

            for x in toks:
                fa.append("POS_" + getattr(x, "pos"))
                poslist.append(getattr(x, "pos"))
            pos_bigrams = zip(poslist, poslist[1:])  # Retrieve Bigrams of POS tags
            for i in pos_bigrams:
                bigram = "/".join(i)
                fa.append("POS_" + bigram)

            pos_trigrams = zip(poslist, poslist[1:], poslist[2:])  # Retrieve Trigrams of POS tags
            for i in pos_trigrams:
                trigram = "/".join(i)
                fa.append("POS_" + trigram)

        lb.append(getattr(item, "act_tag"))
        fe.append(fa)
        count+=1
        spk1=spk2
    testfeatures2.append(fe)
    testlabels2.append(lb)

tagger1 = pycrfsuite.Tagger()
tagger1.open('baseline.crfsuite')
tagger2 = pycrfsuite.Tagger()
tagger2.open('advanced.crfsuite')
sum=0
cor=0
for i in range(len(testfeatures1)):

    pred=tagger1.tag(testfeatures1[i])
    for j in range(len(pred)):
        if pred[j]==testlabels1[i][j]:
            cor+=1
        sum+=1
print("Baseline Acccuracy = "+str(cor*100/sum)+'%\n')
sum=0
cor=0
for i in range(len(testfeatures2)):

    pred=tagger2.tag(testfeatures2[i])
    for j in range(len(pred)):
        if pred[j]==testlabels2[i][j]:
            cor+=1
        sum+=1
print("Advanced Acccuracy = "+str(cor*100/sum)+'%\n')
