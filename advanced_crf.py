import pycrfsuite
from collections import namedtuple
import csv
import glob
import os
import ntpath
import sys
import string


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
def get_data2(data_dir):
    """Generates lists of utterances from each dialog file.

    To get a list of all dialogs call list(get_data(data_dir)).
    data_dir - a dir with csv files containing dialogs"""
    dialog_filenames = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    for dialog_filename in dialog_filenames:
        yield dialog_filename, get_utterances_from_filename(dialog_filename)

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



asd=list(get_data("C:/Users/neku104/PycharmProjects/544HW3/train"))
asd2=list(get_data2("C:/Users/neku104/PycharmProjects/544HW3/test"))


features=[]
labels=[]

trainlabels=[]
testlabels=[]
trainfeatures=[]
testfeatures=[]
filelist=[]


for val in asd:
    spk1 = ''
    spk2 = ''
    count=0
    fe=[]
    lb=[]

    for item in val:
        toklist = []
        poslist = []

        fa=[]
        spk2=getattr(item, "speaker")
        if count==0:
            fa.append("F")
            count+=1
        if spk1!=spk2:
            fa.append("SC")

        toks=getattr(item, "pos")

        if toks is not None:
            for x in toks:
                fa.append("TOKEN_"+getattr(x, "token"))
                toklist.append(getattr(x, "token"))
                fa.append("POS_" + getattr(x, "pos"))
                poslist.append(getattr(x, "pos"))
            tb = zip(toklist, toklist[1:])
            for i in tb:
                bigram = "/".join(i)
                fa.append("TOKEN_" + bigram)
            tt = zip(toklist, toklist[1:], toklist[2:])
            for i in tt:
                trigram = "/".join(i)
                fa.append("TOKEN_" + trigram)

            pb = zip(poslist, poslist[1:])
            for i in pb:
                bigram = "/".join(i)
                fa.append("POS_" + bigram)

            pt = zip(poslist, poslist[1:], poslist[2:])
            for i in pt:
                trigram = "/".join(i)
                fa.append("POS_" + trigram)


        lb.append(getattr(item, "act_tag"))
        fe.append(fa)

        spk1=spk2
    trainfeatures.append(fe)
    trainlabels.append(lb)

trainer=pycrfsuite.Trainer(verbose=False)


for x, y in zip(trainfeatures, trainlabels):

    trainer.append(x, y)
trainer.set_params({
    'c1': 1.0,
    'c2': 1e-4,
    'max_iterations': 75,

    'feature.possible_transitions': True
})


trainer.train('advanced.crfsuite')




for val in asd2:
    spk1 = ''
    spk2 = ''
    count=0
    fe=[]
    lb=[]


    for item in val[1]:
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



            token_bigrams = zip(toklist, toklist[1:])
            for i in token_bigrams:
                bigram = "/".join(i)
                fa.append("TOKEN_" + bigram)

            token_trigrams = zip(toklist, toklist[1:], toklist[2:])
            for i in token_trigrams:
                trigram = "/".join(i)
                fa.append("TOKEN_" + trigram)

            for x in toks:
                fa.append("POS_" + getattr(x, "pos"))
                poslist.append(getattr(x, "pos"))
            pos_bigrams = zip(poslist, poslist[1:])
            for i in pos_bigrams:
                bigram = "/".join(i)
                fa.append("POS_" + bigram)

            pos_trigrams = zip(poslist, poslist[1:], poslist[2:])
            for i in pos_trigrams:
                trigram = "/".join(i)
                fa.append("POS_" + trigram)

        lb.append(getattr(item, "act_tag"))
        fe.append(fa)
        count+=1
        spk1=spk2
    testfeatures.append(fe)
    testlabels.append(lb)
    filelist.append(val[0])
print("Learn")


tagger = pycrfsuite.Tagger()
tagger.open('advanced.crfsuite')


f = open("advancedoutput.txt", 'w')
for i in range(len(testfeatures)):
    f.write("Filename=" + ntpath.basename(filelist[i])+'\n')
    pred=tagger.tag(testfeatures[i])
    for j in range(len(pred)):
        f.write(pred[j]+'\n')
    f.write('\n')
f.close()