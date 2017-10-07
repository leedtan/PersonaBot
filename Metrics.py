from nltk.translate import bleu_score

hypothesis = ['It', 'is', 'a', 'cat', 'at', 'room']
reference = ['It', 'is', 'a', 'cat', 'inside', 'the', 'room']
reference2 = ['Its', 'a', 'cat', 'inside', 'the', 'room']
#there may be several references, if we can automaticall swap out equivalent phrases
BLEUscore = bleu_score.sentence_bleu([reference, reference2], hypothesis)
print(BLEUscore)

