
###BLEU
from nltk.translate import bleu_score

hypothesis = ['It', 'is', 'a', 'cat', 'at', 'room']
reference = ['It', 'is', 'a', 'cat', 'inside', 'the', 'room']
reference2 = ['Its', 'a', 'cat', 'inside', 'the', 'room']
#there may be several references, if we can automaticall swap out equivalent phrases
BLEUscore = bleu_score.sentence_bleu([reference, reference2], hypothesis)
print(BLEUscore)

####SKIPTHOUGHTS
'''

Thought Vectors: Maybe useful evaluation metric, but has some problems
Problem: Probably won't work for technical responses.
Other Problem: Probably wont work for paragraphs, only single sentences.
Install: Git pull from master, then run wgets, then point path_to_models and path_to_tables variables in github code to your downloaded files.
Also run nltk.download() select models -> punkt
Then it works! Run the code from github.
Current output showing it works well:
('idx pair', 0, 1, 'dist', 0.89713234)
('idx pair', 0, 2, 'dist', 1.8779925)
('idx pair', 1, 2, 'dist', 1.8767203)

'''
if 0: #Comment out before 
    from skipthoughtsmaster import skipthoughts
    import numpy as np
    model = skipthoughts.load_model()
    encoder = skipthoughts.Encoder(model)
    s1 = 'hi there how are you?'
    s2 = 'hey hows it going?'
    s3 = 'i hate pizza and bears'
    
    sentences = [s1, s2, s3]
    vects = encoder.encode(sentences)
    
    for idx in range(vects.shape[0]):
        for idx2 in range(idx+1,vects.shape[0]):
            v1 = vects[idx,:]
            v2 = vects[idx2,:]
            dist = np.sqrt(np.sum(np.square(v1 - v2)))
            print('idx pair', idx, idx2, 'dist', dist)
    
    a = 2