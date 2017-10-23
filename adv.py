from modules import *
import torch as T

def adversarial_word_users(wds_b, usrs_b, turns,
           size_wd,batch_size,size_usr,
           sentence_lengths_padded, enc, 
           context,words_padded, decoder, scale = 1e-5):
        
    max_turns = turns.max()
    max_words = wds_b.size()[2]
    encodings = enc(wds_b.view(batch_size * max_turns, max_words, size_wd),
                usrs_b.view(batch_size * max_turns, size_usr), 
                sentence_lengths_padded.view(-1))
    encodings = encodings.view(batch_size, max_turns, -1)
    ctx, _ = context(encodings, turns)
    max_output_words = sentence_lengths_padded[:, 1:].max()
    words_flat = words_padded[:,1:,:max_output_words].contiguous()
    # Training:
    _, log_prob, _ = decoder(ctx[:,:-1:], wds_b[:,1:,:max_output_words],
                             usrs_b[:,1:], sentence_lengths_padded[:,1:], words_flat)
    
    loss = -log_prob
    wds_adv, usrs_adv = T.autograd.grad(loss, [wds_b, usrs_b], grad_outputs=cuda(T.ones(loss.size())), 
                           create_graph=True, retain_graph=True, only_inputs=True)
    wds_adv = (wds_adv > 0).type(T.FloatTensor) * scale - (wds_adv < 0).type(T.FloatTensor) * scale
    usrs_adv = (usrs_adv > 0).type(T.FloatTensor) * scale - (usrs_adv < 0).type(T.FloatTensor) * scale
    wds_adv, usrs_adv = wds_adv.data, usrs_adv.data
    return wds_adv, usrs_adv, tonumpy(loss)[0]
    
def adversarial_encodings_wds_usrs(encodings, batch_size,wds_b,usrs_b,
                      max_turns, context, turns, sentence_lengths_padded,
                      words_padded, decoder, scale=1e-5):
    
    encodings = encodings.view(batch_size, max_turns, -1)
    ctx, _ = context(encodings, turns)
    max_output_words = sentence_lengths_padded[:, 1:].max()
    words_flat = words_padded[:,1:,:max_output_words].contiguous()
    # Training:
    _, log_prob, _ = decoder(ctx[:,:-1:], wds_b[:,1:,:max_output_words],
                             usrs_b[:,1:], sentence_lengths_padded[:,1:], words_flat)
    loss = -log_prob
    wds_adv, usrs_adv, enc_adv = T.autograd.grad(loss, [wds_b,usrs_b,encodings], grad_outputs=cuda(T.ones(loss.size())), 
                           create_graph=True, retain_graph=True, only_inputs=True)
    enc_adv = (enc_adv > 0).type(T.FloatTensor) * scale - (enc_adv < 0).type(T.FloatTensor) * scale
    wds_adv = (wds_adv > 0).type(T.FloatTensor) * scale - (wds_adv < 0).type(T.FloatTensor) * scale
    usrs_adv = (usrs_adv > 0).type(T.FloatTensor) * scale - (usrs_adv < 0).type(T.FloatTensor) * scale
    wds_adv, usrs_adv, enc_adv = wds_adv.data, usrs_adv.data, enc_adv.data
    return wds_adv, usrs_adv, enc_adv, tonumpy(loss)[0]
    
def adversarial_context_wds_usrs(ctx, sentence_lengths_padded,wds_b,usrs_b,
                      words_padded, decoder, scale=1e-7):
    max_output_words = sentence_lengths_padded[:, 1:].max()
    words_flat = words_padded[:,1:,:max_output_words].contiguous()
    # Training:
    _, log_prob, _ = decoder(ctx[:,:-1:], wds_b[:,1:,:max_output_words],
                             usrs_b[:,1:], sentence_lengths_padded[:,1:], words_flat)
    loss = -log_prob
    wds_adv, usrs_adv, ctx_adv = T.autograd.grad(loss, [wds_b,usrs_b,ctx], grad_outputs=cuda(T.ones(loss.size())), 
                           create_graph=True, retain_graph=True, only_inputs=True)
    ctx_adv = (ctx_adv > 0).type(T.FloatTensor) * scale - (ctx_adv < 0).type(T.FloatTensor) * scale
    wds_adv = (wds_adv > 0).type(T.FloatTensor) * scale*100 - (wds_adv < 0).type(T.FloatTensor) * scale*100
    usrs_adv = (usrs_adv > 0).type(T.FloatTensor) * scale*100 - (usrs_adv < 0).type(T.FloatTensor) * scale*100
    wds_adv, usrs_adv, ctx_adv = wds_adv.data, usrs_adv.data, ctx_adv.data
    return wds_adv, usrs_adv, ctx_adv, tonumpy(loss)[0]
    '''
    cls, _, _, nframes = d(data, data_len, embed_d)

    #feature_penalty = [T.pow(r - f,2).mean() for r, f in zip(dists_d, dists_g)]
    loss = binary_cross_entropy_with_logits_per_sample(cls, target, weight=weight) / nframes.float()
    # Check gradient w.r.t. generated output occasionally
    grad = T.autograd.grad(loss, data, grad_outputs=T.ones(loss.size()).cuda(), 
                           create_graph=True, retain_graph=True, only_inputs=True)[0]
                           
    advers = (grad > 0).type(T.FloatTensor) * scale - (grad < 0).type(T.FloatTensor) * scale
    advers = advers.data
    return advers
    '''
