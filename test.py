
import torch as T
from modules import *
from collections import Iterable, namedtuple
from numbers import Integral

BeamItem = namedtuple('BeamItem', ['parent', 'token', 'score', 'state', 'complete'])
one = tovar(T.LongTensor([1]))

def beam_search(dataset,
                decoder,
                word_embedder,
                context_encoding,
                user_emb,
                max_sentence_length,
                beam_width,
                ):
    '''
    Returns: a scalar indicating score (log-likelihood) and a list of ints
    '''
    beam = [BeamItem(
                parent=None,
                token=dataset.start_token_index,
                score=0,
                state=decoder.zero_state(one),  # n_layers, batch_size, state_size
                complete=False,
                )
            for _ in range(beam_width)]
    beam_list = [beam]

    context_encodings = context_encoding.expand(beam_width, 1, context_encoding.size()[2])
    user_embs = user_emb.expand(beam_width, 1, user_emb.size()[2])
    ones = one.unsqueeze(0).expand(beam_width, 1)

    early_stop = False

    for _ in range(max_sentence_length):
        last_beam = beam_list[-1]

        # Heuristic: if the top candidate yields a <EOS> token, stop there.
        # This is some sort of balance between sentence length (shorter the better)
        # and likelihood.
        if last_beam[0].complete:
            early_stop = True
            break

        # Get the decoder inputs from last beam and concatenate them into a batch
        current_word_indices = tovar(T.LongTensor([[b.token] for b in last_beam]))
        decoder_states = T.cat([b.state for b in last_beam], 1)
        word_embs = word_embedder(current_word_indices).unsqueeze(1)

        # Unroll a step
        logprob, decoder_states = decoder(
                context_encodings,
                word_embs,
                user_embs,
                ones,
                initial_state=decoder_states
                )

        # Take top k words from each beam item and form a candidate list with k^2 items
        logprob_topk_score, logprob_topk_indices = logprob[:, 0, 0].data.topk(beam_width, 1)
        candidates = []
        for i in range(beam_width):
            for j in range(beam_width):
                token = logprob_topk_indices[i, j]
                # If the candidate is marked as complete, do not accumulate score.
                score = last_beam[i].score + (
                        logprob_topk_score[i, j] if not last_beam[i].complete else 0)
                candidates.append(BeamItem(
                    parent=last_beam[i],
                    token=token,
                    score=score,
                    state=decoder_states[:, i],
                    complete=last_beam[i].complete or token == dataset.end_token_index
                    ))

        # Take top k candidates and form the next beam
        candidates_topk = sorted(candidates, key=lambda item: item.score, reverse=True)
        beam_list.append(list(candidates_topk))

    last_beam = beam_list[-1]
    if early_stop:
        # Heuristic triggered, take the complete sentence
        current_item = last_beam[0]
    else:
        # Maybe the top one is a partial sentence but there are complete sentences.
        # Prefer those instead.
        for i in range(beam_width):
            if last_beam[i].complete:
                current_item = last_beam[i]
                break
        else:
            # All of them are partial sentences
            current_item = last_beam[0]

    # If current item is marked as complete, roll back until we find the <EOS> token
    if current_item.complete:
        while current_item.token != dataset.end_token_index:
            current_item = current_item.parent

    # Roll back until the first item
    score = current_item.score
    tokens = []
    while current_item is not None:
        tokens.insert(0, current_item.token)
        current_item = current_item.parent

    return score, tokens


def test(dataset,
         encoder,
         context_net,
         decoder,
         word_embedder,
         user_embedder,
         sentences,
         initiator,
         respondent,
         max_sentence_length,
         beam_width=5,
         context_state=None,
         hallucinate=False):
    '''
    sentences: list of list of ints
        representing utterances from the same user
    initiators: int
        the users initiating the dialogue
    respondents: int
        the users responding to the dialogue
    hallucinate: bool
        whether we want to generate the counter-responses from the initiator as well.
        If True, the second item and afterwards in @sentences are ignored.  You still

    Note:
    batch size is always 1
    batch-wise beam search is too complicated.
    '''
    def sentence_to_vars(_sentence):
        sentence = tovar(T.LongTensor([_sentence]))
        sentence_length = tovar(T.LongTensor([len(_sentence)]))
        sentence.volatile = True
        return sentence, sentence_length

    assert isinstance(sentences, Iterable)
    assert isinstance(initiator, Integral)
    assert isinstance(respondent, Integral)
    dialogue = []
    scores = []

    initiator = tovar(T.LongTensor([initiator]))
    respondent = tovar(T.LongTensor([respondent]))
    initiator.volatile = True       # Don't save graph
    respondent.volatile = True

    initiator_embed = user_embedder(initiator.unsqueeze(1))
    respondent_embed = user_embedder(respondent.unsqueeze(1))
    if context_state is None:
        context_state = context_net.zero_state(one)

    _sentence = sentences[0]
    scores = 0

    for i in range(len(sentences)):
        dialogue.append(_sentence)
        scores.append(score)
        sentence, sentence_length = sentence_to_vars(_sentence)

        # Encode the current sentence from the initiator
        sentence_embed = word_embedder(sentence)
        sentence_encoding = encoder(sentence_embed, initiator_embed, sentence_length)

        # Mix the sentence encoding with last context, take one step of Context RNN
        sentence_encoding = sentence_encoding.unsqueeze(1)
        context_encoding, context_state = context_net(sentence_encoding, one, context_state)

        # Decode the current context using beam search
        score, _sentence = beam_search(
                dataset,
                decoder,
                word_embedder,
                context_encoding,
                respondent_embed,
                max_sentence_length,
                beam_width)
        dialogue.append(_sentence)
        scores.append(score)

        # Encode the response (_sentence)
        sentence, sentence_length = sentence_to_vars(_sentence)
        sentence_embed = word_embedder(sentence)
        sentence_encoding = encoder(sentence_embed, respondent_embed, sentence_length)
        sentence_encoding = sentence_encoding.unsqueeze(1)
        context_encoding, context_state = context_net(sentence_encoding, one, context_state)

        if hallucinate:
            score, _sentence = beam_search(
                    dataset,
                    decoder,
                    word_embedder,
                    context_encoding,
                    initiator_embed,
                    max_sentence_length,
                    beam_width)
        elif i != len(sentences) - 1:
            # Proceed to the next initiator's utterance
            _sentence = sentences[i + 1]
            score = 0

    # Add the last sentence and score
    dialogue.append(_sentence)
    scores.append(score)

    return dialogue, scores
