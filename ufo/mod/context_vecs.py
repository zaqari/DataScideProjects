from CommClusters.embeds.RoBERTa import *
import pandas as pd
import numpy as np

data = ['equals tactile', 'of or relating to or proceeding from the sense of touch equals tactile. producing a sensation of touch equals tactile.']

def avgC(lexeme, context, vecs=vecs, layer_no=9):

    # Query to find relevant data points. This part needs to be retooled
    # according to whatever your search function might be, but whatever
    # that search is, you need to get back the indeces in your array
    # where the search items are located.
    item = lexeme.lower()
    #print(item)
    vectors, lexemes = vecs.translate_chunk(context.lower(), layer_no)

    lexemes = ['SOS'] + lexemes.tolist() + ['EOS']

    candidates = [i for i in range(1, len(lexemes) - 1) if lexemes[i] == item or lexemes[i] == 'Ġ' + item]

    if len(candidates) < 1:
        candidates = [i for i in range(1, len(lexemes) - 1) if ((str(lexemes[i])[1:] in item) and (str(lexemes[i + 1])[1:] in item)) or ((str(lexemes[i])[1:] in item) and (str(lexemes[i - 1])[1:] in item))]

    # Since we have the indeces, we can find contiguous spans in the search
    # results by subtracting the indeces for the index i-1 from i. We set
    # that up via the following, where pseudo-candidates is made up of
    # values i-1, with a nonce, negative number at index 0.
    pseudo_dates = np.array([-4] + candidates[:-1])
    candidates = np.array(candidates)

    # Here's the cool part. Non-contiguous spans will have a value
    # greater than one! So we can do a boolean search to find items
    # that aren't equal to 1.
    delta = (candidates - pseudo_dates) != 1

    # Now, we can re-order our search results to combine contiguous
    # spans by reshaping the matrix to be shape delta.sum() and -1!

    #NOTE: This next addition is happening because BERT is a fucker.
    #      Otherwise, skip "try" and stick with the first lexical_indeces
    lexical_indeces = None
    try:
        lexical_indeces = candidates.reshape(delta.sum(), -1)
    except ValueError:
        lexical_indeces = candidates[:int(len(candidates)-(len(candidates)%delta.sum()))].reshape(delta.sum(), -1)

    # Specific for this task, but this takes the vectors we need,
    # sums the contiguous spans, and then in the last step, takes
    # the mean of those spans. This was our objective the whole
    # time.
    new_outs = torch.zeros(size=(lexical_indeces.shape[0], 768))
    for i, idx in enumerate(lexical_indeces):
        try:
            #new_outs[i] = vectors[idx].sum(dim=0)

            #Mean rather than sum of multiple subtokens
            new_outs[i] = vectors[idx].mean(dim=0)

            #Last of multiple subtokens new_outs[i] = vectors[idx[-1]].mean(dim=0)
        except IndexError:
            # new_outs[i] = vectors[idx].sum(dim=0)

            # Mean rather than sum of multiple subtokens
            'NADA'#vectors[idx[:-1]].mean(dim=0)

            # Last of multiple subtokens new_outs[i] = vectors[idx[-1]].mean(dim=0)
    new_outs = new_outs[new_outs.sum(dim=1) != 0]
    return new_outs.mean(dim=0).view(-1)

def nC(lexeme, context, vecs=vecs, layer_no=0):
    # Query to find relevant data points. This part needs to be retooled
    # according to whatever your search function might be, but whatever
    # that search is, you need to get back the indeces in your array
    # where the search items are located.
    item = lexeme.lower()
    vectors, lexemes = vecs.translate_chunk(context.lower(), layer_no)

    lexemes = ['SOS'] + lexemes.tolist() + ['EOS']

    candidates = [i for i in range(1, len(lexemes) - 1)
                  if lexemes[i] == item
                  or lexemes[i] == 'Ġ' + item]

    if len(candidates) < 1:
        candidates = [i for i in range(1, len(lexemes) - 1)
                      if str(lexemes[i] + lexemes[i + 1])[1:] in item
                      or str(lexemes[i - 1] + lexemes[i])[1:] in item]

    # Since we have the indeces, we can find contiguous spans in the search
    # results by subtracting the indeces for the index i-1 from i. We set
    # that up via the following, where pseudo-candidates is made up of
    # values i-1, with a nonce, negative number at index 0.
    pseudo_dates = np.array([-4] + candidates[:-1])
    candidates = np.array(candidates)

    # Here's the cool part. Non-contiguous spans will have a value
    # greater than one! So we can do a boolean search to find items
    # that aren't equal to 1.
    delta = (candidates - pseudo_dates) != 1

    # Now, we can re-order our search results to combine contiguous
    # spans by reshaping the matrix to be shape delta.sum() and -1!

    # NOTE: This next addition is happening because BERT is a fucker.
    #      Otherwise, skip "try" and stick with the first lexical_indeces
    lexical_indeces = None
    try:
        lexical_indeces = candidates.reshape(delta.sum(), -1)
    except ValueError:
        lexical_indeces = candidates[:int(len(candidates) - (len(candidates) % delta.sum()))].reshape(delta.sum(), -1)

    # Specific for this task, but this takes the vectors we need,
    # sums the contiguous spans, and then in the last step, takes
    # the mean of those spans. This was our objective the whole
    # time.
    new_outs = torch.zeros(size=(lexical_indeces.shape[0], 768))
    for i, idx in enumerate(lexical_indeces):
        try:
            new_outs[i] = vectors[idx].mean(dim=0)
        except IndexError:
            new_outs[i] = torch.zeros(size=(768,))

    new_outs = new_outs[new_outs.sum(dim=1) != 0]

    return new_outs

def multiple_nC(w, context, vecs=vecs, layer_no=0):
    # Query to find relevant data points. This part needs to be retooled
    # according to whatever your search function might be, but whatever
    # that search is, you need to get back the indeces in your array
    # where the search items are located.
    item = [lexeme.lower() for lexeme in w]
    vectors, lexemes = vecs.translate_chunk(context.lower(), layer_no)

    lexemes = ['SOS'] + lexemes.tolist() + ['EOS']

    candidates = [i for i in range(1, len(lexemes) - 1)
                  if lexemes[i] == item
                  or lexemes[i] == 'Ġ' + item]

    if len(candidates) < 1:
        candidates = [i for i in range(1, len(lexemes) - 1)
                      if str(lexemes[i] + lexemes[i + 1])[1:] in item
                      or str(lexemes[i - 1] + lexemes[i])[1:] in item]

    # Since we have the indeces, we can find contiguous spans in the search
    # results by subtracting the indeces for the index i-1 from i. We set
    # that up via the following, where pseudo-candidates is made up of
    # values i-1, with a nonce, negative number at index 0.
    pseudo_dates = np.array([-4] + candidates[:-1])
    candidates = np.array(candidates)

    # Here's the cool part. Non-contiguous spans will have a value
    # greater than one! So we can do a boolean search to find items
    # that aren't equal to 1.
    delta = (candidates - pseudo_dates) != 1

    # Now, we can re-order our search results to combine contiguous
    # spans by reshaping the matrix to be shape delta.sum() and -1!

    # NOTE: This next addition is happening because BERT is a fucker.
    #      Otherwise, skip "try" and stick with the first lexical_indeces
    lexical_indeces = None
    try:
        lexical_indeces = candidates.reshape(delta.sum(), -1)
    except ValueError:
        lexical_indeces = candidates[:int(len(candidates) - (len(candidates) % delta.sum()))].reshape(delta.sum(), -1)

    # Specific for this task, but this takes the vectors we need,
    # sums the contiguous spans, and then in the last step, takes
    # the mean of those spans. This was our objective the whole
    # time.
    new_outs = torch.zeros(size=(lexical_indeces.shape[0], 768))
    for i, idx in enumerate(lexical_indeces):
        try:
            new_outs[i] = vectors[idx].mean(dim=0)
        except IndexError:
            new_outs[i] = torch.zeros(size=(768,))

    new_outs = new_outs[new_outs.sum(dim=1) != 0]

    return new_outs