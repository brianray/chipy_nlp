# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# Goals of this talk
# ==================
# 
# * Explore NLTK
# * Have fun thinking about Lexical Graphs
# * Explore the posiblity of comparing differences between langaugas using NLP
# 

# <codecell>

import nltk

# <markdowncell>

# nltk ships with a download utility for downloading grammers and corpora.

# <codecell>

nltk.download()

# <markdowncell>

# Density
# =======

# <codecell>

from nltk import word_tokenize as tokenize

# <codecell>

nltk.pos_tag(tokenize("The quick brown fox jumps over the lazy dog."))

# <codecell>

nltk.pos_tag(tokenize("If I were you I wouldn't do that with these."))

# <markdowncell>

# Create a density checker

# <codecell>

import re

matches = lambda x, re_parts: any([re.findall(y, x) for y in re_parts])

def density(lang_code, data):
    if lang_code == 'en':
        content_tags = ("^NN", "^JJ", "^V")    
    elif lang_code in ('pt', 'es'):
        content_tags = ("\+n", "\+adj", "\+v")
    return len(filter(lambda x: matches(x[1], content_tags), data)) / float(len(data))

assert density('en', [(1, "NN"), (2, "XX")]) == .5
assert matches("H+n", ("n", "adj"))
assert density('pt', [(1, "H+n"), (2, "H+xxx")]) == .5

# <markdowncell>


# <markdowncell>

# Tag Spanish Text
# ================

# <rawcell>

# from nltk.corpus import cess_esp
# sents = cess_esp.tagged_sents()

# <markdowncell>

# Split into training and test set

# <codecell>

training_dx = int(len(sents)*90/100)
training = sents[:training_dx]
test = sents[training_dx+1:]

# <markdowncell>

# train tagger and check accuracy (this takes 40 seconds or so) ...

# <codecell>

from nltk import HiddenMarkovModelTagger
spanish_tagger = HiddenMarkovModelTagger.train(training)
'accuracy %.1f %%' % (tagger.evaluate(test) * 100)

# <codecell>

spanish_tagger.tag(tokenize("A buen entendedor, pocas palabras bastan."))

# <markdowncell>

# Now Portuguese 

# <codecell>

nltk.download()

# <codecell>

from nltk.corpus import floresta
sents = floresta.tagged_sents()

# Split
training_dx = int(len(sents)*90/100)
training = sents[:training_dx]
test = sents[training_dx+1:]

#train
port_tagger = HiddenMarkovModelTagger.train(training)
'accuracy %.1f %%' % (port_tagger.evaluate(test) * 100)

# <markdowncell>

# Language Test data sent from Transifex

# <codecell>


strings = [dict(
           en_source="Without a secure random number generator an attacker"
                     " may be able to predict password reset tokens and take"
                     " over your account.",
           pt=[
               # portuguese translation 1
               "Sem nenhum gerador seguro de números aleatórios, uma pessoa mal"
               " intencionada pode prever a sua password, reiniciar as seguranças"
               " adicionais e tomar conta da sua conta."
               ],
           es=[
               # spanish translation 1
               "Sin un generador de números aleatorios seguro, un atacante podría"
               " predecir los tokens de restablecimiento de contraseñas y tomar"
               " el control de su cuenta.",
               # spanish translation 2
               "Sin un generador de números aleatorios seguro un atacante podría"
               " predecir los tokens de reinicio de su contraseña y tomar control"
               " de su cuenta."
               
               ])]

# <markdowncell>

# back to our original example, one phrase is more dense than the other.

# <codecell>

assert density("en", nltk.pos_tag(tokenize("If I were you I wouldn’t do that with these."))) \
       < density("en", nltk.pos_tag(tokenize("The quick brown fox jumps over the lazy dog.")))

# <codecell>


port_tagger.tag(tokenize(strings[0]['pt'][0]))



# <codecell>


