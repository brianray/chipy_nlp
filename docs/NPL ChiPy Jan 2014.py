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
    elif lang_code == 'pt':
        content_tags = ("\+n", "\+adj", "\+v")
    elif lang_code == "es":
        content_tags = ("^v", "^a", "^n")
    return len(filter(lambda x: matches(x[1], content_tags), data)) / float(len(data))

assert density('en', [(1, "NN"), (2, "XX")]) == .5
assert matches("H+n", ("n", "adj"))
assert density('pt', [(1, "H+n"), (2, "H+xxx")]) == .5
assert density('es', [(1, "xxxx"), (2, "vmip3s0")]) == .5

# <markdowncell>

# back to our original example, one phrase is more dense than the other.

# <codecell>

assert density("en", nltk.pos_tag(tokenize("If I were you I wouldn’t do that with these."))) \
       < density("en", nltk.pos_tag(tokenize("The quick brown fox jumps over the lazy dog.")))

# <markdowncell>

# Tag Spanish Text
# ================

# <codecell>

from nltk.corpus import cess_esp
sents = cess_esp.tagged_sents()

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
'accuracy %.1f %%' % (spanish_tagger.evaluate(test) * 100)

# <codecell>

spanish_tagger.tag(tokenize("A buen entendedor, pocas palabras bastan."))

# <codecell>

spanish_tagger.tag(tokenize("El gato blanco se sentó en la alfombra."))

# <markdowncell>

# Now Portuguese 

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

# Cross Language Testing
# ======================
# 
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

# <codecell>

import numpy as np

for data_dict in strings:
    en_density = density("en", nltk.pos_tag(tokenize(data_dict['en_source'])))
    print "english", en_density
    es_densities = np.mean([density("es", spanish_tagger.tag(tokenize(x))) for x in data_dict['es']])
    print "spanish", es_densities
    pt_densities = np.mean([density("pt", port_tagger.tag(tokenize(x))) for x in data_dict['pt']])
    print "portuguese", pt_densities

# <codecell>

.59 - .43

# <codecell>

.59 - .37

# <codecell>

.43 - .37

# <markdowncell>

# Other fun stuff
# ===============
# http://nltk.org/book/
# 
#  	Natural Language Processing with Python
# --- Analyzing Text with the Natural Language Toolkit
# 
# Steven Bird, Ewan Klein, and Edward Loper
# 

# <codecell>

from nltk.book import *

# <codecell>

nltk.book.text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])

# <codecell>

def tabulate(cfdist, words, categories):
    print '%-16s' % 'Category',
    for word in words:                                  # column headings
        print '%6s' % word,
    print
    for category in categories:
        print '%-16s' % category,                       # row heading
        for word in words:                              # for each word
            print '%6d' % cfdist[category][word],       # print table cell
        print                                           # end the row

# <codecell>

from nltk.corpus import brown

cfd = nltk.ConditionalFreqDist(
          (genre, word)
          for genre in brown.categories()
          for word in brown.words(categories=genre))
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
tabulate(cfd, modals, genres)

# <codecell>

text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
text.similar('woman')

# <codecell>

text.similar('bought')

# <markdowncell>

# grammer and logic

# <codecell>

v = """
bertie => b
olive => o
cyril => c
boy => {b}
girl => {o}
dog => {c}
walk => {o, c}
see => {(b, o), (c, b), (o, c)}
"""
val = nltk.parse_valuation(v)
g = nltk.Assignment(val.domain)
m = nltk.Model(val.domain, val)
sent = 'Cyril sees every boy'
grammar_file = 'grammars/book_grammars/simple-sem.fcfg'
results = nltk.batch_evaluate([sent], grammar_file, m, g)[0]
for (syntree, semrep, value) in results:
    print semrep
    print value

# <codecell>

sent = 'Cyril sees a boy'
results = nltk.batch_evaluate([sent], grammar_file, m, g)[0]
for (syntree, semrep, value) in results:
    print semrep
    print value

# <codecell>

nltk.download()

# <codecell>


