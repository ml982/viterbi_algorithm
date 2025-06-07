from viterbi_algorithm.algorithm import Viterbi

# Example sentence
sentence = "the cat ate a fish"
# there are 2 determiners (the, and), 2 nouns (cat, fish), 1 verb (ate)

# states
states = ['NOUN', 'VERB', 'DET']
states_matrix = [0, 1, 2]

# observations
words = ['the', 'cat', 'ate', 'a', 'fish']
words_matrix = [i for i in range(len(words))]

# transition probabilities
transition_probs = {
    'NOUN':  {'NOUN': 0.1, 'VERB': 0.6, 'DET': 0.3},
    'VERB':  {'NOUN': 0.4, 'VERB': 0.1, 'DET': 0.5},
    'DET':   {'NOUN': 0.9, 'VERB': 0.1},
}

trans_matrix = [
    [0.1, 0.6, 0.3],
    [0.4, 0.1, 0.5],
    [0.9, 0.1, 0.0]
]

# emission probabilities
emission_probs = {
    'NOUN': {
        'cat': 0.5,
        'fish': 0.4,
        'ate': 0.1,
    },
    'VERB': {
        'ate': 0.8,
        'fish': 0.2,
    },
    'DET': {
        'the': 0.6,
        'a': 0.4,
    }
}

emiss_matrix = [
    [0, 0.5, 0.1, 0, 0.4],
    [0, 0, 0.8, 0, 0.2],
    [0.6, 0, 0, 0.4, 0]
]

# initial probabilities
start_probs = {'NOUN': 0.2, 'VERB': 0.0, 'DET': 0.8}
start_matrix = [0.2, 0, 0.8]

best_path = Viterbi(words, states,
                    start_probs, transition_probs, emission_probs)[0]
print(best_path)  # Output: ['DET', 'NOUN', 'VERB', 'DET', 'NOUN']
