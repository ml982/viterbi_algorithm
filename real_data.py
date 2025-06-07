import nltk
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from viterbi_algorithm.algorithm import Viterbi

# Viterbi algorithm for speech tagging using real-world data


# emission probability
def get_emission_prob(train_data):
    """
    Calculate the emission probabilities from the training data.
    Parameters:
    train_data: list of sentences, where each sentence is a list of
    (word, tag) tuples.
    """
    emission_count = defaultdict(Counter)
    tags_count = Counter()

    for sent in train_data:
        for (word, tag) in sent:
            emission_count[tag][word] += 1
            tags_count[tag] += 1

    emission_prob = {
        tag: {
            word: count / tags_count[tag]
            for word, count in word_dict.items()
        }
        for tag, word_dict in emission_count.items()
    }

    return emission_prob


# transition probability
def get_transition_prob(train_data):
    """
    Calculate the transition probabilities from the training data.

    Parameters:
    train_data: list of sentences, where each sentence is a list of
    (word, tag) tuples.
    """
    transition_count = defaultdict(Counter)
    start_count = Counter()

    for sent in train_data:
        prev_tag = None
        for (word, tag) in sent:
            if prev_tag is None:
                start_count[tag] += 1
            else:
                transition_count[prev_tag][tag] += 1
            prev_tag = tag

    transition_prob = {
        prev_tag: {
            curr_tag: count / sum(next_tags.values())
            for curr_tag, count in next_tags.items()
        }
        for prev_tag, next_tags in transition_count.items()
    }

    return transition_prob


# initial probability
def get_start_prob(train_data):
    """
    Calculate the initial probabilities from the training data.

    Parameters:
    train_data: list of sentences, where each sentence is a list of
    (word, tag) tuples.
    """
    start_count = Counter()

    for sent in train_data:
        if sent:
            start_count[sent[0][1]] += 1

    total_starts = sum(start_count.values())
    start_prob = {tag: count / total_starts for tag,
                  count in start_count.items()}

    return start_prob


# compare the accuracy
def compute_accuracy_over_test(test_data, states, start_probs, trans_probs,
                               emiss_probs):
    """
    Compute the accuracy of the Viterbi algorithm over a test dataset.

    Parameters:
    test_data: list of sentences, where each sentence is a list of
    (word, tag) tuples.
    states: list of possible states.
    start_probs: dict mapping states to their initial probabilities.
    trans_probs: dict mapping states to their transition probabilities.
    emiss_probs: dict mapping states to their emission probabilities.
    """
    total_correct = 0
    total_tags = 0

    for sent in test_data:
        words = [word for word, _ in sent]
        obs = [w.lower() for w in words]
        gold_tags = [tag for _, tag in sent]

        predicted_tags = Viterbi(obs, states, start_probs, trans_probs,
                                 emiss_probs)[0]
        # Compare predicted to gold
        for pred, gold in zip(predicted_tags, gold_tags):
            if pred == gold:
                total_correct += 1
            total_tags += 1

    return total_correct / total_tags if total_tags > 0 else 0.0


# get data from nltk corpus - use universal, treebank, browncorpus tagsets
tagged_sents = nltk.corpus.treebank.tagged_sents(tagset='universal')
tagged_sents2 = nltk.corpus.treebank.tagged_sents(tagset='brown')


train_data1, test_data1 = train_test_split(tagged_sents,
                                           test_size=0.15, random_state=42)

emiss1 = get_emission_prob(train_data1)
trans1 = get_transition_prob(train_data1)
initial1 = get_start_prob(train_data1)
states1 = list(emiss1.keys())
accuracy1 = compute_accuracy_over_test(test_data1, states1, initial1, trans1,
                                       emiss1)
print(f"POS tagging accuracy: {accuracy1:.2%}")  # accuracy = 88.75%
