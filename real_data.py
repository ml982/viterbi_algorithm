import nltk
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
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

# Calculate for 15% of the data
train_data1, test_data1 = train_test_split(tagged_sents,
                                           test_size=0.15, random_state=42)
emiss1 = get_emission_prob(train_data1)
trans1 = get_transition_prob(train_data1)
initial1 = get_start_prob(train_data1)
states1 = list(emiss1.keys())
accuracy1 = compute_accuracy_over_test(test_data1, states1, initial1, trans1,
                                       emiss1)  # accuracy = 88.75%

# Expand for more sizes of test data
test_sizes = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
              0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
train_data_size = []
accuracies = []
for size in test_sizes:
    train_data, test_data = train_test_split(tagged_sents,
                                             test_size=size, random_state=42)
    train_data_size.append(1 - size)
    emiss_prob = get_emission_prob(train_data)
    trans_prob = get_transition_prob(train_data)
    initial_prob = get_start_prob(train_data)
    states = list(emiss_prob.keys())
    accuracy = compute_accuracy_over_test(test_data, states, initial_prob,
                                          trans_prob, emiss_prob)
    accuracies.append(accuracy)
    print(f"Training data size: {1 - size}, Accuracy: {accuracy:.2%}")

# Plotting the results
plt.plot(train_data_size, accuracies, marker='o')
plt.xlabel('Training Data Size')
plt.ylabel('Accuracy')
plt.axis((0, 1, 0.78, 0.9))
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0.78, 0.91, 0.01))
plt.title('HMM Accuracy using Viterbi Algorithm vs Training Data Size')
plt.grid()
plt.show()
