import numpy as np
# Using pseudo code from P381 in Jurafsky and Martin (2025) to implement the
# algorithm


def viterbi(obs, states, start_prob, trans_prob, emiss_prob):
    """
    Viterbi algorithm for finding the most probable sequence of states
    given a sequence of observations.
    
    Parameters:
    obs: list of observations (indices)
    states: list of possible states (indices)
    start_prob: list of initial probabilities for each state
    trans_prob: 2D list of transition probabilities between states
    emiss_prob: 2D list of emission probabilities for each state
    state and observation
    """
    # intialize the Viterbi matrix and backpointer
    N = len(states)
    T = len(obs)
    viterbi = np.zeros((N, T))
    backpointer = np.zeros((N, T), dtype=int)

    # Intialisation
    for s in range(N):
        viterbi[s, 0] = start_prob[s] * emiss_prob[s][obs[0]]
        backpointer[s, 0] = 0

    # Recursion
    for t in range(1, T):
        for s in range(N):
            max_prob = -1
            best_prev = -1
            for s_prev in range(N):
                prob = viterbi[s_prev, t-1] * \
                    trans_prob[s_prev][s] * emiss_prob[s][obs[t]]
                if prob > max_prob:
                    max_prob = prob
                    best_prev = s_prev
            viterbi[s, t] = max_prob
            backpointer[s, t] = best_prev

    # Termination
    best_path_prob = max(viterbi[:, T-1])
    best_path_pointer = np.argmax(viterbi[:, T-1])
    best_path = [best_path_pointer]

    # Backtrack to find the best path
    for t in range(T-1, 0, -1):
        best_path.insert(0, backpointer[best_path[0], t])

    return best_path, best_path_prob


# Adjust above code for real-world usage
def Viterbi(obs, states, start_prob, trans_prob, emiss_prob):
    """
    Viterbi algorithm for finding the most probable sequence of states
    given a sequence of observations.

    Parameters:
    obs: list of observations
    states: list of possible states
    start_prob: dict mapping states to their initial probabilities
    trans_prob: dict mapping states to their transition probabilities
    emiss_prob: dict mapping states to their emission probabilities
    """
    # change states to indices
    state_to_idx = {state: i for i, state in enumerate(states)}
    
    # Initialize the Viterbi matrix and backpointer
    N, T = len(states), len(obs)
    viterbi = np.zeros((N, T))
    backpointer = np.zeros((N, T), dtype=int)

    # Initialisation
    for s in range(N):
        state = states[s]
        viterbi[s, 0] = start_prob.get(state, 1e-10) * \
            emiss_prob[state].get(obs[0], 1e-10)  # for unknown data
        backpointer[s, 0] = 0

    # Recursion
    for t in range(1, T):
        for s in range(N):
            current_state = states[s]
            max_prob = -1
            best_prev = -1
            for s_prev in range(N):
                prev_state = states[s_prev]
                prob = viterbi[s_prev, t-1] * \
                    trans_prob[prev_state].get(current_state, 1e-10) * \
                    emiss_prob[current_state].get(obs[t], 1e-10)
                if prob > max_prob:
                    max_prob = prob
                    best_prev = s_prev
            viterbi[s, t] = max_prob
            backpointer[s, t] = best_prev

    # Termination
    best_path_prob = max(viterbi[:, T-1])
    best_path_pointer = np.argmax(viterbi[:, T-1])
    best_path = [states[best_path_pointer]]  # return state names instead

    # Backtrack to find the best path
    for t in range(T-1, 0, -1):
        best_path.insert(0, states[backpointer[state_to_idx[best_path[0]], t]])

    return best_path, best_path_prob
