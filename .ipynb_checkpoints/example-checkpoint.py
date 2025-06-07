# Python code snippet
# Define the hidden Markov model
state_transition_probabilities = {('Sunny', 'Sunny'): 0.7, ('Sunny', 'Rainy'): 0.3, ('Rainy', 'Sunny'): 0.4, ('Rainy', 'Rainy'): 0.6}
observation_emission_probabilities = {('Sunny', 'Happy'): 0.8, ('Sunny', 'Sad'): 0.2, ('Rainy', 'Happy'): 0.4, ('Rainy', 'Sad'): 0.6}
initial_state_probabilities = {'Sunny': 0.6, 'Rainy': 0.4}

# Observed data
observations = ['Happy', 'Happy', 'Sad']

# Initialize the probability matrix
probability_matrix = {}

# Forward pass to fill the probability matrix
for time_step, observation in enumerate(observations):
    for state in state_transition_probabilities.keys():
        if time_step == 0:
            # For the first time step, use the initial state probabilities
            probability_matrix[state] = initial_state_probabilities[state] * observation_emission_probabilities[state, observation]
        else:
            # For subsequent time steps, calculate the maximum probability from previous states
            probability_matrix[state] = max(
                probability_matrix[prev_state] * state_transition_probabilities[prev_state, state] * observation_emission_probabilities[state, observation]
                for prev_state in state_transition_probabilities.keys()
            )

# Backtrack to find the most likely sequence of hidden states
sequence = []
current_state = max(probability_matrix, key=probability_matrix.get)
for time_step in range(len(observations) - 1, -1, -1):
    sequence.insert(0, current_state)
    current_state = max(
        state for state in state_transition_probabilities.keys() if probability_matrix[state] * state_transition_probabilities[state, current_state] * observation_emission_probabilities[current_state, observations[time_step]] == probability_matrix[current_state]
    )

print("Most likely sequence of hidden states:", sequence)