# Chatbot AI Agent Deva

This project explores the development of a chatbot, Deva, using a **Reinforcement Learning (RL)** approach. Unlike traditional intent-based systems, this implementation casts the chatbot interaction as an environment where an agent learns to select optimal responses through trial and error, guided by rewards. It leverages **BERT embeddings** for state representation and integrates with the **Gymnasium** (formerly Gym) environment for RL training.

-----

## Features

  * **Reinforcement Learning Environment:** Defines a custom `EnhancedChatbotEnv` based on Gymnasium, allowing the chatbot to learn conversational policies through interactions.
  * **BERT-based State Representation:** Utilizes a pre-trained **BERT model (`bert-base-uncased`)** to convert user input and session history into dense vector embeddings, providing rich contextual information to the RL agent.
  * **Session History Integration:** Incorporates a simplified session history into the state observation, aiming to give the agent context from recent interactions.
  * **Reward Mechanism:** Implements a basic reward system where the agent receives a reward of `1` for choosing the "expected" response and `0` otherwise, encouraging it to select the correct answers from the dataset.
  * **`stable-baselines3` Integration:** Sets up the environment for training a reinforcement learning agent, specifically using the **PPO (Proximal Policy Optimization)** algorithm from `stable-baselines3`.

-----

## Setup and Installation

### Prerequisites

  * Python 3.x
  * Google Colab (recommended for seamless GPU access and dependency management)

### Libraries

You can install the required libraries using pip:

```bash
pip install gym openpyxl transformers torch numpy stable_baselines3 shimmy
```

*Note: The code explicitly installs `shimmy` for `gym` compatibility with newer `gymnasium` versions, which is good practice.*

### Data

The project requires an Excel file named `dummy data.xlsx`. This file is expected to be located in your Google Drive at `/content/drive/MyDrive/Colab Notebooks/Chatbot deva Ai agent0/`. The dataset should contain at least three columns: `User Input`, `Bot Response`, and `Intent`.

-----

## Usage

The provided code is structured as a Google Colab notebook, demonstrating the setup of the RL environment and the initiation of training.

### 1\. Mount Google Drive

The first step in the notebook is to mount your Google Drive to access the dataset:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2\. Install Dependencies

Ensure all necessary libraries are installed by running the `pip install` commands.

### 3\. Load Data and Initialize BERT

The `dummy data.xlsx` file is loaded into a Pandas DataFrame, and the BERT tokenizer and model are initialized for generating text embeddings.

### 4\. Define the Chatbot RL Environment

The `EnhancedChatbotEnv` class is the core of the RL setup. It defines:

  * **`__init__`**: Initializes the environment with your dataset, action space (unique bot responses), and observation space (BERT embedding dimension + history embedding).
  * **`reset()`**: Resets the environment to an initial state at the beginning of an episode.
  * **`step(action)`**: Takes an action (a chosen bot response), calculates a reward, updates the environment's state, and determines if the episode is `done`.
  * **`_get_state(idx)`**: Generates the observation state, which includes the BERT embedding of the current user input concatenated with a simplified embedding of the session history.

### 5\. Train the RL Agent

The `stable-baselines3` library is used to define and train the RL agent. In this example, the **PPO algorithm** with an `MlpPolicy` is chosen:

```python
from stable_baselines3 import PPO

# Initialize the environment
env = EnhancedChatbotEnv(df)

# Create the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Start training the model
model.learn(total_timesteps=10000)

# Save the trained model
model.save("enhanced_chatbot_rl_model")
```

*Note: The provided code snippet for `model.learn()` seems to have been interrupted (`KeyboardInterrupt`). For successful training, this cell needs to run to completion.*

-----

## Next Steps

To fully realize the potential of this RL chatbot, you might consider:

  * **More Sophisticated Reward Functions:** Instead of a simple `1` or `0`, consider rewards that account for conversational flow, sentiment, or user satisfaction (e.g., using a separate sentiment analysis model).
  * **Advanced State Representation:** Experiment with different ways to encode session history or user profiles into the observation space.
  * **Hyperparameter Tuning:** Optimize the hyperparameters for the PPO agent (or other RL algorithms) for better performance.
  * **Deployment:** After training, integrate the learned policy (`enhanced_chatbot_rl_model.zip`) into a deployable chatbot application.
  * **Evaluation Metrics:** Define metrics to evaluate the chatbot's performance in terms of coherence, helpfulness, and user engagement.

-----
