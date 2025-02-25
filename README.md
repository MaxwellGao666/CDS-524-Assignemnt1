# CDS-524-Assignemnt1
# Tetris Reinforcement Learning with Deep Q-Networks (DQN)

This repository contains an implementation of a Deep Q-Network (DQN) to play Tetris using reinforcement learning. The project focuses on training an AI agent to master Tetris by optimizing long-term rewards through iterative learning. The implementation includes enhancements for real-time monitoring, interactive controls, and robust model management.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Testing](#testing)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Reinforcement learning (RL) has shown great potential in solving complex decision-making problems in dynamic environments. This project applies Q-Learning and its advanced variant, Deep Q-Network (DQN), to the classic game of Tetris. The goal is to train an AI agent to play Tetris by learning optimal strategies through trial and error.

## Features
- **Real-Time Monitoring**: A user interface (UI) provides real-time insights into training dynamics, including epoch count, score, buffer status, and exploration rate (ε).
- **Interactive Controls**: Users can manually save models or exit training using keyboard inputs.
- **Robust Model Management**: Models are saved every 1,000 epochs, including optimizer states and performance metrics.
- **Extended Training Support**: The default training duration is extended to 10,000 epochs to accommodate complex policy learning.

## Installation
To set up the environment, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/tetris-dqn.git
   cd tetris-dqn
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Training
To train the DQN model, run the following command:
```bash
python train2.py --width 10 --height 20 --block_size 30 --batch_size 512 --lr 1e-3 --gamma 0.99 --initial_epsilon 1 --final_epsilon 1e-3 --num_decay_epochs 2000 --num_epochs 10000 --save_interval 1000 --replay_memory_size 30000 --log_path tensorboard --saved_path trained_models
```

### Testing
To test a trained model, use the following command:
```bash
python test2.py --model_path trained_models/tetris_auto_9000.pt --width 10 --height 20 --block_size 30 --num_episodes 20 --render --cuda
```

## Results
The performance of the trained models is evaluated based on average score, lines cleared, and survival steps. Below is an example of the results:

| Model               | Average Score | Average Lines Cleared | Average Survival Steps |
|---------------------|---------------|-----------------------|------------------------|
| tetris_auto_1000.pt | 33.1 ± 2.4    | 0.0 ± 0.0             | 35.1 ± 2.4             |
| tetris_auto_3000.pt | 3249.4 ± 2089.7 | 191.1 ± 126.6         | 514.9 ± 316.9          |
| tetris_auto_6000.pt | 14924.8 ± 20446.8 | 832.9 ± 1195.1        | 2116.8 ± 2986.6        |
| tetris_auto_9000.pt | 2065.7 ± 1729.7 | 90.8 ± 73.4           | 261.6 ± 183.3          |

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Note**: Experimental data and results are subject to change based on further testing and optimization.
