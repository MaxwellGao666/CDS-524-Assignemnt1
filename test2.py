import argparse
import os
import pygame
import numpy as np
import torch
from src.tetris import Tetris
from src.deep_q_network import DeepQNetwork  # Ensure this path matches your project structure

def get_test_args():
    """Get testing parameters"""
    parser = argparse.ArgumentParser(description="Test trained Tetris model")
    parser.add_argument("--model_path", type=str, required=True, 
                      help="Path to trained model (.pt file)")
    parser.add_argument("--width", type=int, default=10, help="Game grid width")
    parser.add_argument("--height", type=int, default=20, help="Game grid height")
    parser.add_argument("--block_size", type=int, default=30, help="Block pixel size")
    parser.add_argument("--num_episodes", type=int, default=10, 
                      help="Number of test episodes")
    parser.add_argument("--render", action="store_true", 
                      help="Enable visual rendering")
    parser.add_argument("--cuda", action="store_true",
                      help="Enable CUDA acceleration")
    return parser.parse_args()

def test(args):
    # Initialize environment
    pygame.init()
    env = Tetris(width=args.width, 
                height=args.height, 
                block_size=args.block_size)
    
    # Initialize model
    model = DeepQNetwork()
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state'])
    
    if args.cuda and torch.cuda.is_available():
        model.cuda()
        print("Using CUDA acceleration")
    model.eval()  # Set to evaluation mode

    # Metrics tracking
    total_scores = []
    total_lines = []
    total_steps = []
    
    # Create display window if rendering
    if args.render:
        screen_width = args.width * args.block_size
        screen_height = args.height * args.block_size + 60
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Tetris DQN Testing")

    # Testing loop
    for episode in range(args.num_episodes):
        state = env.reset()
        done = False
        score = 0
        lines = 0
        steps = 0
        
        while not done:
            if args.render:
                # Handle quit event
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                
                # Render game state
                screen.fill((0, 0, 0))
                env.render(screen)
                pygame.display.flip()

            # Get possible actions
            next_steps = env.get_next_states()
            if not next_steps:
                break  # Terminate if no valid moves
            
            # Prepare input states
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states)
            if args.cuda and torch.cuda.is_available():
                next_states = next_states.cuda()

            # Model prediction
            with torch.no_grad():
                predictions = model(next_states)[:, 0]
            
            # Select best action
            best_idx = torch.argmax(predictions).item()
            best_action = next_actions[best_idx]

            # Execute action
            reward, done = env.step(best_action, render=args.render)
            
            # Update metrics
            score = env.score
            lines = env.cleared_lines
            steps += 1

        # Record episode results
        total_scores.append(score)
        total_lines.append(lines)
        total_steps.append(steps)
        
        # Print episode summary
        print(f"Episode {episode+1}/{args.num_episodes}")
        print(f"▷ Score: {score} | Lines cleared: {lines} | Steps survived: {steps}")
        print("-" * 40)

    # Final statistics
    print("\n=== Test Results ===")
    print(f"Average score: {np.mean(total_scores):.1f} ± {np.std(total_scores):.1f}")
    print(f"Average lines: {np.mean(total_lines):.1f} ± {np.std(total_lines):.1f}")
    print(f"Average steps: {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
    
    pygame.quit()

if __name__ == "__main__":
    args = get_test_args()
    test(args)