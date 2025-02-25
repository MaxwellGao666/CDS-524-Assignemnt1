import argparse
import os
import shutil
from random import random, randint, sample
import sys
import pygame
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from src.deep_q_network import DeepQNetwork
from src.tetris import Tetris
from collections import deque


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--replay_memory_size", type=int, default=30000,
                       help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def train(opt):
    # 初始化 Pygame
    pygame.init()
    status_font = pygame.font.SysFont("Arial", 18, bold=True)
    
    # 设置窗口尺寸（游戏区域+60像素信息区域）
    screen_width = opt.width * opt.block_size
    screen_height = opt.height * opt.block_size + 60
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Tetris DQN Training")

    # 初始化环境和模型
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    model = DeepQNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    # CUDA支持
    if torch.cuda.is_available():
        model.cuda()
        print("Using CUDA acceleration")
    else:
        print("Training on CPU")

    # 初始化经验回放
    replay_memory = deque(maxlen=opt.replay_memory_size)
    state = env.reset()
    if torch.cuda.is_available():
        state = state.cuda()

    # 训练参数
    epoch = 0
    manual_save_triggered = False
    last_score = 0

    # 主训练循环
    while epoch < opt.num_epochs:
        # 实时事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:  # S键保存
                    manual_save_triggered = True
                if event.key == pygame.K_q:  # Q键退出
                    pygame.quit()
                    sys.exit()

        # 游戏逻辑
        next_steps = env.get_next_states()
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * 
                (opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        
        # 动作选择
        random_action = random() <= epsilon
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()

        model.eval()
        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        model.train()

        index = randint(0, len(next_steps)-1) if random_action else torch.argmax(predictions).item()
        next_state = next_states[index]
        action = next_actions[index]

        # 执行动作
        reward, done = env.step(action, render=True)
        last_score = env.score if done else last_score

        # 存储经验
        replay_memory.append([
            state.cpu() if torch.cuda.is_available() else state,
            reward,
            next_state.cpu() if torch.cuda.is_available() else next_state,
            done
        ])

        # 状态转移
        state = env.reset() if done else next_state
        if torch.cuda.is_available() and not done:
            state = state.cuda()

        # === 核心训练逻辑 ===
        if len(replay_memory) >= opt.batch_size:
            epoch += 1  # 确保每次训练都递增epoch
            
            # 采样训练批次
            batch = sample(replay_memory, opt.batch_size)
            state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
            
            # 转换为张量
            state_batch = torch.stack(state_batch)
            reward_batch = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1)
            next_state_batch = torch.stack(next_state_batch)
            
            # CUDA支持
            if torch.cuda.is_available():
                state_batch = state_batch.cuda()
                reward_batch = reward_batch.cuda()
                next_state_batch = next_state_batch.cuda()

            # 训练步骤
            model.train()
            q_values = model(state_batch)
            
            with torch.no_grad():
                next_pred = model(next_state_batch)
            
            y_batch = torch.cat([
                reward if done else reward + opt.gamma * pred
                for reward, done, pred in zip(reward_batch, done_batch, next_pred)
            ]).unsqueeze(1)

            optimizer.zero_grad()
            loss = criterion(q_values, y_batch)
            loss.backward()
            optimizer.step()

        # === 实时UI更新 ===
        screen.fill((0, 0, 0))
        env.render()  # 渲染游戏画面
        
        # 绘制信息面板
        info_surface = pygame.Surface((screen_width, 60))
        info_surface.fill((40, 40, 40))
        
        # 信息内容（调整布局）
        info_lines = [
            f"Epoch: {epoch}/{opt.num_epochs}",
            f"Score: {last_score}",
            f"Buffer: {len(replay_memory)}/{opt.replay_memory_size}",
            f"ε: {epsilon:.3f}  [S]保存  [Q]退出"
        ]
        
        # 渲染文本（调整垂直间距）
        for idx, text in enumerate(info_lines):
            text_surface = status_font.render(text, True, (255, 255, 255))
            info_surface.blit(text_surface, (10, 5 + idx*18))  # 每行间隔18像素
        
        # 定位信息面板（游戏区域底部上方）
        screen.blit(info_surface, (0, opt.height*opt.block_size - 5))
        pygame.display.flip()

        # === 保存逻辑 ===
        if manual_save_triggered or (epoch % opt.save_interval == 0 and epoch > 0):
            try:
                save_name = f"tetris_manual_{epoch}.pt" if manual_save_triggered else f"tetris_auto_{epoch}.pt"
                save_path = os.path.join(opt.saved_path, save_name)
                
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'score': last_score
                }, save_path)
                
                print(f"\n💾 保存成功！类型: {'手动' if manual_save_triggered else '自动'} | 路径: {save_path}")
            except Exception as e:
                print(f"\n❌ 保存失败: {str(e)}")
            finally:
                manual_save_triggered = False

    # 训练结束保存最终模型
    final_save_path = os.path.join(opt.saved_path, "tetris_final.pt")
    torch.save(model.state_dict(), final_save_path)
    print(f"\n🎉 训练完成！最终模型已保存至: {final_save_path}")
    pygame.quit()


if __name__ == "__main__":
    opt = get_args()
    train(opt)