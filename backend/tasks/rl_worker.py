import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from datetime import datetime, timezone
from backend.extensions import db
from backend.models.rl import RLTrainingJob, RLAgentState
from backend.agents.environment import TradingEnv
from backend.agents.agent import QLearningAgent
from backend.tasks.job_guard import monitor_job

@monitor_job("RL Training Worker")
def run_rl_training(app, job_id, filepath, params):
    """Run REAL RL training using actual CSV data and Q-Learning"""
    
    with app.app_context():
        job = db.session.get(RLTrainingJob, job_id)
        if not job:
            print(f"[RL ERROR] Job {job_id} not found")
            return
        
        try:
            print(f"[RL] 🚀 Starting REAL training for job {job_id}")
            job.status = "running"
            job.started_at = datetime.now(timezone.utc)
            db.session.commit()

            # ========== LOAD REAL CSV DATA ==========
            print(f"[RL] 📊 Loading CSV data from {filepath}")
            df = pd.read_csv(filepath)
            
            # Validate CSV has Close column
            if 'Close' not in df.columns:
                raise ValueError("CSV must contain 'Close' column")
            
            # Extract actual price data
            prices = df["Close"].values.tolist()
            print(f"[RL] ✅ Loaded {len(prices)} price points from CSV")
            print(f"[RL]    Price range: ${min(prices):.2f} - ${max(prices):.2f}")

            # ========== INITIALIZE REAL ENVIRONMENT ==========
            env = TradingEnv(
                data=prices,
                initial_balance=params["initial_balance"],
                tx_cost=0.001  # 0.1% transaction cost
            )
            print(f"[RL] 🏗️  Environment initialized with ${params['initial_balance']} balance")

            # ========== INITIALIZE REAL Q-LEARNING AGENT ==========
            agent = QLearningAgent(
                action_size=3,
                learning_rate=params["learning_rate"],
                discount_factor=params["discount_factor"],
                exploration_rate=params["epsilon"],
                exploration_decay=params["epsilon_decay"],
                min_exploration=0.01
            )
            print(f"[RL] 🤖 Agent initialized:")
            print(f"[RL]    Learning rate (α): {params['learning_rate']}")
            print(f"[RL]    Discount factor (γ): {params['discount_factor']}")
            print(f"[RL]    Exploration rate (ε): {params['epsilon']}")

            episodes = params["episodes"]
            portfolio_history = []
            reward_history = []

            # ========== REAL TRAINING LOOP ==========
            print(f"[RL] 🏋️  Starting training for {episodes} episodes...")
            
            for episode in range(1, episodes + 1):
                state = env.reset()
                done = False
                total_reward = 0
                steps = 0

                # Run episode until done
                while not done:
                    # Agent chooses action using epsilon-greedy
                    action = agent.choose_action(state)
                    
                    # Environment executes action
                    next_state, reward, done, info = env.step(action)
                    
                    # Agent learns from experience
                    agent.learn(state, action, reward, next_state, done)
                    
                    state = next_state
                    total_reward += reward
                    steps += 1
                
                # Get final portfolio value
                final_price = prices[-1]
                final_portfolio_value = info['portfolio_value']
                
                portfolio_history.append(final_portfolio_value)
                reward_history.append(total_reward)

                # Decay exploration rate
                agent.decay_epsilon()

                # Progress logging
                if episode % 50 == 0 or episode == episodes:
                    log_msg = f"[RL] Episode {episode}/{episodes}: " \
                             f"Value=${final_portfolio_value:.2f}, " \
                             f"Reward={total_reward:.2f}, " \
                             f"Trades={info['total_trades']}, " \
                             f"ε={agent.epsilon:.3f}, " \
                             f"States={len(agent.q_table)}"
                    print(log_msg)

                # Update job progress
                job.progress = int((episode / episodes) * 100)
                if episode % 10 == 0:  # Commit every 10 episodes
                    db.session.commit()

            print(f"[RL] ✅ Training completed!")
            print(f"[RL]    Total Q-table states: {len(agent.q_table)}")

            # ========== SAVE Q-TABLE TO DATABASE ==========
            print(f"[RL] 💾 Saving Q-table to database...")
            
            # Clear old states for this job
            try:
                deleted = db.session.query(RLAgentState).filter_by(
                    user_id=job.user_id,
                    job_id=job_id
                ).delete()
                db.session.commit()
                print(f"[RL]    Cleared {deleted} old states")
            except Exception as e:
                print(f"[RL WARNING] Could not clear old states: {e}")
                db.session.rollback()
            
            # Save new states in batches
            saved_count = 0
            batch_size = 100
            states_to_add = []
            
            for state_key, q_values in agent.q_table.items():
                try:
                    rl_state = RLAgentState(
                        user_id=job.user_id,
                        job_id=job_id,
                        state_key=str(state_key),
                        q_buy=float(q_values[1]),      # Action 1 = BUY
                        q_hold=float(q_values[0]),     # Action 0 = HOLD
                        q_sell=float(q_values[2]),     # Action 2 = SELL
                        visit_count=1
                    )
                    states_to_add.append(rl_state)
                    saved_count += 1
                    
                    if len(states_to_add) >= batch_size:
                        db.session.bulk_save_objects(states_to_add)
                        db.session.commit()
                        states_to_add = []
                        
                except Exception as e:
                    print(f"[RL ERROR] Failed to save state {state_key}: {e}")
                    db.session.rollback()
                    continue
            
            # Save remaining states
            if states_to_add:
                db.session.bulk_save_objects(states_to_add)
                db.session.commit()
            
            print(f"[RL] ✅ Saved {saved_count} Q-table states to database")

            # ========== SAVE Q-TABLE AS PICKLE ==========
            try:
                models_dir = app.config['MODELS_FOLDER']
                os.makedirs(models_dir, exist_ok=True)
                
                q_table_filename = f'q_table_job_{job_id}.pkl'
                q_table_path = os.path.join(models_dir, q_table_filename)
                
                agent.save(q_table_path)
                job.q_table_path = q_table_path
                print(f"[RL] ✅ Q-table pickle saved to {q_table_path}")
                
            except Exception as e:
                print(f"[RL ERROR] Failed to save pickle: {e}")
                job.error_message = f"Pickle save failed: {str(e)}"

            # ========== SAVE FINAL RESULTS ==========
            final_value = portfolio_history[-1]
            total_reward = reward_history[-1]
            
            job.final_balance = float(final_value)
            job.total_reward = float(sum(reward_history))
            job.portfolio_history = json.dumps(portfolio_history)
            job.reward_history = json.dumps(reward_history)

            # Save results JSON file
            try:
                # Use absolute path to ensure consistency
                results_dir = os.path.join(os.getcwd(), 'results')
                os.makedirs(results_dir, exist_ok=True)
                
                results_data = {
                    "job_id": job_id,
                    "episodes": episodes,
                    "initial_balance": params["initial_balance"],
                    "final_balance": job.final_balance,
                    "total_reward": job.total_reward,
                    "profit_loss": job.final_balance - params["initial_balance"],
                    "return_pct": ((job.final_balance - params["initial_balance"]) / params["initial_balance"] * 100),
                    "portfolio_history": portfolio_history,
                    "reward_history": reward_history,
                    "q_table_states": saved_count,
                    "training_params": params
                }
                
                results_path = os.path.join(results_dir, f"results_{job_id}.json")
                with open(results_path, "w") as f:
                    json.dump(results_data, f, indent=4)
                    
                print(f"[RL] ✅ Results saved to {results_path}")

            except Exception as e:
                print(f"[RL ERROR] Failed to save results: {e}")
            
            # ========== GENERATE PLOT ==========
            try:
                plt.figure(figsize=(12, 6))
                
                plt.subplot(1, 2, 1)
                plt.plot(portfolio_history, color='#00f5ff', linewidth=2)
                plt.axhline(y=params["initial_balance"], color='red', 
                           linestyle='--', label='Initial Balance')
                plt.title(f'Portfolio Value Over Episodes (Job {job_id})')
                plt.xlabel('Episode')
                plt.ylabel('Portfolio Value ($)')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                plt.subplot(1, 2, 2)
                plt.plot(reward_history, color='#ff00ff', linewidth=2)
                plt.title('Cumulative Reward Per Episode')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_path = os.path.join(results_dir, 'trading_plot.png')
                plt.savefig(plot_path, dpi=100, bbox_inches='tight')
                plt.close()
                
                print(f"[RL] ✅ Plot saved to {plot_path}")

            except Exception as e:
                print(f"[RL ERROR] Failed to generate plot: {e}")

            # ========== UPDATE JOB STATUS ==========
            job.status = "completed"
            job.progress = 100
            job.completed_at = datetime.now(timezone.utc)
            db.session.commit()
            
            print(f"[RL] 🎉 Training complete for job {job_id}!")
            print(f"[RL]    Initial: ${params['initial_balance']:.2f}")
            print(f"[RL]    Final: ${job.final_balance:.2f}")
            print(f"[RL]    Profit/Loss: ${job.final_balance - params['initial_balance']:.2f}")
            print(f"[RL]    Return: {((job.final_balance - params['initial_balance']) / params['initial_balance'] * 100):.2f}%")
            
        except Exception as e:
            print(f"[RL ERROR] Training failed: {e}")
            import traceback
            traceback.print_exc()
            
            job.status = "failed"
            job.error_message = str(e)
            db.session.rollback()
            db.session.commit()
