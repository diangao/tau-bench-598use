#!/usr/bin/env python3
# Copyright Sierra

import os
import json
import argparse
import datetime
from typing import List, Dict, Any

from tau_bench.types import RunConfig
from tau_bench.run import run
from tau_bench.agents import GraphMemoryAgent, VectorMemoryAgent, SemanticMemoryAgent

# Configure directories
RESULTS_DIR = "results/tictactoe_experiment"
LOGS_DIR = f"{RESULTS_DIR}/logs"
SUMMARIES_DIR = f"{RESULTS_DIR}/summaries"


def create_necessary_dirs():
    """Create the necessary directories for the experiment."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(SUMMARIES_DIR, exist_ok=True)
    print(f"Created directories: {RESULTS_DIR}, {LOGS_DIR}, {SUMMARIES_DIR}")


def run_experiment(
    model: str = "gpt-3.5-turbo",
    model_provider: str = "openai",
    temperature: float = 0.0,
    num_episodes: int = 100,  # Default to 100 episodes for faster testing
    agent_types: List[str] = ["graph", "vector", "semantic"],
    seed: int = 42,
):
    """
    Run the Tic-Tac-Toe experiment with the specified settings.
    
    Args:
        model: The LLM to use for the agent
        model_provider: The provider of the LLM
        temperature: The temperature setting for the LLM
        num_episodes: Number of episodes to run per agent
        agent_types: List of agent types to evaluate
        seed: Random seed for reproducibility
    """
    print(f"Starting Tic-Tac-Toe experiment with model {model} from {model_provider}")
    
    # Create timestamp for this experiment run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"tictactoe_experiment_{timestamp}"
    
    # Store results for all agents
    all_results = {}
    
    # Run each agent type
    for agent_type in agent_types:
        print(f"\n=== Running experiment with {agent_type} memory agent ===\n")
        
        agent_strategy = "react"  
        
        # Build config for this agent
        config = RunConfig(
            model=model,
            model_provider=model_provider,
            user_model=model,  # Use the same model for user as agent
            user_model_provider=model_provider,  # Use the same provider for user as agent
            env="tictactoe_discrete",  # Our new environment
            agent_strategy=agent_strategy,  # ReAct
            temperature=temperature,
            task_split="test",  # Using test split
            num_trials=1,  # Just one trial per episode
            start_index=0,
            end_index=num_episodes-1,  # Adjust for 0-indexing
            log_dir=f"{LOGS_DIR}/{agent_type}_{experiment_id}",
            seed=seed,
            user_strategy="llm",  # Using LLM for user responses
        )
        
        # Override agent_factory in run.py by monkey patching
        from tau_bench import run as tau_run
        original_agent_factory = tau_run.agent_factory
        
        # Define our custom agent factory to inject the memory agents
        def custom_agent_factory(tools_info, wiki, config):
            if agent_type == "graph":
                return GraphMemoryAgent(
                    tools_info=tools_info,
                    wiki=wiki,
                    model=config.model,
                    provider=config.model_provider,
                    use_reasoning=True,  # 改回使用ReAct
                    temperature=config.temperature,
                )
            elif agent_type == "vector":
                return VectorMemoryAgent(
                    tools_info=tools_info,
                    wiki=wiki,
                    model=config.model,
                    provider=config.model_provider,
                    use_reasoning=True,  # 改回使用ReAct
                    temperature=config.temperature,
                )
            elif agent_type == "semantic":
                return SemanticMemoryAgent(
                    tools_info=tools_info,
                    wiki=wiki,
                    model=config.model,
                    provider=config.model_provider,
                    use_reasoning=True,  # 改回使用ReAct
                    temperature=config.temperature,
                )
            else:
                return original_agent_factory(tools_info, wiki, config)
        
        # Apply our monkey patch
        tau_run.agent_factory = custom_agent_factory
        
        try:
            # Run the experiment with this agent
            results = run(config)
            
            # Restore original agent factory
            tau_run.agent_factory = original_agent_factory
            
            # Store results
            all_results[agent_type] = [r.model_dump() for r in results]
            
            # Calculate statistics
            win_rate = sum(1 for r in results if r.reward > 0.5) / len(results)
            print(f"{agent_type} agent win rate: {win_rate:.2f}")
            
            # Extract memory statistics
            retrieval_times = []
            storage_counts = []
            
            for r in results:
                if "memory_stats" in r.info:
                    stats = r.info["memory_stats"]
                    if "retrieval_times" in stats:
                        retrieval_times.extend(stats["retrieval_times"])
                    if "storage_count" in stats:
                        storage_counts.append(stats["storage_count"])
            
            if retrieval_times:
                avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
                print(f"Average retrieval time: {avg_retrieval_time:.4f}s")
            
            if storage_counts:
                avg_storage_count = sum(storage_counts) / len(storage_counts)
                print(f"Average storage count: {avg_storage_count:.2f}")
            
        except Exception as e:
            print(f"Error running {agent_type} agent: {e}")
    
    # Save summary of all results
    summary_path = f"{SUMMARIES_DIR}/{experiment_id}_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "experiment_id": experiment_id,
            "model": model,
            "model_provider": model_provider,
            "temperature": temperature,
            "num_episodes": num_episodes,
            "agent_types": agent_types,
            "results": all_results
        }, f, indent=2)
    
    print(f"\nExperiment completed. Summary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tic-Tac-Toe experiment with memory agents")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Model to use for the agent")
    parser.add_argument("--model-provider", type=str, default="openai", help="Provider of the model")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for the model")
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of episodes to run per agent")
    parser.add_argument("--agents", type=str, default="graph,vector,semantic", help="Comma-separated list of agent types to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Create necessary directories
    create_necessary_dirs()
    
    # Parse agent types
    agent_types = args.agents.split(",")
    
    # Run the experiment
    run_experiment(
        model=args.model,
        model_provider=args.model_provider,
        temperature=args.temperature,
        num_episodes=args.num_episodes,
        agent_types=agent_types,
        seed=args.seed,
    ) 