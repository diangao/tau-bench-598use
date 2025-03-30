# Copyright Sierra

import json
import time
from typing import List, Dict, Any, Optional, Tuple, Union

from tau_bench.agents.base import Agent
from tau_bench.agents.chat_react_agent import ChatReActAgent
from tau_bench.envs.base import Env
from tau_bench.memory.custom_memory import GraphMemory, VectorMemory, SemanticMemory
from tau_bench.types import Action, SolveResult, RESPOND_ACTION_NAME


class MemoryAgent(ChatReActAgent):
    """
    Base class for agents with memory capabilities.
    
    This agent extends the standard ReAct agent with a memory component
    that can store and retrieve information about past states and actions.
    """
    
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        memory_module: Any = None,  # BaseMemory instance
        use_reasoning: bool = True,
        temperature: float = 0.0,
        max_memory_items: int = 5,
    ) -> None:
        """Initialize the memory agent with the standard parameters plus a memory module."""
        super().__init__(
            tools_info=tools_info,
            wiki=wiki,
            model=model,
            provider=provider,
            use_reasoning=use_reasoning,
            temperature=temperature,
        )
        
        self.memory = memory_module
        self.max_memory_items = max_memory_items
        self.memory_prompt_template = """
        # Previous relevant experiences
        
        Here are some relevant past game states and actions that might help you:
        
        {memory_items}
        
        Remember to use these past experiences when making your decision, but adapt to the current board state.
        """
    
    def _format_memory_items(self, items: List[Dict[str, Any]]) -> str:
        """Format the memory items for inclusion in the prompt."""
        formatted = ""
        
        for i, item in enumerate(items):
            state = item.get('state', '')
            action = item.get('action', {})
            similarity = item.get('similarity', 0.0)
            
            # Clean up state representation
            state_lines = state.strip().split('\n')
            
            formatted += f"## Similar situation {i+1} (similarity: {similarity:.2f})\n\n"
            formatted += "Board state:\n"
            formatted += '\n'.join(state_lines) + "\n\n"
            
            if action:
                if isinstance(action, dict):
                    if 'name' in action and action['name'] == 'make_move':
                        if 'arguments' in action and isinstance(action['arguments'], dict):
                            args = action['arguments']
                            row = args.get('row')
                            col = args.get('col')
                            formatted += f"Action taken: make_move at position ({row}, {col})\n\n"
                        elif 'kwargs' in action and isinstance(action['kwargs'], dict):
                            args = action['kwargs']
                            row = args.get('row')
                            col = args.get('col')
                            formatted += f"Action taken: make_move at position ({row}, {col})\n\n"
            
            # Add a separator
            formatted += "---\n\n"
        
        return formatted
    
    def generate_next_step(
        self, messages: List[Dict[str, Any]], current_state: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Action, float]:
        """
        Generate the next step by augmenting the ReAct prompt with memory.
        
        Args:
            messages: The conversation history
            current_state: The current state (board) for retrieval
        
        Returns:
            Tuple of (message, action, cost)
        """
        # If we have memory and a current state, retrieve relevant experiences
        memory_prompt = ""
        if self.memory and current_state:
            relevant_items = self.memory.retrieve(current_state, k=self.max_memory_items)
            if relevant_items:
                memory_prompt = self.memory_prompt_template.format(
                    memory_items=self._format_memory_items(relevant_items)
                )
        
        # If we have memory items to add, create a new system message with the memory
        if memory_prompt:
            # Modify the system message to include memory
            new_messages = messages.copy()
            
            # Find the system message
            for i, msg in enumerate(new_messages):
                if msg["role"] == "system":
                    # Augment the system message with memory
                    new_messages[i]["content"] = msg["content"] + "\n\n" + memory_prompt
                    break
            
            # Generate with augmented system message
            start_time = time.time()
            message, action, cost = super().generate_next_step(new_messages)
            generation_time = time.time() - start_time
            
            # Log the retrieval and generation times
            print(f"Memory retrieval + generation took {generation_time:.4f}s")
            
            return message, action, cost
        
        # If no memory to add, just use the parent method
        return super().generate_next_step(messages)
    
    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        """
        Solve the task using memory-augmented reasoning.
        
        This method extends the standard solve method to store states and actions in memory.
        """
        response = env.reset(task_index=task_index)
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": response.observation},
        ]
        total_cost = 0.0
        info = {}
        
        # Initialize memory stats
        memory_stats = {
            "retrieval_times": [],
            "storage_count": 0,
        }
        
        current_state = response.observation
        
        for step in range(max_num_steps):
            # Generate next step with current state for memory retrieval
            message, action, cost = self.generate_next_step(messages, current_state)
            
            # Execute the action
            response = env.step(action)
            obs = response.observation
            reward = response.reward
            info = {**info, **response.info.model_dump()}
            
            # Store the current state and action in memory
            if self.memory:
                # Store state-action pair
                self.memory.store(
                    current_state,
                    {
                        "name": action.name,
                        "kwargs": action.kwargs,
                        "next_state": obs,  # Store the observation as the next state
                    }
                )
                memory_stats["storage_count"] += 1
                
                # Update memory stats
                if hasattr(self.memory, "retrieval_times") and self.memory.retrieval_times:
                    memory_stats["retrieval_times"].extend(self.memory.retrieval_times)
            
            # Update current state for next iteration
            current_state = obs
            
            if action.name != RESPOND_ACTION_NAME:
                obs = f"API output: {obs}"
            
            messages.extend(
                [
                    message,
                    {"role": "user", "content": obs},
                ]
            )
            total_cost += cost
            
            if response.done:
                break
        
        # Add memory stats to info
        info["memory_stats"] = memory_stats
        
        return SolveResult(
            messages=messages,
            reward=reward,
            info=info,
            total_cost=total_cost,
        )


class GraphMemoryAgent(MemoryAgent):
    """Agent that uses graph-based memory for state-action reasoning."""
    
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        use_reasoning: bool = True,
        temperature: float = 0.0,
        max_memory_items: int = 5,
    ) -> None:
        memory = GraphMemory()
        super().__init__(
            tools_info=tools_info,
            wiki=wiki,
            model=model,
            provider=provider,
            memory_module=memory,
            use_reasoning=use_reasoning,
            temperature=temperature,
            max_memory_items=max_memory_items,
        )


class VectorMemoryAgent(MemoryAgent):
    """Agent that uses vector-based memory for state-action reasoning."""
    
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        use_reasoning: bool = True,
        temperature: float = 0.0,
        max_memory_items: int = 5,
    ) -> None:
        memory = VectorMemory()
        super().__init__(
            tools_info=tools_info,
            wiki=wiki,
            model=model,
            provider=provider,
            memory_module=memory,
            use_reasoning=use_reasoning,
            temperature=temperature,
            max_memory_items=max_memory_items,
        )


class SemanticMemoryAgent(MemoryAgent):
    """Agent that uses semantic memory for natural language understanding of states."""
    
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        use_reasoning: bool = True,
        temperature: float = 0.0,
        max_memory_items: int = 5,
    ) -> None:
        memory = SemanticMemory()
        super().__init__(
            tools_info=tools_info,
            wiki=wiki,
            model=model,
            provider=provider,
            memory_module=memory,
            use_reasoning=use_reasoning,
            temperature=temperature,
            max_memory_items=max_memory_items,
        ) 