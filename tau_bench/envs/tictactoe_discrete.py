# Copyright Sierra

import numpy as np
import json
import random
from typing import Optional, List, Dict, Any, Tuple, Union

from tau_bench.envs.base import Env
from tau_bench.envs.user import UserStrategy
from tau_bench.types import Action, Task, EnvInfo, EnvResetResponse, EnvResponse, RewardResult, RewardOutputInfo


class TicTacToeEnv(Env):
    """
    A 3x3 Tic-Tac-Toe environment for τ-Bench.
    
    The environment represents the board as a 3x3 grid. Players take turns
    placing 'X' or 'O' on the board. The first player to get 3 in a row
    (horizontally, vertically, or diagonally) wins.
    """
    
    def __init__(
        self,
        user_strategy: Union[str, UserStrategy] = UserStrategy.LLM,
        user_model: str = "gpt-4o",
        user_provider: Optional[str] = None,
        task_split: str = "test",
        task_index: Optional[int] = None,
    ):
        # Generate tasks for Tic-Tac-Toe
        tasks = self._generate_tasks()
        
        # Define tools for moving in the game
        tools = [TicTacToeMoveTool]
        
        # Rules and wiki for the game
        wiki = """
        # Tic-Tac-Toe Game
        
        This is a standard 3x3 Tic-Tac-Toe game. The rules are:
        1. The game is played on a 3x3 grid.
        2. Players take turns placing 'X' or 'O' on the board.
        3. The first player is 'X', and the second player is 'O'.
        4. The first player to get 3 in a row (horizontally, vertically, or diagonally) wins.
        5. If the grid is full and no one has won, the game is a draw.
        
        You are playing as 'X', and your goal is to win the game.
        """
        
        rules = [
            "You must place your mark on an empty cell.",
            "Players take turns making moves.",
            "The game ends when someone wins or the board is full."
        ]
        
        # Placeholder for data load function that initializes the game board
        def data_load_func():
            return {"board": [[" " for _ in range(3)] for _ in range(3)], "current_player": "X"}
        
        super().__init__(
            data_load_func=data_load_func,
            tools=tools,
            tasks=tasks,
            wiki=wiki,
            rules=rules,
            user_strategy=user_strategy,
            user_model=user_model,
            user_provider=user_provider,
            task_index=task_index,
        )
        
        self.terminate_tools = []
    
    def reset(self, task_index: Optional[int] = None) -> EnvResetResponse:
        """Reset the environment by initializing a new board."""
        response = super().reset(task_index=task_index)
        # Add a rendering of the initial board state to the observation
        board_render = self._render_board()
        response.observation += f"\nInitial board state:\n{board_render}\nYou are X. Make your move."
        return response
    
    def step(self, action: Action) -> EnvResponse:
        """Process an action (a move in the game) and update the board state."""
        if action.name == "make_move":
            try:
                row = action.kwargs.get("row")
                col = action.kwargs.get("col")
                
                # Validate the move
                if not (0 <= row < 3 and 0 <= col < 3):
                    return EnvResponse(
                        observation="Invalid move: Position is outside the board.",
                        reward=0,
                        done=False,
                        info=EnvInfo(task=self.task, source=action.name)
                    )
                
                if self.data["board"][row][col] != " ":
                    return EnvResponse(
                        observation="Invalid move: Cell is already occupied.",
                        reward=0,
                        done=False,
                        info=EnvInfo(task=self.task, source=action.name)
                    )
                
                # Make the move
                self.data["board"][row][col] = self.data["current_player"]
                
                # Check if the game is over
                winner = self._check_winner()
                is_draw = self._is_board_full()
                done = winner is not None or is_draw
                
                # Prepare the observation
                board_render = self._render_board()
                
                if winner:
                    observation = f"Move made at row {row}, col {col}.\n{board_render}\nPlayer {winner} wins!"
                    reward = 1.0 if winner == "X" else 0.0  # Reward is 1 if the agent (X) wins
                elif is_draw:
                    observation = f"Move made at row {row}, col {col}.\n{board_render}\nThe game is a draw."
                    reward = 0.0
                else:
                    # Switch player
                    self.data["current_player"] = "O" if self.data["current_player"] == "X" else "X"
                    
                    # If it's O's turn, make a random move for O
                    if self.data["current_player"] == "O" and not done:
                        o_row, o_col = self._make_random_move()
                        self.data["board"][o_row][o_col] = "O"
                        board_render = self._render_board()
                        
                        # Check if O's move resulted in a win or draw
                        winner = self._check_winner()
                        is_draw = self._is_board_full()
                        done = winner is not None or is_draw
                        
                        if winner:
                            observation = f"Move made at row {row}, col {col}.\nO moves at row {o_row}, col {o_col}.\n{board_render}\nPlayer {winner} wins!"
                            reward = 1.0 if winner == "X" else 0.0
                        elif is_draw:
                            observation = f"Move made at row {row}, col {col}.\nO moves at row {o_row}, col {o_col}.\n{board_render}\nThe game is a draw."
                            reward = 0.0
                        else:
                            self.data["current_player"] = "X"  # Switch back to X
                            observation = f"Move made at row {row}, col {col}.\nO moves at row {o_row}, col {o_col}.\n{board_render}\nIt's your turn (X)."
                            reward = 0.0
                    else:
                        observation = f"Move made at row {row}, col {col}.\n{board_render}\nIt's {self.data['current_player']}'s turn."
                        reward = 0.0
                
                # Record the action
                self.actions.append(action)
                
                return EnvResponse(
                    observation=observation,
                    reward=reward,
                    done=done,
                    info=EnvInfo(task=self.task, source=action.name)
                )
            
            except Exception as e:
                return EnvResponse(
                    observation=f"Error: {str(e)}",
                    reward=0,
                    done=False,
                    info=EnvInfo(task=self.task, source=action.name)
                )
        
        # If not a make_move action, use the parent implementation
        return super().step(action)
    
    def _generate_tasks(self) -> List[Task]:
        """Generate tasks for the Tic-Tac-Toe environment."""
        tasks = []
        
        # Creating 10 initial tasks for testing
        for i in range(1000):  # Will generate 1000 tasks for the experiment
            # Each task is simply to play the game
            task = Task(
                user_id=f"user_{i}",
                actions=[],  # No predetermined actions
                instruction="Play Tic-Tac-Toe. You are X, and the computer is O. Your goal is to win by getting 3 Xs in a row, column, or diagonal. Make valid moves by specifying the row (0-2) and column (0-2) of your move.",
                outputs=[]  # No specific outputs required
            )
            tasks.append(task)
        
        return tasks
    
    def _render_board(self) -> str:
        """Render the current board state as a string."""
        board = self.data["board"]
        result = "```\n"
        result += "  0 1 2\n"
        for i, row in enumerate(board):
            result += f"{i} "
            for cell in row:
                result += f"{cell}|"
            result = result[:-1]  # Remove the last '|'
            result += "\n"
        result += "```"
        return result
    
    def _check_winner(self) -> Optional[str]:
        """Check if there's a winner and return the winning player ('X' or 'O')."""
        board = self.data["board"]
        
        # Check rows
        for row in board:
            if row[0] == row[1] == row[2] != " ":
                return row[0]
        
        # Check columns
        for col in range(3):
            if board[0][col] == board[1][col] == board[2][col] != " ":
                return board[0][col]
        
        # Check diagonals
        if board[0][0] == board[1][1] == board[2][2] != " ":
            return board[0][0]
        if board[0][2] == board[1][1] == board[2][0] != " ":
            return board[0][2]
        
        return None
    
    def _is_board_full(self) -> bool:
        """Check if the board is full (draw condition)."""
        board = self.data["board"]
        for row in board:
            for cell in row:
                if cell == " ":
                    return False
        return True
    
    def _make_random_move(self) -> Tuple[int, int]:
        """Make a random move for player O."""
        board = self.data["board"]
        empty_cells = []
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    empty_cells.append((i, j))
        
        return random.choice(empty_cells) if empty_cells else (0, 0)
    
    def calculate_reward(self) -> RewardResult:
        """Calculate the reward at the end of an episode."""
        winner = self._check_winner()
        is_draw = self._is_board_full()
        
        if winner == "X":
            reward = 1.0
        else:
            reward = 0.0
        
        info = RewardOutputInfo(r_outputs=reward, outputs={"win": winner == "X", "draw": is_draw})
        return RewardResult(reward=reward, info=info, actions=self.actions)


class TicTacToeMoveTool:
    """Tool for making a move in the Tic-Tac-Toe game."""
    
    @staticmethod
    def get_info() -> Dict[str, Any]:
        """Get the tool information in OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": "make_move",
                "description": "Make a move on the Tic-Tac-Toe board by specifying the row and column.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "row": {
                            "type": "integer",
                            "description": "The row index (0-2) where you want to place your mark."
                        },
                        "col": {
                            "type": "integer",
                            "description": "The column index (0-2) where you want to place your mark."
                        }
                    },
                    "required": ["row", "col"]
                }
            }
        }
    
    @classmethod
    def invoke(cls, data: Dict[str, Any], row: int, col: int) -> str:
        """This method is not actually used as the logic is in the env.step method."""
        return f"Moving to row {row}, col {col}" 