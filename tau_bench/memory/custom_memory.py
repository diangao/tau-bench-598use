# Copyright Sierra

import time
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import networkx as nx
from abc import ABC, abstractmethod
from sklearn.metrics.pairwise import cosine_similarity
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    print("WARNING: sentence-transformers not available, using dummy embeddings for SemanticMemory")


class BaseMemory(ABC):
    """Base class for all memory modules."""
    
    def __init__(self, name: str):
        self.name = name
        self.retrieval_times: List[float] = []
    
    @abstractmethod
    def store(self, state: str, action: Optional[Dict[str, Any]] = None) -> None:
        """Store a state-action pair in memory."""
        pass
    
    @abstractmethod
    def retrieve(self, current_state: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant experiences from memory."""
        pass
    
    def get_avg_retrieval_time(self) -> float:
        """Get the average retrieval time in seconds."""
        if not self.retrieval_times:
            return 0.0
        return sum(self.retrieval_times) / len(self.retrieval_times)


class GraphMemory(BaseMemory):
    """
    Graph-based memory that stores state-action pairs as nodes and edges in a graph.
    
    States are represented as nodes, and actions as edges connecting states.
    Retrieval is done by finding the most similar states and traversing their outgoing edges.
    """
    
    def __init__(self):
        super().__init__("graph_memory")
        self.graph = nx.DiGraph()
        self.state_counter = 0
        self.state_to_id: Dict[str, int] = {}
    
    def _get_state_id(self, state: str) -> int:
        """Get or create an ID for a state."""
        if state not in self.state_to_id:
            self.state_to_id[state] = self.state_counter
            self.state_counter += 1
        return self.state_to_id[state]
    
    def store(self, state: str, action: Optional[Dict[str, Any]] = None) -> None:
        """
        Store a state-action pair in the graph.
        
        If an action is provided, it's stored as an edge from the current state to the next state.
        """
        state_id = self._get_state_id(state)
        
        if state_id not in self.graph:
            self.graph.add_node(state_id, state=state)
        
        if action and 'next_state' in action:
            next_state = action['next_state']
            next_state_id = self._get_state_id(next_state)
            
            if next_state_id not in self.graph:
                self.graph.add_node(next_state_id, state=next_state)
            
            # Add an edge from current state to next state with the action as the edge property
            self.graph.add_edge(state_id, next_state_id, action=action)
            
            print(f"[GraphMemory] Stored state transition: {state_id} -> {next_state_id}")
    
    def _state_similarity(self, state1: str, state2: str) -> float:
        """
        Compute a basic similarity between two states.
        
        For the TicTacToe environment, we can use a simple similarity metric based on
        the fraction of matching cells in the board representation.
        """
        # Simple token-based similarity for TicTacToe states
        tokens1 = set(state1.split())
        tokens2 = set(state2.split())
        
        # Jaccard similarity
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union
    
    def retrieve(self, current_state: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the k most relevant experiences from memory based on the current state.
        
        For graph memory, we find the most similar states to the current state,
        then traverse their outgoing edges to find potential next actions.
        """
        start_time = time.time()
        
        if not self.graph.nodes:
            self.retrieval_times.append(time.time() - start_time)
            return []
        
        # Find the most similar states to the current state
        similarities = []
        for node_id in self.graph.nodes:
            node_state = self.graph.nodes[node_id]['state']
            similarity = self._state_similarity(current_state, node_state)
            similarities.append((node_id, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Take the top k similar states
        similar_states = similarities[:k]
        
        # Retrieve actions from these states
        retrieved_experiences = []
        for state_id, similarity in similar_states:
            # Get outgoing edges (actions) from this state
            for _, next_state_id, edge_data in self.graph.out_edges(state_id, data=True):
                if 'action' in edge_data:
                    action = edge_data['action']
                    state = self.graph.nodes[state_id]['state']
                    next_state = self.graph.nodes[next_state_id]['state']
                    
                    retrieved_experiences.append({
                        'state': state,
                        'action': action,
                        'next_state': next_state,
                        'similarity': similarity
                    })
        
        # Sort by similarity
        retrieved_experiences.sort(key=lambda x: x['similarity'], reverse=True)
        retrieved_experiences = retrieved_experiences[:k]
        
        end_time = time.time()
        self.retrieval_times.append(end_time - start_time)
        
        print(f"[GraphMemory] Retrieved {len(retrieved_experiences)} experiences in {end_time - start_time:.4f}s")
        
        return retrieved_experiences


class VectorMemory(BaseMemory):
    """
    Vector-based memory that embeds states and actions using simple vector representations.
    
    States are embedded as vectors, and retrieval finds the closest states using cosine similarity.
    """
    
    def __init__(self):
        super().__init__("vector_memory")
        self.states: List[str] = []
        self.actions: List[Optional[Dict[str, Any]]] = []
        self.state_vectors: List[np.ndarray] = []
    
    def _embed_state(self, state: str) -> np.ndarray:
        """
        Create a simple embedding for a TicTacToe board state.
        
        For TicTacToe, we use a very simple embedding by counting 'X' and 'O'
        positions and representing them as a 9-dimensional vector.
        """
        # Create a 9-dimensional vector (3x3 board)
        # 1 for X, -1 for O, 0 for empty
        vector = np.zeros(9)
        
        try:
            # Parse the board state from the string representation
            # Example format:
            # ```
            #   0 1 2
            # 0 X| |O
            # 1  |X| 
            # 2 O| |X
            # ```
            lines = state.strip().split('\n')
            board_lines = [line for line in lines if line.strip().startswith('0 ') or 
                                                    line.strip().startswith('1 ') or 
                                                    line.strip().startswith('2 ')]
            
            for i, line in enumerate(board_lines):
                cells = line.strip().split(' ')[1:]
                for j, cell in enumerate(cells):
                    cell = cell.replace('|', '')
                    if cell == 'X':
                        vector[i*3 + j] = 1
                    elif cell == 'O':
                        vector[i*3 + j] = -1
        except Exception as e:
            print(f"[VectorMemory] Error embedding state: {e}")
            # Return a zero vector if parsing fails
            return np.zeros(9)
        
        return vector
    
    def store(self, state: str, action: Optional[Dict[str, Any]] = None) -> None:
        """Store a state-action pair in memory."""
        self.states.append(state)
        self.actions.append(action)
        self.state_vectors.append(self._embed_state(state))
        
        print(f"[VectorMemory] Stored state {len(self.states)}")
    
    def retrieve(self, current_state: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the k most similar states using vector similarity.
        """
        start_time = time.time()
        
        if not self.states:
            self.retrieval_times.append(time.time() - start_time)
            return []
        
        # Embed the current state
        current_vector = self._embed_state(current_state)
        
        # Calculate similarities
        similarities = []
        for i, vector in enumerate(self.state_vectors):
            if np.any(vector) and np.any(current_vector):  # Avoid zero vectors
                sim = cosine_similarity([current_vector], [vector])[0][0]
            else:
                sim = 0.0
            similarities.append((i, sim))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Take top k
        top_k = similarities[:k]
        
        # Build results
        results = []
        for idx, sim in top_k:
            results.append({
                'state': self.states[idx],
                'action': self.actions[idx],
                'similarity': sim
            })
        
        end_time = time.time()
        self.retrieval_times.append(end_time - start_time)
        
        print(f"[VectorMemory] Retrieved {len(results)} experiences in {end_time - start_time:.4f}s")
        
        return results


class SemanticMemory(BaseMemory):
    """
    Semantic memory that uses language model embeddings for states and retrieves based on semantic similarity.
    
    Uses sentence-transformers for embedding if available, otherwise falls back to a simple embedding.
    """
    
    def __init__(self):
        super().__init__("semantic_memory")
        self.states: List[str] = []
        self.actions: List[Optional[Dict[str, Any]]] = []
        self.embeddings: List[np.ndarray] = []
        
        # Try to load a sentence transformer model
        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                print("[SemanticMemory] Loaded sentence-transformers model")
            except Exception as e:
                print(f"[SemanticMemory] Failed to load sentence-transformers model: {e}")
                self.model = None
        else:
            self.model = None
    
    def _embed_text(self, text: str) -> np.ndarray:
        """
        Create a semantic embedding for a text description of the state.
        
        If sentence-transformers is available, use it to create the embedding.
        Otherwise, fall back to a simple token-based embedding.
        """
        if self.model:
            # Use sentence-transformers to create embeddings
            try:
                return self.model.encode([text])[0]
            except Exception as e:
                print(f"[SemanticMemory] Error using sentence-transformers: {e}")
                # Fall back to simple embedding
        
        # Simple fallback embedding based on token counts
        # Just counts the occurrences of each character as a simple vector
        unique_chars = set(text)
        vector = np.zeros(256)  # Use ASCII range
        
        for char in text:
            code = ord(char) % 256
            vector[code] += 1
        
        return vector / (np.sum(vector) + 1e-10)  # Normalize
    
    def _generate_state_description(self, state: str) -> str:
        """
        Generate a textual description of the board state.
        
        This function converts the board representation into a natural language description
        that can be embedded using a language model.
        """
        # Try to parse the board state from its string representation
        try:
            board = []
            lines = state.strip().split('\n')
            
            # Filter out non-board lines
            board_lines = [line for line in lines if line.strip().startswith('0 ') or 
                                                    line.strip().startswith('1 ') or 
                                                    line.strip().startswith('2 ')]
            
            for i, line in enumerate(board_lines):
                row = []
                cells = line.strip().split(' ')[1:]
                for cell in cells:
                    cell = cell.replace('|', '')
                    row.append(cell.strip())
                board.append(row)
            
            # Generate description
            description = "Tic-tac-toe board state: "
            
            # Count X and O
            x_count = sum(row.count('X') for row in board)
            o_count = sum(row.count('O') for row in board)
            
            description += f"There are {x_count} X's and {o_count} O's on the board. "
            
            # Describe board positions
            for i in range(3):
                for j in range(3):
                    if i < len(board) and j < len(board[i]) and board[i][j] in ['X', 'O']:
                        description += f"{board[i][j]} is at position ({i},{j}). "
            
            # Check for potential winning conditions
            # (This is a simplified version)
            
            return description
        
        except Exception as e:
            print(f"[SemanticMemory] Error parsing board: {e}")
            # If parsing fails, return the original state
            return f"TicTacToe board state: {state}"
    
    def store(self, state: str, action: Optional[Dict[str, Any]] = None) -> None:
        """Store a state-action pair in memory with semantic embedding."""
        description = self._generate_state_description(state)
        embedding = self._embed_text(description)
        
        self.states.append(state)
        self.actions.append(action)
        self.embeddings.append(embedding)
        
        print(f"[SemanticMemory] Stored state {len(self.states)}: {description[:50]}...")
    
    def retrieve(self, current_state: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve semantically similar states based on their natural language descriptions.
        """
        start_time = time.time()
        
        if not self.states:
            self.retrieval_times.append(time.time() - start_time)
            return []
        
        # Generate description and embed the current state
        current_desc = self._generate_state_description(current_state)
        current_embedding = self._embed_text(current_desc)
        
        # Calculate similarities
        similarities = []
        for i, embedding in enumerate(self.embeddings):
            if embedding.shape == current_embedding.shape:
                sim = cosine_similarity([current_embedding], [embedding])[0][0]
            else:
                print(f"[SemanticMemory] Embedding shape mismatch: {embedding.shape} vs {current_embedding.shape}")
                sim = 0.0
            similarities.append((i, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top k
        top_k = similarities[:k]
        
        # Build results
        results = []
        for idx, sim in top_k:
            results.append({
                'state': self.states[idx],
                'action': self.actions[idx],
                'similarity': sim,
                'description': self._generate_state_description(self.states[idx])
            })
        
        end_time = time.time()
        self.retrieval_times.append(end_time - start_time)
        
        print(f"[SemanticMemory] Retrieved {len(results)} experiences in {end_time - start_time:.4f}s")
        for i, r in enumerate(results):
            print(f"  {i+1}. Similarity: {r['similarity']:.4f}, Desc: {r['description'][:50]}...")
        
        return results 