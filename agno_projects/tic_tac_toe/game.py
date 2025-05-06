"""
Tic Tac Toe Game Logic

This module implements the core game logic for a Tic Tac Toe game.
It provides a Board class that manages:
- The game board state (3x3 grid)
- Move validation
- Move application
- Board state retrieval

The Board class is designed to be simple and focused on the game rules,
without handling UI, turn management, or win detection (which are handled
in the main application).
"""

from typing import List, Optional


class Board:
    def __init__(self):
        """Initialize a 3x3 empty board."""
        self.reset_board()

    def reset_board(self) -> None:
        """Reset the board to empty state."""
        self.board = [[' ' for _ in range(3)] for _ in range(3)]

    def is_valid_move(self, row: int, col: int) -> bool:
        """Check if a move is valid (within bounds and cell is empty)."""
        return (0 <= row < 3 and 0 <= col < 3 and
                self.board[row][col] == ' ')

    def apply_move(self, player_symbol: str, row: int, col: int) -> bool:
        """Apply a move if it's valid. Returns True if successful, False otherwise."""
        if not self.is_valid_move(row, col):
            return False
        self.board[row][col] = player_symbol
        return True

    def get_board_state(self) -> List[List[str]]:
        """Return the current board state as a 2D list."""
        return [row[:] for row in self.board]

