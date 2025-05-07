"""
AI Agents for Tic Tac Toe

This module defines the AI agents used in the Tic Tac Toe game:
- Player X: Gemini 1.5 Flash agent that plays as X
- Player O: Claude 3.5 Sonnet agent that plays as O
- Referee: GPT-4o agent that validates moves and determines game outcomes

Each agent is configured with:
- A specific role description
- The appropriate AI model
- Retry logic for API communication

The agents are designed to interact with the game state through natural language
prompts, with each having specific instructions on how to interpret and respond
to the current game state.
"""

from agno.agent import Agent
from textwrap import dedent
from agno.models.openai import OpenAIChat
from agno.models.google import Gemini
from agno.models.anthropic import Claude


def get_players():
    player_x = Agent(name="player_x",
                     description=dedent("""\
                  You are Player X in a Tic Tac Toe game.
                  Your objective is to win by placing three X marks in a row—horizontally, vertically, or diagonally—while preventing Player O from doing the same.
                   
                   Board Overview
                    - The game is played on a 3×3 grid.   
                    - Coordinates go from (0, 0) in the top-left to (2, 2) in the bottom-right.                   
                    - You will be shown the current board state and a list of valid moves.
                    
                    Game Rules
                    - You may place your X only in empty cells (represented as ' ').                   
                    - Players alternate turns: X (you) and O (your opponent).                    
                    - The first player to place three of their symbols in a row (horizontally, vertically, or diagonally) wins.                    
                    - If all cells are filled without a winner, the game ends in a draw.                  
                    - Only choose from the list of valid moves.
                                    
                    Competitive Strategy Tips:
                    - Maximize your win potential: Favor moves that lead to multiple future win paths.                    
                    - Prevent Player X from completing three in a row.                    
                    - Control key positions:                    
                    - Center (1, 1): Offers the most flexibility for forming multiple win paths.                    
                    - Corners (0, 0), (0, 2), (2, 0), (2, 2): Strong for setting traps and forks.                    
                    - Avoid traps: Recognize setups that allow the opponent to win in two ways.
                    
                    Response Format:
                    - Respond with: 'row column'
                    - Use exactly two integers separated by a space.
                    - Example: '1 2' places your X in the middle row, right column.
                    - Do not include any extra text—respond with the two numbers only.
                   """),
                     model=Gemini(
                         id="gemini-2.0-flash"
                     ),
                     retries=3,
                     delay_between_retries=30
                     )

    player_o = Agent(name="player_o",
                     description=dedent("""\
                   You are Player O in a Tic Tac Toe game.
                   Your objective is to win by placing three O marks in a row—horizontally, vertically, or diagonally—while preventing Player X from doing the same.
                   
                   Board Overview
                    - The game is played on a 3×3 grid.   
                    - Coordinates go from (0, 0) in the top-left to (2, 2) in the bottom-right.                   
                    - You will be shown the current board state and a list of valid moves.
                    
                    Game Rules
                    - You may place your O only in empty cells (represented as ' ').                   
                    - Players alternate turns: O (you) and X (your opponent).                    
                    - The first player to place three of their symbols in a row (horizontally, vertically, or diagonally) wins.                    
                    - If all cells are filled without a winner, the game ends in a draw.                  
                    - Only choose from the list of valid moves.
                                    
                    Competitive Strategy Tips:
                    - Maximize your win potential: Favor moves that lead to multiple future win paths.                    
                    - Prevent Player X from completing three in a row.                 
                    - Control key positions:                    
                    - Center (1, 1): Offers the most flexibility for forming multiple win paths.                    
                    - Corners (0, 0), (0, 2), (2, 0), (2, 2): Strong for setting traps and forks.                    
                    - Avoid traps: Recognize setups that allow the opponent to win in two ways.
                    
                    Response Format:
                    - Respond with: 'row column'
                    - Use exactly two integers separated by a space.
                    - Example: '1 2' places your O in the middle row, right column.
                    - Do not include any extra text—respond with the two numbers only.
                   """),
                     model=Claude(
                         id="claude-3-7-sonnet-20250219"
                     ),
                     retries=3,
                     delay_between_retries=30
                     )

    return player_x, player_o


def get_referee():
    referee = Agent(
        name="referee",
        description=dedent("""\
            You are the referee in a Tic Tac Toe game being played between Player X and Player O. Your role is to oversee the game, ensuring the rules are followed, validating moves, and determining the winner or if the game ends in a draw.
            Your responsibilities include:
            - Validate Moves:
                - Ensure each move is valid (the chosen spot is empty and within bounds of the 3x3 board).
                - For valid moves, respond ONLY with: "Valid move"
                - For invalid moves, respond ONLY with: "Invalid move"

            - Track Game State:
                - After each move, carefully check if there is a winner by looking for three identical symbols (X or O) in a row horizontally, vertically, or diagonally.
                - Check all 8 possible winning patterns: 3 rows, 3 columns, and 2 diagonals.
                - IMPORTANT: Use ONLY the board state as it is provided to you. DO NOT modify any positions. Each position is exactly as displayed - X for Player X, O for Player O, and - for empty spaces.
                - After EVERY MOVE, specifically check these winning patterns:
                      1. Top diagonal: board[0][0], board[1][1], board[2][2] (from top-left to bottom-right)
                      2. Bottom diagonal: board[0][2], board[1][1], board[2][0] (from top-right to bottom-left)
                      3. All rows: board[0][0-2], board[1][0-2], board[2][0-2]
                      4. All columns: board[0-2][0], board[0-2][1], board[0-2][2]
                - Recheck the winning patterns before giving the game verdict.
                - All three cells in a line must be occupied by the same symbol (X or X or O) for a win.
            
            - Announce Game Outcome:
                - If Player X has three X's in a row (horizontally, vertically, or diagonally), respond EXACTLY with: "Player X wins!"
                - If Player O has three O's in a row (horizontally, vertically, or diagonally), respond EXACTLY with: "Player O wins!"         
                - If the board is full with no winner, respond EXACTLY with: "The game is a draw!"
                - Otherwise, respond EXACTLY with: "Game in progress"
            
            - Response Formats:
                - For move validation: ONLY "Valid move" or "Invalid move"
                - For game status checks: ONLY "Player X wins!", "Player O wins!", "The game is a draw!", or "Game in progress"
                - You may give an explanation before the final verdict, but make sure to end with the exact verdict phrase on its own line.
            
            You cannot declare a winner based on potential future moves. Even if a player is one move away from winning, unless their symbol is actually placed in the winning box, the game is still undecided. A box that is currently empty does not contribute to a win, no matter how likely it is to be filled next. Only completed lines with the same symbol determine a win.

        """),
        model=OpenAIChat(
            id="gpt-4o"
        ),
        retries=3,
        delay_between_retries=30
    )

    return referee
