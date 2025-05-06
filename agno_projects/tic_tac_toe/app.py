"""
Tic Tac Toe Game with AI Players

This Streamlit application implements a Tic Tac Toe game played entirely by AI agents.
Three different AI models participate:
- Player X: Gemini 1.5 Flash
- Player O: Claude 3.5 Sonnet
- Referee: GPT-4o

The application handles:
- Game board rendering and state management
- Turn-based gameplay between AI agents
- Move validation via the referee
- Win/draw detection
- Game flow and UI presentation
"""

import streamlit as st
from agents import get_players, get_referee
from game import Board
import time


def setup_page_config():
    """Configure the Streamlit page settings"""
    st.set_page_config(
        page_title="Tic Tac Toe",
        page_icon="🎮",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def custom_css():
    """Apply custom CSS styling for the game board"""
    st.markdown("""
    <style>
        /* Custom board grid */
        .board-grid {
            display: grid;
            grid-template-columns: repeat(3, 50px);
            grid-template-rows: repeat(3, 50px);
            grid-gap: 4px;
            background-color: #f0f0f0;
            padding: 6px;
            border-radius: 8px;
            border: 2px solid #ddd;
            margin: 0 auto;
            width: fit-content;
        }

        .board-cell {
            width: 50px;
            height: 50px;
            background: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            font-weight: bold;
            border: 1px solid #ccc;
            border-radius: 4px;
        }

        /* Ensure win message stays visible */
        .win-message {
            padding: 10px;
            background-color: #e8f4f8;
            border-radius: 5px;
            margin: 10px 0;
            font-weight: bold;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)


def format_board_for_agents(board_state):
    """Format the board state in a readable way for AI agents"""
    formatted = "Board:\n"
    formatted += "    0   1   2\n"
    for i, row in enumerate(board_state):
        formatted += f"{i} | " + " | ".join([cell if cell != " " else "-" for cell in row]) + " |\n"
    return formatted


def render_board(board_state):
    """Render the game board using custom HTML/CSS"""
    board_html = '<div class="board-grid">'
    for i, row in enumerate(board_state):
        for j, cell in enumerate(row):
            display = cell if cell != " " else "-"
            board_html += f'<div class="board-cell">{display}</div>'
    board_html += '</div>'

    st.markdown(board_html, unsafe_allow_html=True)


def get_valid_moves(board):
    """Get a list of valid moves in (row,col) format"""
    valid_moves = []
    for r in range(3):
        for c in range(3):
            if board.is_valid_move(r, c):
                valid_moves.append(f"({r},{c})")
    return valid_moves


def initialize_game():
    """Initialize or reset the game state"""
    st.session_state.board = Board()
    st.session_state.current_symbol = 'X'
    st.session_state.game_over = False
    st.session_state.winner_message = None
    st.session_state.player_x, st.session_state.player_o = get_players()
    st.session_state.referee = get_referee()
    st.session_state.current_player = st.session_state.player_x
    st.session_state.game_started = True


def handle_player_move(board, current_symbol, player, valid_moves):
    """Process a player's move and get referee validation"""
    # Format board for AI agent
    formatted_board = format_board_for_agents(board.get_board_state())

    # Prepare player prompt
    player_prompt = (
        f"Current board state:\n{formatted_board}\n"
        f"Valid moves are: {', '.join(valid_moves)}\n"
        f"IMPORTANT: Choose ONLY from the valid moves listed above!\n"
        f"Make your move (row column):"
    )

    # Get move from current player
    player_response = player.run(player_prompt)
    move = player_response.content
    st.markdown(
        f"""<div style="font-size:20px; font-weight:600; margin:10px 0;">
            Player <strong>{current_symbol}</strong> chose: <code>{move}</code>
        </div>""",
        unsafe_allow_html=True
    )

    try:
        row, col = map(int, move.strip().split())
        return row, col
    except (ValueError, IndexError) as e:
        st.error(f"⚠️ Error parsing move: {e}. Expected format: `row col` (e.g., `1 2`)")
        return None, None


def validate_move(board, row, col, current_symbol, referee, valid_moves):
    """Have the referee validate if the move is legal"""
    formatted_board = format_board_for_agents(board.get_board_state())

    referee_prompt = (
        f"Board state:\n{formatted_board}\n"
        f"These are the Valid moves which are available: {', '.join(valid_moves)}\n"
        f"And Player {current_symbol} attempted move: {row} {col}\n"
        f"Is this move valid? The move should be on an empty space (-) and within bounds (0-2)."
    )

    referee_response = referee.run(referee_prompt)
    ref_content = referee_response.content
    st.markdown(
        f"""<div style="font-size:20px; font-weight:600; margin:10px 0;">
        <strong>Referee says:</strong>{ref_content}
        </div>""",
        unsafe_allow_html=True
    )

    return "valid move" in ref_content.lower() and "invalid" not in ref_content.lower()


def check_game_status(board, current_symbol, row, col, referee):
    """Check if the game has ended (win or draw)"""
    updated_board_state = board.get_board_state()
    formatted_board = format_board_for_agents(updated_board_state)

    game_status_prompt = (
        f"Check for a winner or draw:\n"
        f"Board state:\n{formatted_board}\n"
        f"After Player {current_symbol}'s move to position ({row},{col}), "
        f"Check if there is a winner - horizontally, vertically, or diagonally; or if it is a draw (when the board is full)"
    )

    status_response = referee.run(game_status_prompt)
    status_content = status_response.content

    game_over = False
    winner_message = None

    if "Player X wins!" in status_content:
        winner_message = "Referee verdict:🎉 Player X wins! 🎉"
        game_over = True
    elif "Player O wins!" in status_content:
        winner_message = "Referee verdict:🎉 Player O wins! 🎉"
        game_over = True
    elif "The game is a draw!" in status_content:
        winner_message = "Referee verdict:🤝 Game is a draw! 🤝"
        game_over = True

    return game_over, winner_message


def main():
    """Main application function"""
    st.markdown(
        """
        <h1 style=' text-align: center; font-size: 42px; font-weight: 800; margin-top: 10px; margin-bottom: 20px; '>
            🎮 Tic Tac Toe
        </h1>
        """,
        unsafe_allow_html=True
    )

    # Initialize session state variables if needed
    if 'game_started' not in st.session_state:
        st.session_state.game_started = False
        st.session_state.game_over = False
        st.session_state.winner_message = None

    # GAME SETUP: Show start screen or handle game restart
    if not st.session_state.game_started:
        st.info("This game is played between two AI agents, with a third agent acting as the referee:"
                "\n1. Player X – Gemini 1.5 Flash"
                "\n2. Player O – Claude 3.5 Sonnet"
                "\n3. Referee – GPT-4o"
                "\n\nClick 'Start Game' to start the game.👇")
        if st.button("Start Game"):
            initialize_game()
            st.rerun()
        st.stop()

    # GAME IN PROGRESS: Show status if game is ongoing
    if st.session_state.game_started and not st.session_state.game_over:
        st.info("Game in progress.")
        st.info("Note: Board updates may take a few seconds.")

    # GAME BOARD: Render the current board state
    board = st.session_state.board
    board_state = board.get_board_state()
    valid_moves = get_valid_moves(board)
    render_board(board_state)

    # GAME RESULT: Show winner message if available
    if st.session_state.winner_message:
        st.markdown(f"""<div class="win-message">{st.session_state.winner_message}</div>""", unsafe_allow_html=True)

    # GAME OVER: Show restart option if game is complete
    if st.session_state.game_over:
        st.info("Click 'Start New Game' to start a new game.👇")

        if st.button("Start New Game"):
            initialize_game()
            st.rerun()

    # GAME TURN: Process the current player's turn if game is still active
    else:
        st.markdown(
            f"""<div style="font-size:22px; font-weight:bold; text-align:left; margin-top:10px;">
                Player {st.session_state.current_symbol}'s Turn
            </div>""",
            unsafe_allow_html=True
        )

        # Get player's move
        row, col = handle_player_move(
            board,
            st.session_state.current_symbol,
            st.session_state.current_player,
            valid_moves
        )

        if row is not None and col is not None:
            # Validate move with referee
            if validate_move(board, row, col, st.session_state.current_symbol, st.session_state.referee, valid_moves):
                # Apply the move to the board
                board.apply_move(st.session_state.current_symbol, row, col)

                # Check for game end conditions
                game_over, winner_message = check_game_status(
                    board,
                    st.session_state.current_symbol,
                    row,
                    col,
                    st.session_state.referee
                )

                # Update game state if needed
                if game_over:
                    st.session_state.game_over = True
                    st.session_state.winner_message = winner_message
                else:
                    # Switch player for next turn
                    if st.session_state.current_symbol == 'X':
                        st.session_state.current_symbol = 'O'
                        st.session_state.current_player = st.session_state.player_o
                    else:
                        st.session_state.current_symbol = 'X'
                        st.session_state.current_player = st.session_state.player_x

                st.rerun()
            else:
                # Replace standard warning with custom markdown that will disappear on rerun
                st.markdown(
                    """
                    <div style="background-color:#FFF3CD; padding:10px; border-radius:5px; border-left:5px solid #FFD700; margin:10px 0;">
                        ❌ <b>Invalid move according to referee!</b> Player gets another turn.
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                st.rerun()


if __name__ == "__main__":
    setup_page_config()
    custom_css()
    main()
