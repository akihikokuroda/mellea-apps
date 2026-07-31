#!/usr/bin/env python3
"""
Othello (Reversi) game with AI opponent powered by Mellea.

Players take turns placing discs on an 8x8 board. Discs are black (human) or white (AI).
When a player places a disc, any opponent discs in a straight line between the new disc and
an existing disc of the player's color are flipped.
"""

from typing import Literal
from pydantic import BaseModel
from mellea import start_session, generative
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import RejectionSamplingStrategy


class Move(BaseModel):
    row: int
    col: int


@generative
def get_ai_move(board_state: str, valid_moves: list[tuple[int, int]]) -> Move:
    """
    Determine the best move for the AI player (white discs).

    The board is represented as 8 lines of 8 characters:
    - '●' = black (human)
    - '○' = white (AI)
    - '·' = empty

    Return a move as row (0-7) and col (0-7).

    Example board state:
    ········
    ········
    ···●····
    ··●○····
    ···○●···
    ········
    ········
    ········
    """
    ...


class OthelloGame:
    def __init__(self):
        self.board = [[None for _ in range(8)] for _ in range(8)]
        self.board[3][3] = "white"
        self.board[3][4] = "black"
        self.board[4][3] = "black"
        self.board[4][4] = "white"
        self.session = start_session(model_id="granite3.3:8b")

    def print_board(self):
        """Display the board with coordinates."""
        print("\n  0 1 2 3 4 5 6 7")
        for row in range(8):
            print(f"{row} ", end="")
            for col in range(8):
                cell = self.board[row][col]
                if cell == "black":
                    print("● ", end="")
                elif cell == "white":
                    print("○ ", end="")
                else:
                    print("· ", end="")
            print()
        print()

    def get_valid_moves(self, color: str) -> list[tuple[int, int]]:
        """Find all valid moves for a given color."""
        valid = []
        opponent = "white" if color == "black" else "black"

        for row in range(8):
            for col in range(8):
                if self.board[row][col] is not None:
                    continue

                # Check all 8 directions
                for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                    r, c = row + dr, col + dc
                    found_opponent = False

                    # Scan in this direction for opponent discs followed by player disc
                    while 0 <= r < 8 and 0 <= c < 8:
                        if self.board[r][c] == opponent:
                            found_opponent = True
                        elif self.board[r][c] == color and found_opponent:
                            valid.append((row, col))
                            break
                        else:
                            break
                        r += dr
                        c += dc

        return list(set(valid))

    def make_move(self, row: int, col: int, color: str) -> bool:
        """Place a disc and flip opponent discs. Returns True if move was valid."""
        if self.board[row][col] is not None:
            return False

        opponent = "white" if color == "black" else "black"
        flipped_any = False

        # Check all 8 directions
        for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            r, c = row + dr, col + dc
            to_flip = []

            # Collect opponent discs in this direction
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == opponent:
                to_flip.append((r, c))
                r += dr
                c += dc

            # If we hit a player disc, flip all collected opponent discs
            if to_flip and 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == color:
                for fr, fc in to_flip:
                    self.board[fr][fc] = color
                flipped_any = True

        if flipped_any:
            self.board[row][col] = color
            return True

        return False

    def get_score(self) -> tuple[int, int]:
        """Return (black_count, white_count)."""
        black = sum(1 for row in self.board for cell in row if cell == "black")
        white = sum(1 for row in self.board for cell in row if cell == "white")
        return black, white

    def board_to_string(self) -> str:
        """Convert board to string representation for AI."""
        lines = []
        for row in range(8):
            line = ""
            for col in range(8):
                if self.board[row][col] == "black":
                    line += "●"
                elif self.board[row][col] == "white":
                    line += "○"
                else:
                    line += "·"
            lines.append(line)
        return "\n".join(lines)

    def play(self):
        """Main game loop."""
        print("Welcome to Othello! You are black (●), AI is white (○)")
        print("Enter moves as 'row col' (e.g., '2 3')")
        print()

        while True:
            self.print_board()
            black_moves = self.get_valid_moves("black")
            white_moves = self.get_valid_moves("white")

            if not black_moves and not white_moves:
                break

            # Human turn (black)
            if black_moves:
                print(f"Valid moves: {black_moves}")
                while True:
                    try:
                        move_input = input("Your move (row col): ").strip()
                        row, col = map(int, move_input.split())
                        if self.make_move(row, col, "black"):
                            break
                        else:
                            print("Invalid move! Try again.")
                    except (ValueError, IndexError):
                        print("Invalid input! Enter row col (e.g., 2 3)")
            else:
                print("No valid moves for black. Passing...")

            self.print_board()

            # AI turn (white)
            white_moves = self.get_valid_moves("white")
            if white_moves:
                print("AI is thinking...")
                board_str = self.board_to_string()
                move = get_ai_move(
                    self.session,
                    board_state=board_str,
                    valid_moves=white_moves
                )

                if self.make_move(move.row, move.col, "white"):
                    print(f"AI played at ({move.row}, {move.col})")
                else:
                    print("AI move validation failed, trying first valid move...")
                    self.make_move(white_moves[0][0], white_moves[0][1], "white")
            else:
                print("No valid moves for AI. Passing...")

        # Game over
        black_score, white_score = self.get_score()
        print("\n=== GAME OVER ===")
        print(f"Black: {black_score}")
        print(f"White: {white_score}")

        if black_score > white_score:
            print("You win!")
        elif white_score > black_score:
            print("AI wins!")
        else:
            print("It's a tie!")


if __name__ == "__main__":
    game = OthelloGame()
    game.play()
