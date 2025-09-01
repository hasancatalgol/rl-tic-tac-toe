import sys
import time
import pygame
from pygame import gfxdraw
from functools import lru_cache 
from dataclasses import dataclass
from pathlib import Path  # <-- add this
import os
# number of rows and number of columns
BOARD_SIZE = 3
# this is the reward of winning a game
REWARD = 10


# ---------- Instrumentation ----------
@dataclass
class MinimaxStats:
    nodes_visited: int = 0      # number of child positions expanded (non-terminals) across the whole search
    cutoffs: int = 0            # number of alpha-beta pruning cutoffs
    leaves: int = 0             # number of terminal (win/lose/draw) evaluations
    max_depth: int = 0          # deepest recursion depth reached (root = 0)


def fmt_ab(a, b):
    def f(x):
        if x == float("inf"):
            return "+inf"
        if x == -float("inf"):
            return "-inf"
        # keep small ints as ints
        return int(x) if abs(x - int(x)) < 1e-9 else f"{x:.2f}"
    return f(a), f(b)


class TicTacToe:
    def __init__(self, board=None):
        self.board = board if board is not None else {i: ' ' for i in range(1, 10)}
        self.player = 'O'
        self.computer = 'X'

    # ----- Console helpers (kept but not used by Pygame) -----
    def print_board(self):
        print(self.board[1] + '|' + self.board[2] + '|' + self.board[3])
        print('-+-+-')
        print(self.board[4] + '|' + self.board[5] + '|' + self.board[6])
        print('-+-+-')
        print(self.board[7] + '|' + self.board[8] + '|' + self.board[9])
        print('\n')

    # ----- Core game logic used by Pygame -----
    def is_cell_free(self, position):
        return self.board[position] == ' '

    def place(self, player, position):
        """Place symbol for player at position if free. Returns True if success."""
        if self.is_cell_free(position):
            self.board[position] = player
            return True
        return False

    def clear(self):
        for k in self.board.keys():
            self.board[k] = ' '

    def is_winning(self, player):
        # diagonals
        if self.board[1] == player and self.board[5] == player and self.board[9] == player:
            return True
        if self.board[3] == player and self.board[5] == player and self.board[7] == player:
            return True
        # rows and columns
        for i in range(BOARD_SIZE):
            if self.board[3 * i + 1] == player and self.board[3 * i + 2] == player and self.board[3 * i + 3] == player:
                return True
            if self.board[i + 1] == player and self.board[i + 4] == player and self.board[i + 7] == player:
                return True
        return False

    def is_draw(self):
        for key in self.board.keys():
            if self.board[key] == ' ':
                return False
        return True

    def free_positions(self):
        return [p for p in self.board.keys() if self.board[p] == ' ']

    # ---------- Instrumented search ----------
    def best_move(self, verbose=True):
        """Compute the computer's best move using minimax with alpha-beta; print metrics."""
        t0 = time.perf_counter()
        stats = MinimaxStats()

        legal = self.free_positions()
        if verbose:
            print("\n=== Computer turn (X) ===")
            print("Board:")
            self.print_board()
            print(f"Remaining action space (legal moves): {legal}")

        best_score = -float('inf')
        best_move = None

        # Root alpha/beta
        alpha, beta = -float('inf'), float('inf')

        per_move = []  # (move, score, nodes_sub, cutoffs_sub)

        for position in sorted(legal):
            # Try the move
            self.board[position] = self.computer

            # Snapshot stats to measure this subtree only
            before_nodes = stats.nodes_visited
            before_cutoffs = stats.cutoffs
            before_leaves = stats.leaves
            before_depth = stats.max_depth

            score = self._minimax(1, alpha, beta, False, stats)  # next layer is minimizer (player)

            # Undo
            self.board[position] = ' '

            # Subtree deltas
            nodes_sub = stats.nodes_visited - before_nodes
            cutoffs_sub = stats.cutoffs - before_cutoffs
            leaves_sub = stats.leaves - before_leaves
            maxd_after = stats.max_depth

            per_move.append((position, score, nodes_sub, cutoffs_sub, leaves_sub))

            # Root-level alpha update and possible prune log (we can't prune siblings at root with beta yet,
            # but we log alpha window tightening)
            old_alpha = alpha
            if score > best_score:
                best_score = score
                best_move = position
            alpha = max(alpha, score)
            if verbose:
                oa, ob = fmt_ab(old_alpha, beta)
                na, nb = fmt_ab(alpha, beta)
                print(f"  Move {position}: score={score:>3} | nodes={nodes_sub:>4} leaves={leaves_sub:>4} cutoffs={cutoffs_sub:>3} | αβ {oa},{ob} -> {na},{nb}")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if verbose:
            # Sort per move by score (desc) then position
            ranked = sorted(per_move, key=lambda t: (-t[1], t[0]))
            print("\nRoot evaluation summary (higher is better for X):")
            for mv, sc, nd, co, lf in ranked:
                print(f"   • move {mv}: score={sc:>3}  nodes={nd:>4}  leaves={lf:>4}  cutoffs={co:>3}")

            print(f"\nBest choice: move {best_move} with score {best_score}")
            print(f"Totals: nodes={stats.nodes_visited}, leaves={stats.leaves}, cutoffs={stats.cutoffs}, "
                  f"max_depth={stats.max_depth}, time={elapsed_ms:.2f} ms")
            print("=== End of evaluation ===\n")

        return best_move

    def _minimax(self, depth, alpha, beta, is_maximizer, stats: MinimaxStats):
        """Returns score from X (computer) perspective. Updates stats."""
        stats.max_depth = max(stats.max_depth, depth)

        # Terminal tests
        if self.is_winning(self.computer):
            stats.leaves += 1
            return REWARD - depth
        if self.is_winning(self.player):
            stats.leaves += 1
            return -REWARD + depth
        if self.is_draw():
            stats.leaves += 1
            return 0

        if is_maximizer:  # computer's turn (maximize)
            best_score = -float('inf')
            for position in self.board.keys():
                if self.board[position] == ' ':
                    stats.nodes_visited += 1
                    self.board[position] = self.computer
                    score = self._minimax(depth + 1, alpha, beta, False, stats)
                    self.board[position] = ' '
                    if score > best_score:
                        best_score = score
                    alpha = max(alpha, score)
                    if alpha >= beta:
                        stats.cutoffs += 1
                        break
            return best_score
        else:  # player's turn (minimize)
            best_score = float('inf')
            for position in self.board.keys():
                if self.board[position] == ' ':
                    stats.nodes_visited += 1
                    self.board[position] = self.player
                    score = self._minimax(depth + 1, alpha, beta, True, stats)
                    self.board[position] = ' '
                    if score < best_score:
                        best_score = score
                    beta = min(beta, score)
                    if alpha >= beta:
                        stats.cutoffs += 1
                        break
            return best_score


# ---------------------- Pygame UI ----------------------
GRID_SIZE = 600
INFO_HEIGHT = 80
WINDOW_W = GRID_SIZE
WINDOW_H = GRID_SIZE + INFO_HEIGHT

LINE_WIDTH = 10        # thicker lines to match icon
MARK_WIDTH = 12        # thickness for X/O strokes
TILE_MARGIN = 14       # inner padding to make square tiles
PADDING = 38           # space inside a cell for marks

# Icon-inspired palette
PURPLE = (123, 47, 247)   # #7B2FF7
PINK   = (241, 7, 163)    # #F107A3
INK    = (15, 15, 15)     # X/O + grid color (almost black)

# UI bar colors (unchanged)
INFO_BG = (230, 230, 230)
INFO_TXT = (20, 20, 20)
WIN_TXT = (0, 150, 0)
LOSE_TXT = (180, 0, 0)
DRAW_TXT = (120, 120, 0)



def pos_to_rowcol(pos):
    idx = pos - 1
    return idx // 3, idx % 3


def rowcol_to_pos(row, col):
    return row * 3 + col + 1


def draw_grid(screen, color):
    cell = GRID_SIZE // BOARD_SIZE
    for i in range(1, BOARD_SIZE):
        # vertical
        pygame.draw.line(screen, color, (i * cell, 0), (i * cell, GRID_SIZE), LINE_WIDTH)
        # horizontal
        pygame.draw.line(screen, color, (0, i * cell), (GRID_SIZE, i * cell), LINE_WIDTH)


def make_horizontal_gradient_surface(w, h, left_color, right_color):
    """Create a horizontal gradient surface from left_color -> right_color."""
    surf = pygame.Surface((w, h))
    for x in range(w):
        t = x / max(1, w - 1)
        col = (
            int(left_color[0] + (right_color[0] - left_color[0]) * t),
            int(left_color[1] + (right_color[1] - left_color[1]) * t),
            int(left_color[2] + (right_color[2] - left_color[2]) * t),
        )
        pygame.draw.line(surf, col, (x, 0), (x, h), 1)
    return surf


def draw_pretty_board(surface):
    """Draw gradient tiles and a bold dark grid matching the icon."""
    cell = GRID_SIZE // BOARD_SIZE
    tile_w = cell - 2 * TILE_MARGIN
    tile_h = cell - 2 * TILE_MARGIN

    # Pre-render a tile once and stash it on the function (simple memoization)
    if not hasattr(draw_pretty_board, "_tile"):
        draw_pretty_board._tile = make_horizontal_gradient_surface(tile_w, tile_h, PURPLE, PINK)

    tile = draw_pretty_board._tile

    # fill background a touch lighter than the grid
    surface.fill((245, 245, 245))

    # place gradient tiles
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            x = c * cell + TILE_MARGIN
            y = r * cell + TILE_MARGIN
            surface.blit(tile, (x, y))

    # bold grid lines
    for i in range(1, BOARD_SIZE):
        pygame.draw.line(surface, INK, (i * cell, 0), (i * cell, GRID_SIZE), LINE_WIDTH)
        pygame.draw.line(surface, INK, (0, i * cell), (GRID_SIZE, i * cell), LINE_WIDTH)

def draw_aa_ring(dst_surf, cx, cy, outer_r, width, color):
    """Draw a crisp anti-aliased ring centered at (cx,cy)."""
    outer_r = int(outer_r)
    inner_r = max(0, outer_r - int(width))
    size = outer_r * 2 + 2  # +2 for AA safety
    ring = pygame.Surface((size, size), pygame.SRCALPHA)

    # solid disk
    gfxdraw.filled_circle(ring, outer_r, outer_r, outer_r, color)
    # punch out the hole
    if inner_r > 0:
        gfxdraw.filled_circle(ring, outer_r, outer_r, inner_r, (0, 0, 0, 0))
        # AA inner edge
        gfxdraw.aacircle(ring, outer_r, outer_r, inner_r, color)

    # AA outer edge
    gfxdraw.aacircle(ring, outer_r, outer_r, outer_r, color)

    # blit centered
    dst_surf.blit(ring, (cx - outer_r, cy - outer_r))

@lru_cache(maxsize=64)
def make_ring_sprite(outer_r: int, width: int, color: tuple, ss: int = 4) -> pygame.Surface:
    """
    Return a crisp ring (donut) sprite with outer radius `outer_r` and stroke `width`.
    Renders at ss× resolution, then downsamples for perfect edges. Cached by args.
    """
    outer_r = max(1, int(outer_r))
    width   = max(1, int(width))

    # Hi-res canvas
    size_lo = outer_r * 2 + 2                 # +2 for AA safety
    size_hi = size_lo * ss
    surf_hi = pygame.Surface((size_hi, size_hi), pygame.SRCALPHA)

    or_hi = outer_r * ss
    w_hi  = width * ss
    ir_hi = max(0, or_hi - w_hi)

    # Fill outer disk
    gfxdraw.filled_circle(surf_hi, or_hi, or_hi, or_hi, color)
    # Punch the hole
    if ir_hi > 0:
        gfxdraw.filled_circle(surf_hi, or_hi, or_hi, ir_hi, (0, 0, 0, 0))
        gfxdraw.aacircle(surf_hi, or_hi, or_hi, ir_hi, color)   # AA inner edge

    gfxdraw.aacircle(surf_hi, or_hi, or_hi, or_hi, color)       # AA outer edge

    # Downsample to target pixels
    surf_lo = pygame.transform.smoothscale(surf_hi, (size_lo, size_lo))
    return surf_lo


def draw_marks(screen, game):
    cell = GRID_SIZE // BOARD_SIZE
    for pos, mark in game.board.items():
        if mark == ' ':
            continue
        r = (pos - 1) // 3
        c = (pos - 1) % 3
        x = c * cell
        y = r * cell
        cx = x + cell // 2
        cy = y + cell // 2
        half = cell // 2 - PADDING  # outer radius

        if mark == 'X':
            # your existing lines are fine, but you can supersample these too if you like
            pygame.draw.line(screen, INK, (cx - half, cy - half), (cx + half, cy + half), MARK_WIDTH)
            pygame.draw.line(screen, INK, (cx - half, cy + half), (cx + half, cy - half), MARK_WIDTH)
        else:
            ring = make_ring_sprite(half, MARK_WIDTH, INK)  # cached sprite
            screen.blit(ring, (cx - ring.get_width() // 2, cy - ring.get_height() // 2))


def cell_from_mouse(mx, my):
    if mx < 0 or my < 0 or mx >= GRID_SIZE or my >= GRID_SIZE:
        return None
    cell = GRID_SIZE // BOARD_SIZE
    col = mx // cell
    row = my // cell
    return rowcol_to_pos(row, col)


def status_text(font, text, color):
    return font.render(text, True, color)


def main():
    pygame.init()
    pygame.display.set_caption("Tic-Tac-Toe - Minimax AI")

    # --- set window icon from assets/icon.png (no pathlib) ---
    icon_path = os.path.join(os.path.dirname(__file__), "assets", "icon.png")
    if not os.path.exists(icon_path):
        # also try relative to current working dir, just in case
        alt = os.path.join("assets", "icon.png")
        if os.path.exists(alt):
            icon_path = alt

    try:
        icon_surf = pygame.image.load(icon_path)  # OK after set_mode()
        # optional: scale to a common size (Windows/SDL likes ~32x32)
        icon_surf = pygame.transform.smoothscale(icon_surf, (32, 32))
        pygame.display.set_icon(icon_surf)
        # tip: you can also set a shorter title to see the icon better
        # pygame.display.set_caption("Tic-Tac-Toe")
    except Exception as e:
        print(f"[icon] Could not load '{icon_path}': {e}")
    # ----------------------------------------------------------

    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 32)
    big_font = pygame.font.SysFont(None, 48)

    # Colors
    BG = (245, 245, 245)
    GRID = (30, 30, 30)
    X_COLOR = (50, 50, 50)
    O_COLOR = (20, 120, 220)
    INFO_BG = (230, 230, 230)
    INFO_TXT = (20, 20, 20)
    WIN_TXT = (0, 150, 0)
    LOSE_TXT = (180, 0, 0)
    DRAW_TXT = (120, 120, 0)

    game = TicTacToe()
    player_turn = False  # Computer starts
    game_over = False
    result_text = ""

    # Let the computer open immediately (will print metrics to console)
    def computer_move():
        nonlocal game_over, result_text
        move = game.best_move(verbose=True)  # <-- prints
        if move is not None:
            game.place(game.computer, move)
        # Check state
        if game.is_winning(game.computer):
            game_over = True
            result_text = "Computer wins!"
        elif game.is_draw():
            game_over = True
            result_text = "Draw!"

    computer_move()
    player_turn = True

    # Small delay between turns for UX
    AI_THINK_MS = 250
    pending_ai = False
    ai_timer_ms = 0

    while True:
        dt = clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)
                if event.key == pygame.K_r:
                    # reset
                    game.clear()
                    player_turn = False
                    game_over = False
                    result_text = ""
                    pending_ai = False
                    ai_timer_ms = 0
                    computer_move()
                    player_turn = True

            if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and not game_over and player_turn and not pending_ai:
                pos = cell_from_mouse(*event.pos)
                if pos is not None and game.place(game.player, pos):
                    # Check after player's move
                    if game.is_winning(game.player):
                        game_over = True
                        result_text = "Player wins!"
                    elif game.is_draw():
                        game_over = True
                        result_text = "Draw!"
                    else:
                        # trigger delayed AI move
                        pending_ai = True
                        ai_timer_ms = 0
                        player_turn = False

        # Handle delayed AI "thinking"
        if pending_ai and not game_over:
            ai_timer_ms += dt
            if ai_timer_ms >= AI_THINK_MS:
                computer_move()  # <-- prints fresh metrics each time
                pending_ai = False
                if not game_over:
                    player_turn = True

        # ----- Drawing -----
        screen.fill(BG)
        # Board area
        board_surface = pygame.Surface((GRID_SIZE, GRID_SIZE))
        draw_pretty_board(board_surface)   # gradient tiles + bold grid
        draw_marks(board_surface, game)    # black X/O
        screen.blit(board_surface, (0, 0))

        # Info area
        pygame.draw.rect(screen, INFO_BG, (0, GRID_SIZE, WINDOW_W, INFO_HEIGHT))
        if game_over:
            color = WIN_TXT if "Player" in result_text else (LOSE_TXT if "Computer" in result_text else DRAW_TXT)
            txt = big_font.render(result_text + "  (Press R to restart)", True, color)
        else:
            turn_msg = "Your turn (O)" if player_turn else ("Computer is thinking…" if pending_ai else "Computer's turn")
            txt = status_text(font, turn_msg + " — Press R to restart, ESC to quit", INFO_TXT)
        screen.blit(txt, (20, GRID_SIZE + (INFO_HEIGHT - txt.get_height()) // 2))

        pygame.display.flip()


if __name__ == '__main__':
    main()
