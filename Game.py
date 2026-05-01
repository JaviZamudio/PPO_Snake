from Players import Player

import pygame


import random
from collections import deque

from configs import (
    CELL_SIZE,
    COLOR_APPLE,
    COLOR_BG,
    COLOR_SNAKE,
    SNAKE_PADDING,
    STATE_SIZE,
)


class Game:
    EMPTY = 0
    SNAKE = 1
    HEAD = 2
    APPLE = 3

    # 0=up, 1=right, 2=down, 3=left
    DIR_VECTORS = {
        0: (-1, 0),
        1: (0, 1),
        2: (1, 0),
        3: (0, -1),
    }

    def __init__(
        self,
        player: Player,
        grid_size=5,
        initial_apple_pos: tuple[int, int] | None = None,
        prefered_apple_positions: list[tuple[int, int]] | None = None,
    ):
        if not 2 <= grid_size <= STATE_SIZE:
            raise ValueError(f"size must be between 2 and {STATE_SIZE}")

        self.grid_size = grid_size
        self.window_size = self.grid_size * CELL_SIZE

        self.prefered_apple_positions = prefered_apple_positions
        self.initial_apple_pos = initial_apple_pos

        pygame.init()
        pygame.display.set_caption("Grid Snake")
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()

        self.player = player

        self.running = True
        self.state = [
            [self.EMPTY for _ in range(STATE_SIZE)] for _ in range(STATE_SIZE)
        ]

        self.score: int = 0
        self.high_score: int = 0
        self.apple: tuple[int, int] | None = None

        self.reset()

    def _set_cell(self, pos, value):
        r, c = pos
        self.state[r][c] = value

    def _spawn_apple(self, pos: tuple[int, int] | None | list[tuple[int, int]] = None):
        empty_cells = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if self.state[r][c] == self.EMPTY
        ]

        if not empty_cells:
            return

        pos_is_list = isinstance(pos, list)

        # Spawn priority: First try pos, then prefered_apple_positions, then random
        if pos is not None:
            if pos_is_list:
                # intersect list with empty cells, and pick random
                valid_positions = [p for p in pos if p in empty_cells]
                if valid_positions:
                    self.apple = random.choice(valid_positions)
                    self._set_cell(self.apple, self.APPLE)
                    return
            else:
                if pos in empty_cells:
                    self.apple = pos
                    self._set_cell(self.apple, self.APPLE)
                    return
        if self.prefered_apple_positions:
            valid_positions = [
                p for p in self.prefered_apple_positions if p in empty_cells
            ]
            if valid_positions:
                self.apple = random.choice(valid_positions)
                self._set_cell(self.apple, self.APPLE)
                return
            
        self.apple = random.choice(empty_cells)
        self._set_cell(self.apple, self.APPLE)

    def _is_opposite_direction(self, new_dir):
        return (new_dir + 2) % 4 == self.direction

    def _notify_player(self, handler_name):
        if self.player is None:
            return
        handler = getattr(self.player, handler_name, None)
        if callable(handler):
            handler(self.state)

    def _update_game_state(self, requested_direction):
        if requested_direction is None:
            self.running = False
            return

        # Prevent instant 180-degree turns into self.
        if requested_direction in self.DIR_VECTORS and not self._is_opposite_direction(
            requested_direction
        ):
            self.direction = requested_direction

        if self._is_opposite_direction(requested_direction):
            self._notify_player("handle_invalid_move")

        dr, dc = self.DIR_VECTORS[self.direction]
        head_r, head_c = self.snake[-1]
        next_head = (head_r + dr, head_c + dc)
        nr, nc = next_head

        # Wall collision.
        if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
            self._notify_player("handle_crash")
            self._handle_game_over()
            return

        tail = self.snake[0]
        target_value = self.state[nr][nc]

        # Body collision; moving into tail is allowed only if tail will move away.
        if target_value in (self.SNAKE, self.HEAD) and next_head != tail:
            self._notify_player("handle_bite")
            self._handle_game_over()
            return

        ate_apple = next_head == self.apple

        self._set_cell((head_r, head_c), self.SNAKE)
        self.snake.append(next_head)
        self._set_cell(next_head, self.HEAD)

        if not ate_apple:
            old_tail = self.snake.popleft()
            self._set_cell(old_tail, self.EMPTY)
        else:
            self._notify_player("handle_eat")
            self.score += 1
            self.high_score = max(self.high_score, self.score)
            self._spawn_apple()

    def render_game_board(self):
        self.screen.fill(COLOR_BG)

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell_value = self.state[r][c]
                px = c * CELL_SIZE
                py = r * CELL_SIZE

                if cell_value in (self.SNAKE, self.HEAD):
                    rect = pygame.Rect(
                        px + SNAKE_PADDING,
                        py + SNAKE_PADDING,
                        CELL_SIZE - 2 * SNAKE_PADDING,
                        CELL_SIZE - 2 * SNAKE_PADDING,
                    )
                    pygame.draw.rect(self.screen, COLOR_SNAKE, rect)
                elif cell_value == self.APPLE:
                    rect = pygame.Rect(px, py, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(self.screen, COLOR_APPLE, rect)

        # Score display
        font = pygame.font.SysFont(None, 24)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        # High score display
        high_score_text = font.render(
            f"High Score: {self.high_score}", True, (255, 255, 255)
        )
        self.screen.blit(high_score_text, (10, 30))

        pygame.display.flip()

    def run_game_loop(self):
        self.render_game_board()

        while self.running:
            move = self.player.get_move(state=self.state, grid_size=self.grid_size)
            self._update_game_state(move)
            self.render_game_board()
            self.clock.tick(60)

    def _handle_game_over(self):
        print(f"Game over! Score: {self.score}, High Score: {self.high_score}")
        self.running = False

    def reset(self):
        self.running = True
        self.state = [
            [self.EMPTY for _ in range(STATE_SIZE)] for _ in range(STATE_SIZE)
        ]

        center_r = self.grid_size // 2
        center_c = self.grid_size // 2

        head = (center_r, center_c)
        body = (center_r, center_c - 1)
        tail = (center_r, center_c - 2)

        self.snake = deque([tail, body, head])

        self.direction = 1
        self._set_cell(tail, self.SNAKE)
        self._set_cell(body, self.SNAKE)
        self._set_cell(head, self.HEAD)

        self._spawn_apple(
            pos=self.initial_apple_pos
            # pos=[
            #     (1, 1),
            #     (1, self.grid_size - 2),
            #     (self.grid_size - 2, 1),
            #     (self.grid_size - 2, self.grid_size - 2),
            # ]
        )
        self.score = 0
