CELL_SIZE = 64 // 2
SNAKE_PADDING = 6 // 2
STATE_SIZE = 20
COLOR_BG = (0, 0, 0)
COLOR_SNAKE = (40, 120, 255)
COLOR_APPLE = (220, 40, 40)

DEBUG_MODE = False

def print_debug(*args, **kwargs):
    if DEBUG_MODE:
        print(*args, **kwargs)