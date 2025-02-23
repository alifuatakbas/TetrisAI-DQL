class Board:
    def __init__(self, width=10, height=20):
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
        self.colors = [[None for _ in range(width)] for _ in range(height)]

    def is_valid_move(self, piece, x, y):
        for i, row in enumerate(piece['shape']):
            for j, cell in enumerate(row):
                if cell:
                    new_x = x + j
                    new_y = y + i
                    if (new_x < 0 or new_x >= self.width or
                            new_y >= self.height or
                            (new_y >= 0 and self.grid[new_y][new_x])):
                        return False
        return True

    def place_piece(self, piece, x, y):
        for i, row in enumerate(piece['shape']):
            for j, cell in enumerate(row):
                if cell and y + i >= 0:
                    self.grid[y + i][x + j] = 1
                    self.colors[y + i][x + j] = piece['color']

    def clear_lines(self):
        lines_cleared = 0
        y = self.height - 1
        while y >= 0:
            if all(self.grid[y]):
                lines_cleared += 1
                # Üstteki satırları aşağı kaydır
                for ny in range(y, 0, -1):
                    self.grid[ny] = self.grid[ny - 1][:]
                    self.colors[ny] = self.colors[ny - 1][:]
                # En üst satırı temizle
                self.grid[0] = [0] * self.width
                self.colors[0] = [None] * self.width
            else:
                y -= 1
        return lines_cleared