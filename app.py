from flask import Flask, request, render_template, jsonify
import numpy as np

app = Flask(__name__)

# Example boards (same as before)
easy_board = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

hard_board = [
    [0, 0, 0, 0, 0, 0, 0, 1, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 6, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 0, 4, 0, 7],
    [3, 0, 0, 4, 0, 7, 0, 0, 6],
    [7, 0, 4, 0, 9, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 5, 9, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [5, 3, 0, 0, 0, 0, 0, 0, 0]
]

expert_board = [
    [0, 0, 0, 0, 0, 7, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 4, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 5],
    [0, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 0, 0, 0, 0],
    [0, 0, 0, 7, 0, 0, 0, 0, 0],
    [0, 4, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 3, 0]
]

def get_possible_values(board, row, col):
    if board[row][col] != 0:
        return set()

    possible_values = set(range(1, 10))
    
    # Remove values from the same row
    possible_values -= set(board[row])
    
    # Remove values from the same column
    possible_values -= set(board[i][col] for i in range(9))
    
    # Remove values from the same 3x3 subgrid
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    possible_values -= set(board[i][j] for i in range(start_row, start_row + 3)
                                     for j in range(start_col, start_col + 3))
    
    return possible_values

def get_cell_with_least_entropy(board):
    min_entropy = float('inf')
    min_cell = None
    
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                possible_values = get_possible_values(board, i, j)
                if len(possible_values) < min_entropy:
                    min_entropy = len(possible_values)
                    min_cell = (i, j)
    
    return min_cell

def collapse_wave_function(board):
    def solve():
        cell = get_cell_with_least_entropy(board)
        if cell is None:
            return True
        
        row, col = cell
        possible_values = get_possible_values(board, row, col)
        
        for num in possible_values:
            board[row][col] = num
            if solve():
                return True
            board[row][col] = 0
        
        return False
    
    if solve():
        return board
    else:
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        grid = request.json.get('grid')  # Get the grid from JSON request
        if grid:
            grid = [grid[i:i + 9] for i in range(0, len(grid), 9)]
            solution = collapse_wave_function(grid)
            return jsonify({'solution': solution})
    return render_template('index.html', grid=None, solution=None)

@app.route('/get_board/<difficulty>')
def get_board(difficulty):
    if difficulty == 'easy':
        return jsonify(easy_board)
    elif difficulty == 'hard':
        return jsonify(hard_board)
    elif difficulty == 'expert':
        return jsonify(expert_board)
    else:
        return jsonify([])

if __name__ == '__main__':
    app.run(debug=True)
