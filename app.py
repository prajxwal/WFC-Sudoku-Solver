from flask import Flask, request, render_template, jsonify
import numpy as np
import time
import random
import copy

app = Flask(__name__)


# ============== VALIDATION FUNCTIONS ==============

def validate_grid_format(grid):
    """Validate that the grid has the correct format (81 cells, values 0-9)."""
    if not isinstance(grid, list):
        return False, "Grid must be a list"
    
    if len(grid) != 81:
        return False, f"Grid must have exactly 81 cells, got {len(grid)}"
    
    for i, val in enumerate(grid):
        if not isinstance(val, int):
            return False, f"Cell {i} must be an integer"
        if val < 0 or val > 9:
            return False, f"Cell {i} has invalid value {val}. Must be 0-9"
    
    return True, None


def validate_sudoku_constraints(board):
    """Check if the current board state violates any Sudoku constraints."""
    errors = []
    
    # Check rows
    for i in range(9):
        row = [board[i][j] for j in range(9) if board[i][j] != 0]
        duplicates = [x for x in set(row) if row.count(x) > 1]
        if duplicates:
            errors.append(f"Row {i + 1} has duplicate values: {duplicates}")
    
    # Check columns
    for j in range(9):
        col = [board[i][j] for i in range(9) if board[i][j] != 0]
        duplicates = [x for x in set(col) if col.count(x) > 1]
        if duplicates:
            errors.append(f"Column {j + 1} has duplicate values: {duplicates}")
    
    # Check 3x3 boxes
    for box_row in range(3):
        for box_col in range(3):
            box = []
            for i in range(3):
                for j in range(3):
                    val = board[box_row * 3 + i][box_col * 3 + j]
                    if val != 0:
                        box.append(val)
            duplicates = [x for x in set(box) if box.count(x) > 1]
            if duplicates:
                errors.append(f"Box ({box_row + 1}, {box_col + 1}) has duplicate values: {duplicates}")
    
    return len(errors) == 0, errors


# ============== WFC-INSPIRED CSP SOLVER ==============
# This is a Hybrid Wave Function Collapse + Constraint Satisfaction solver.
# It uses domain-based state, queue-based constraint propagation, and
# entropy-driven cell selection with bounded speculation.

from typing import Optional, Tuple
from collections import deque


def get_peers(row: int, col: int) -> list[Tuple[int, int]]:
    """Get all peer cells (same row, column, or 3x3 box)."""
    peers = set()
    
    # Same row
    for c in range(9):
        if c != col:
            peers.add((row, c))
    
    # Same column
    for r in range(9):
        if r != row:
            peers.add((r, col))
    
    # Same 3x3 box
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if (r, c) != (row, col):
                peers.add((r, c))
    
    return list(peers)


# Pre-compute peers for performance
PEERS_CACHE = [[get_peers(r, c) for c in range(9)] for r in range(9)]


def initialize_domains(grid: list[list[int]]) -> Optional[list[list[set[int]]]]:
    """
    Initialize domains for all cells.
    - Fixed cells have domain = {value}
    - Empty cells start with {1-9} and are pruned by constraints
    Returns None if initial state has contradictions.
    """
    # Create initial domains
    domains = [[set(range(1, 10)) if grid[r][c] == 0 else {grid[r][c]} 
                for c in range(9)] for r in range(9)]
    
    # Prune domains based on initial constraints
    for r in range(9):
        for c in range(9):
            if grid[r][c] != 0:
                value = grid[r][c]
                # Remove this value from all peers
                for pr, pc in PEERS_CACHE[r][c]:
                    domains[pr][pc].discard(value)
                    if len(domains[pr][pc]) == 0:
                        return None  # Contradiction in initial state
    
    return domains


def domains_to_board(domains: list[list[set[int]]]) -> list[list[int]]:
    """Convert domains to board representation. Collapsed cells show value, others show 0."""
    return [[list(domains[r][c])[0] if len(domains[r][c]) == 1 else 0 
             for c in range(9)] for r in range(9)]


def is_solved(domains: list[list[set[int]]]) -> bool:
    """Check if all domains have collapsed to single values."""
    return all(len(domains[r][c]) == 1 for r in range(9) for c in range(9))


def propagate(domains: list[list[set[int]]], queue: deque, stats: dict) -> bool:
    """
    Queue-based constraint propagation.
    When a domain collapses to size 1, remove that value from all peers.
    Returns True if consistent, False if contradiction detected.
    """
    while queue:
        row, col = queue.popleft()
        stats['propagations'] += 1
        
        if len(domains[row][col]) == 0:
            return False  # Contradiction
        
        if len(domains[row][col]) != 1:
            continue  # Not collapsed yet, skip
        
        value = list(domains[row][col])[0]
        
        # Remove value from all peers
        for pr, pc in PEERS_CACHE[row][col]:
            if value in domains[pr][pc]:
                domains[pr][pc].discard(value)
                
                if len(domains[pr][pc]) == 0:
                    return False  # Contradiction
                
                if len(domains[pr][pc]) == 1:
                    queue.append((pr, pc))  # Newly collapsed, propagate
    
    return True


def select_min_entropy_cell(domains: list[list[set[int]]]) -> Optional[Tuple[int, int]]:
    """
    Select the cell with minimum entropy (fewest possibilities > 1).
    Returns None if all cells are collapsed.
    Uses randomization for tie-breaking.
    """
    min_entropy = float('inf')
    candidates = []
    
    for r in range(9):
        for c in range(9):
            size = len(domains[r][c])
            if size > 1:
                if size < min_entropy:
                    min_entropy = size
                    candidates = [(r, c)]
                elif size == min_entropy:
                    candidates.append((r, c))
    
    if not candidates:
        return None
    
    return random.choice(candidates)


def deep_copy_domains(domains: list[list[set[int]]]) -> list[list[set[int]]]:
    """Create a deep copy of domains for speculation snapshots."""
    return [[domains[r][c].copy() for c in range(9)] for r in range(9)]


def solve_sudoku(grid: list[list[int]], stats: dict) -> Tuple[Optional[list[list[int]]], bool]:
    """
    Solve Sudoku using Hybrid WFC-inspired CSP approach.
    
    Algorithm:
    1. Initialize domains from grid
    2. Run deterministic constraint propagation
    3. If not solved, perform bounded speculation:
       - Snapshot state
       - Pick min-entropy cell
       - Try each value with propagation
       - Backtrack on contradiction
    
    Returns (solution, success) tuple.
    """
    # Initialize domains
    domains = initialize_domains(grid)
    if domains is None:
        return None, False  # Invalid initial state
    
    # Initial propagation for all collapsed cells
    queue = deque()
    for r in range(9):
        for c in range(9):
            if len(domains[r][c]) == 1:
                queue.append((r, c))
    
    if not propagate(domains, queue, stats):
        return None, False
    
    # Speculation stack for bounded, explicit backtracking
    # Each entry: (domains_snapshot, cell, remaining_values)
    speculation_stack = []
    max_speculation_depth = 81  # Hard limit
    max_iterations = 100000  # Safety limit to prevent infinite loops
    
    while True:
        stats['iterations'] += 1
        
        # Safety check for infinite loops
        if stats['iterations'] > max_iterations:
            return None, False
        
        if is_solved(domains):
            return domains_to_board(domains), True
        
        # Select min-entropy cell
        cell = select_min_entropy_cell(domains)
        
        if cell is None:
            # No cell to fill but not solved - need to backtrack
            while speculation_stack:
                domains, (prev_row, prev_col), remaining = speculation_stack.pop()
                stats['backtracks'] += 1
                
                # Try remaining values for this cell
                while remaining:
                    value = remaining.pop()
                    test_domains = deep_copy_domains(domains)
                    test_domains[prev_row][prev_col] = {value}
                    
                    queue = deque([(prev_row, prev_col)])
                    if propagate(test_domains, queue, stats):
                        # This value works, push back and continue
                        speculation_stack.append((domains, (prev_row, prev_col), remaining))
                        domains = test_domains
                        stats['cells_collapsed'] += 1
                        break
                else:
                    # All values exhausted, continue backtracking
                    continue
                break  # Found a valid value, exit backtrack loop
            else:
                # Stack exhausted, no solution
                return None, False
            continue
        
        row, col = cell
        remaining_values = list(domains[row][col])
        random.shuffle(remaining_values)  # Randomize for variety
        
        if len(speculation_stack) >= max_speculation_depth:
            return None, False  # Too deep, give up
        
        # Try values for this cell
        while remaining_values:
            value = remaining_values.pop()
            test_domains = deep_copy_domains(domains)
            test_domains[row][col] = {value}
            stats['cells_collapsed'] += 1
            stats['speculations'] += 1
            
            queue = deque([(row, col)])
            if propagate(test_domains, queue, stats):
                # This value works
                speculation_stack.append((domains, (row, col), remaining_values))
                domains = test_domains
                break
            else:
                stats['backtracks'] += 1
        else:
            # All values failed for this cell - need to backtrack
            if not speculation_stack:
                return None, False
            
            # Backtrack to previous speculation
            domains, (prev_row, prev_col), remaining = speculation_stack.pop()
            stats['backtracks'] += 1
            
            # Put this cell back to try more values
            if remaining:
                speculation_stack.append((domains, (prev_row, prev_col), remaining))



# ============== PUZZLE GENERATOR ==============

def generate_solved_puzzle():
    """Generate a complete solved Sudoku puzzle."""
    board = [[0] * 9 for _ in range(9)]
    
    # Fill diagonal 3x3 boxes first (they don't affect each other)
    for box in range(3):
        nums = list(range(1, 10))
        random.shuffle(nums)
        for i in range(3):
            for j in range(3):
                board[box * 3 + i][box * 3 + j] = nums[i * 3 + j]
    
    # Solve the rest using WFC-CSP solver
    stats = {'iterations': 0, 'backtracks': 0, 'propagations': 0, 'cells_collapsed': 0, 'speculations': 0}
    solution, success = solve_sudoku(board, stats)
    
    if success and solution:
        return solution
    else:
        # Fallback: try again with a fresh board
        return generate_solved_puzzle()


def remove_numbers(board, difficulty):
    """Remove numbers from a solved puzzle based on difficulty."""
    puzzle = copy.deepcopy(board)
    
    # Number of cells to remove based on difficulty
    cells_to_remove = {
        'easy': 35,      # ~46 clues remaining
        'medium': 45,    # ~36 clues remaining
        'hard': 50,      # ~31 clues remaining
        'expert': 55     # ~26 clues remaining
    }
    
    remove_count = cells_to_remove.get(difficulty, 40)
    cells = [(i, j) for i in range(9) for j in range(9)]
    random.shuffle(cells)
    
    removed = 0
    for row, col in cells:
        if removed >= remove_count:
            break
        
        if puzzle[row][col] != 0:
            puzzle[row][col] = 0
            removed += 1
    
    return puzzle


def generate_puzzle(difficulty='medium'):
    """Generate a new Sudoku puzzle with the specified difficulty."""
    solved = generate_solved_puzzle()
    puzzle = remove_numbers(solved, difficulty)
    return puzzle


# ============== API ROUTES ==============

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            data = request.json
            
            if not data or 'grid' not in data:
                return jsonify({
                    'success': False,
                    'error': 'No grid data provided'
                }), 400
            
            grid = data.get('grid')
            
            # Validate grid format
            is_valid, error = validate_grid_format(grid)
            if not is_valid:
                return jsonify({
                    'success': False,
                    'error': f'Invalid grid format: {error}'
                }), 400
            
            # Convert to 2D array
            board = [grid[i:i + 9] for i in range(0, len(grid), 9)]
            
            # Validate Sudoku constraints
            is_valid, errors = validate_sudoku_constraints(board)
            if not is_valid:
                return jsonify({
                    'success': False,
                    'error': 'Invalid Sudoku puzzle',
                    'validation_errors': errors
                }), 400
            
            # Count empty cells
            empty_cells = sum(1 for row in board for cell in row if cell == 0)
            
            # Solve the puzzle with performance tracking
            stats = {
                'iterations': 0,
                'backtracks': 0,
                'propagations': 0,
                'cells_collapsed': 0,
                'speculations': 0
            }
            
            start_time = time.perf_counter()
            solution, success = solve_sudoku(board, stats)
            solve_time = time.perf_counter() - start_time
            
            if success:
                return jsonify({
                    'success': True,
                    'solution': solution,
                    'stats': {
                        'solve_time_ms': round(solve_time * 1000, 2),
                        'iterations': stats['iterations'],
                        'backtracks': stats['backtracks'],
                        'propagations': stats['propagations'],
                        'speculations': stats['speculations'],
                        'cells_collapsed': stats['cells_collapsed'],
                        'empty_cells_initial': empty_cells
                    }
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'No solution exists for this puzzle'
                }), 400
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Server error: {str(e)}'
            }), 500
    
    return render_template('index.html', grid=None, solution=None)


@app.route('/generate/<difficulty>')
def generate_board(difficulty):
    """Generate a new random Sudoku puzzle."""
    try:
        valid_difficulties = ['easy', 'medium', 'hard', 'expert']
        
        if difficulty not in valid_difficulties:
            return jsonify({
                'success': False,
                'error': f'Invalid difficulty. Choose from: {valid_difficulties}'
            }), 400
        
        start_time = time.perf_counter()
        puzzle = generate_puzzle(difficulty)
        generation_time = time.perf_counter() - start_time
        
        clues = sum(1 for row in puzzle for cell in row if cell != 0)
        
        return jsonify({
            'success': True,
            'puzzle': puzzle,
            'difficulty': difficulty,
            'clues': clues,
            'generation_time_ms': round(generation_time * 1000, 2)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to generate puzzle: {str(e)}'
        }), 500


@app.route('/validate', methods=['POST'])
def validate_puzzle():
    """Validate a Sudoku puzzle without solving it."""
    try:
        data = request.json
        
        if not data or 'grid' not in data:
            return jsonify({
                'success': False,
                'error': 'No grid data provided'
            }), 400
        
        grid = data.get('grid')
        
        # Validate format
        is_valid, error = validate_grid_format(grid)
        if not is_valid:
            return jsonify({
                'valid': False,
                'error': error
            }), 400
        
        # Convert and validate constraints
        board = [grid[i:i + 9] for i in range(0, len(grid), 9)]
        is_valid, errors = validate_sudoku_constraints(board)
        
        return jsonify({
            'valid': is_valid,
            'errors': errors if not is_valid else []
        })
        
    except Exception as e:
        return jsonify({
            'valid': False,
            'error': f'Validation error: {str(e)}'
        }), 500


if __name__ == '__main__':
    app.run(debug=True)
