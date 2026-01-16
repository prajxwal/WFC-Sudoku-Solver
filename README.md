# WFC-Sudoku Solver

A Sudoku solver using a **Hybrid Wave Function Collapse + Constraint Satisfaction** approach.

## Features

- **Hybrid WFC-CSP Solver** - Domain-based constraint propagation with entropy-driven cell selection
- **Random Puzzle Generator** - Generate puzzles at Easy, Medium, Hard, and Expert difficulty
- **Performance Metrics** - Track solve time, iterations, backtracks, and propagations
- **Input Validation** - Validates puzzle constraints before solving
- **Minimal UI** - Clean black/white design with light/dark theme toggle
- **Solve Animation** - Watch cells fill in as the puzzle is solved

## How It Works

This is NOT simple backtracking. The solver uses:

1. **Domain Representation** - Each cell stores a set of possible values `{1-9}`
2. **Constraint Propagation** - When a cell collapses to one value, remove it from all peers (queue-based)
3. **Entropy Selection** - Always pick the cell with fewest possibilities first (fail-fast)
4. **Bounded Speculation** - Explicit state snapshots for backtracking, no hidden recursion

## Installation

```bash
git clone https://github.com/prajxwal/WFC-Sudoku-Solver.git
cd WFC-Sudoku-Solver
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

### Web Interface

- **Enter Puzzle** - Type numbers into the grid (1-9), leave empty for blanks
- **Solve** - Click to solve using the WFC-CSP algorithm
- **Clear** - Reset the grid
- **Generate** - Create random puzzles at different difficulties

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | POST | Solve puzzle (JSON body: `{grid: [81 integers]}`) |
| `/generate/<difficulty>` | GET | Generate puzzle (easy/medium/hard/expert) |
| `/validate` | POST | Validate puzzle constraints |

## Algorithm Details

```
1. Initialize domains for all cells
   - Fixed cells: domain = {value}
   - Empty cells: domain = {1-9}, pruned by constraints

2. Propagate constraints (queue-based)
   - When domain.size == 1, remove value from 20 peers
   - Chain reaction of collapses

3. If not solved, speculate:
   - Snapshot state
   - Pick min-entropy cell
   - Try each value with propagation
   - Restore on contradiction

4. Repeat until solved or exhausted
```

## Requirements

- Python 3.8+
- Flask
- NumPy

## License

MIT License - see [LICENSE](LICENSE)
