# WFC-Sudoku Solver

## Overview

This Sudoku Solver application allows users to input Sudoku puzzles and find solutions using a Wave Function Collapse (WFC)-inspired algorithm. The web application features a user-friendly interface where users can enter their own puzzles or choose from predefined easy, hard, and expert puzzles.

## Features

- **Sudoku Puzzle Input:** Enter Sudoku puzzles into a 9x9 grid.
- **Solve Button:** Submit the puzzle to solve it using the WFC-inspired algorithm.
- **Clear Button:** Reset the grid to start a new puzzle.
- **Example Boards:** Load predefined Sudoku puzzles of varying difficulty levels.
- **Responsive Design:** User-friendly interface with a dark theme.

## Requirements

- **Python 3.x**
- **Flask:** Web framework for Python
- **NumPy:** Library for numerical operations

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/prajxwal/sudoku-solver.git
   cd sudoku-solver
   .

1. **Install Dependencies from `requirements.txt`:**

   ```bash
   pip install -r requirements.txt



## Usage
1. **Run the Application:**

    ```bash
   python app.py
   
 **The application will be available at http://127.0.0.1:5000/ by default.**
 ## Using the Web Interface

- **Input Puzzle:** Enter numbers into the Sudoku grid. Empty cells are represented by blank spaces.
- **Solve Puzzle:** Click the "Solve" button to find the solution. The solution will be displayed in the grid.
- **Clear Puzzle:** Click the "Clear" button to reset the grid.
- **Load Example Boards:** Click on the "Easy", "Hard", or "Expert" buttons to load predefined Sudoku puzzles.

## Wave Function Collapse Algorithm

The solver uses a WFC-inspired approach:

1. **Possible Values Calculation:** For each empty cell, calculate the possible values based on Sudoku constraints (row, column, 3x3 subgrid).

2. **Entropy Minimization:** Select the cell with the fewest possible values to place a number.

3. **Backtracking:** Place numbers and recursively solve the puzzle, backtracking if necessary.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

   
   
   




