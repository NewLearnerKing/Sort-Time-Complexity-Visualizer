# Sorting Time Visualization

A Python project that compares and visualizes the performance of five fundamental sorting algorithms: Bubble Sort, Insertion Sort, Selection Sort, Quick Sort, and Merge Sort. The project generates performance graphs showing how these algorithms perform under different scenarios (best case, average case, and worst case).

## 📊 Features

- **Five Sorting Algorithms**: Implements Bubble Sort, Insertion Sort (with binary search), Selection Sort, Quick Sort (iterative, stack-based for large arrays), and Merge Sort
- **Performance Analysis**: Measures execution time for different array sizes
- **Multiple Test Cases**: Tests best case, average case, and worst case scenarios
- **Visualization**: Generates comparative graphs using matplotlib
- **Configurable Parameters**: Customizable array sizes, step sizes, and number of trials
- **Unit Tests**: Comprehensive test suite for algorithm verification

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/NewLearnerKing/Sort-Time-Complexity-Visualizer.git
   cd sort-time-visualization
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📋 Requirements

- Python 3.6+
- numpy
- matplotlib

## 🎯 Usage

### Basic Usage

Run the sorting comparison with default parameters:
```bash
python sorting_comparison.py
```

### Advanced Usage

Customize the analysis parameters:
```bash
python sorting_comparison.py --min-size 50 --max-size 10000 --step 500 --trials 10 --log-scale
```

### Command Line Arguments

- `--min-size`: Minimum array size (default: 100)
- `--max-size`: Maximum array size (default: 2100)
- `--step`: Step size between array sizes (default: 200)
- `--trials`: Number of trials for averaging (default: 5)

### Running Tests

Execute the test suite to verify algorithm correctness:
```bash
python test_sorting.py
```

## 📈 Output

The program generates:
1. **Console Output**: Tabular comparison of sorting times for different array sizes
2. **Graph**: A comprehensive visualization saved as `plots/compare.png` showing:
   - Best case performance
   - Average case performance  
   - Worst case performance

## 🔬 Algorithm Details

### Bubble Sort
- **Time Complexity**: O(n²) average and worst case, O(n) best case
- **Space Complexity**: O(1)
- **Stability**: Stable

### Insertion Sort (with binary search)
- **Time Complexity**: O(n²) average and worst case, O(n) best case
- **Space Complexity**: O(1)
- **Stability**: Stable

### Selection Sort
- **Time Complexity**: O(n²) for all cases
- **Space Complexity**: O(1)
- **Stability**: Unstable

### Quick Sort (iterative, stack-based, robust for large arrays)
- **Time Complexity**: O(n log n) average case, O(n²) worst case
- **Space Complexity**: O(log n)
- **Stability**: Unstable

### Merge Sort
- **Time Complexity**: O(n log n) for all cases
- **Space Complexity**: O(n)
- **Stability**: Stable

## 🚦 Performance Notes

While Merge Sort has a better asymptotic time complexity (O(n log n)) than Insertion Sort (O(n²)), for small arrays, Insertion Sort (especially with binary search for the insertion point) can outperform Merge Sort in practice. This is due to lower constant factors, reduced overhead, and better cache locality. As a result, you may observe in the generated graphs that Insertion Sort is sometimes faster than Merge Sort for small input sizes. This practical behavior is why hybrid algorithms (like Timsort, used in Python’s built-in sort) switch to Insertion Sort for small subarrays.

## 📁 Project Structure

```
sort-time-visualization/
├── sorting_comparison.py    # Main program with sorting algorithms
├── test_sorting.py          # Unit tests for algorithms
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── LICENSE                  # Project license
├── .gitignore               # Excludes __pycache__, plot outputs, and other temp files from git
└── plots/                   # Generated visualization files
    └── compare.png          # Performance comparison graph
```

## 🧪 Testing

The project includes comprehensive unit tests that verify:
- Correctness of all five sorting algorithms (Bubble Sort, Insertion Sort, Selection Sort, Quick Sort, Merge Sort)
- Proper handling of various input arrays
- Expected sorted output

Run tests with:
```bash
python -m unittest test_sorting.py
```

## 📊 Sample Output

The program outputs a table showing execution times:

```
Array Size      Bubble Sort     Insertion Sort  Selection Sort  Quick Sort      Merge Sort      
---------------------------------------------------------------------------------------------
100            0.000123        0.000045        0.000078        0.000032        0.000029        
300            0.001234        0.000456        0.000789        0.000201        0.000190        
500            0.003456        0.001234        0.002345        0.000410        0.000398        
...
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 👨‍💻 Author

**Abhishek Patel**

## 🙏 Acknowledgments

- Inspired by the need to understand sorting algorithm performance characteristics
- Built for educational purposes in algorithm analysis and visualization

## 📞 Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

**Note**: This project is designed for educational purposes and demonstrates fundamental sorting algorithms. For production use, consider using Python's built-in `sort()` method or `sorted()` function, which implement highly optimized sorting algorithms. Quick Sort is implemented iteratively (stack-based) to avoid recursion depth issues and is robust for large input sizes. Insertion Sort uses binary search for efficiency. Unit tests for Quick Sort and Merge Sort are included.
