import unittest
from sorting_comparison import bubbleSort, insertionSort, selectionSort, quickSort, mergeSort

class TestSortingAlgorithms(unittest.TestCase):
    def test_bubble_sort(self):
        arr = [64, 34, 25, 12, 22]
        bubbleSort(arr)
        self.assertEqual(arr, [12, 22, 25, 34, 64])

    def test_insertion_sort(self):
        arr = [64, 34, 25, 12, 22]
        insertionSort(arr)
        self.assertEqual(arr, [12, 22, 25, 34, 64])

    def test_selection_sort(self):
        arr = [64, 34, 25, 12, 22]
        selectionSort(arr)
        self.assertEqual(arr, [12, 22, 25, 34, 64])

    def test_quick_sort(self):
        arr = [64, 34, 25, 12, 22]
        quickSort(arr)
        self.assertEqual(arr, [12, 22, 25, 34, 64])

    def test_merge_sort(self):
        arr = [64, 34, 25, 12, 22]
        mergeSort(arr)
        self.assertEqual(arr, [12, 22, 25, 34, 64])

if __name__ == "__main__":
    unittest.main()