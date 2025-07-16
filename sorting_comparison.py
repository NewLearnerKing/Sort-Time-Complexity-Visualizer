# Copyright (c) 2025 Abhishek Patel
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import time
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from bisect import bisect_left

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def createArray(n):
    if n <= 0:
        raise ValueError("Array size must be positive")
    try:
        avg = np.random.randint(0, 1000, n)
        best = np.sort(avg)
        worst = best[::-1]
        # For quick sort and merge sort:
        # Best for quick sort: shuffled (random), worst: sorted or reverse-sorted
        # For merge sort, both best and worst are similar, but we use sorted and reverse-sorted for completeness
        quick_best = np.random.permutation(n)  # random permutation
        quick_worst = best  # sorted array (worst for naive quick sort)
        merge_best = best
        merge_worst = worst
        return avg, best, worst, quick_best, quick_worst, merge_best, merge_worst
    except MemoryError:
        raise MemoryError(f"Cannot allocate array of size {n}")


def bubbleSort(arr):
    """
    Performs bubble sort on the input array and returns the time taken.
    
    Args:
        arr (list): The input array to be sorted.
    
    Returns:
        float: Time taken to sort the array in seconds.
    """
    strTime = time.perf_counter()
    n = len(arr)
    for i in range(n):
        flag = 0
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                flag = 1
        if flag == 0:
            break
    totalTime = time.perf_counter() - strTime
    return totalTime


def insertionSort(arr):
    """
    Performs insertion sort on the input array and returns the time taken.
    Uses binary search to find the insertion point.
    """
    strTime = time.perf_counter()
    for i in range(1, len(arr)):
        key = arr[i]
        # Find location where key should be inserted
        j = bisect_left(arr, key, 0, i)
        # Shift elements to make room for key
        arr[j+1:i+1] = arr[j:i]
        arr[j] = key
    totalTime = time.perf_counter() - strTime
    return totalTime


def selectionSort(arr):
    """
    Performs selection sort on the input array and returns the time taken.
    
    Args:
        arr (list): The input array to be sorted.
    
    Returns:
        float: Time taken to sort the array in seconds.
    """
    strTime = time.perf_counter()
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[min_idx] > arr[j]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    totalTime = time.perf_counter() - strTime
    return totalTime

def quickSort(arr):
    """
    Performs quick sort on the input array and returns the time taken.
    Uses an explicit stack to avoid recursion limit issues.
    """
    strTime = time.perf_counter()
    stack = [(0, len(arr) - 1)]
    while stack:
        low, high = stack.pop()
        if low < high:
            # Partition
            pivot = arr[high]
            i = low - 1
            for j in range(low, high):
                if arr[j] <= pivot:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]
            arr[i+1], arr[high] = arr[high], arr[i+1]
            pi = i + 1
            # Push subarrays to stack
            stack.append((low, pi - 1))
            stack.append((pi + 1, high))
    totalTime = time.perf_counter() - strTime
    return totalTime


def mergeSort(arr):
    """
    Performs merge sort on the input array and returns the time taken.
    Sorts in-place.
    """
    def _mergeSort(a):
        if len(a) > 1:
            mid = len(a) // 2
            L = a[:mid]
            R = a[mid:]
            _mergeSort(L)
            _mergeSort(R)
            i = j = k = 0
            while i < len(L) and j < len(R):
                if L[i] < R[j]:
                    a[k] = L[i]
                    i += 1
                else:
                    a[k] = R[j]
                    j += 1
                k += 1
            while i < len(L):
                a[k] = L[i]
                i += 1
                k += 1
            while j < len(R):
                a[k] = R[j]
                j += 1
                k += 1
    strTime = time.perf_counter()
    _mergeSort(arr)
    totalTime = time.perf_counter() - strTime
    return totalTime

def calculate(min_size=100, max_size=2101, step=200, trials=5):
    """
    Calculates the time taken by different sorting algorithms on a large array.
    Returns:
        Different lists with time taken and total number of elements
    """
    bListAvg = []
    bListBest = []
    bListWorst = []
    iListAvg = []
    iListBest = []
    iListWorst = []
    sListAvg = []
    sListBest = []
    sListWorst = []
    qListAvg = []
    qListBest = []
    qListWorst = []
    mListAvg = []
    mListBest = []
    mListWorst = []
    numList = []
    for i in range(min_size, max_size, step):
        b_times, b_times_best, b_times_worst = [], [], []
        i_times, i_times_best, i_times_worst = [], [], []
        s_times, s_times_best, s_times_worst = [], [], []
        q_times, q_times_best, q_times_worst = [], [], []
        m_times, m_times_best, m_times_worst = [], [], []
        for _ in range(trials):
            avg, best, worst, quick_best, quick_worst, merge_best, merge_worst = createArray(i)
            b_times.append(bubbleSort(avg.copy()))
            b_times_best.append(bubbleSort(best.copy()))
            b_times_worst.append(bubbleSort(worst.copy()))
            i_times.append(insertionSort(avg.copy()))
            i_times_best.append(insertionSort(best.copy()))
            i_times_worst.append(insertionSort(worst.copy()))
            s_times.append(selectionSort(avg.copy()))
            s_times_best.append(selectionSort(best.copy()))
            s_times_worst.append(selectionSort(worst.copy()))
            q_times.append(quickSort(avg.copy()))
            q_times_best.append(quickSort(quick_best.copy()))
            q_times_worst.append(quickSort(quick_worst.copy()))
            m_times.append(mergeSort(avg.copy()))
            m_times_best.append(mergeSort(merge_best.copy()))
            m_times_worst.append(mergeSort(merge_worst.copy()))
        bListAvg.append(sum(b_times) / trials)
        bListBest.append(sum(b_times_best) / trials)
        bListWorst.append(sum(b_times_worst) / trials)
        iListAvg.append(sum(i_times) / trials)
        iListBest.append(sum(i_times_best) / trials)
        iListWorst.append(sum(i_times_worst) / trials)
        sListAvg.append(sum(s_times) / trials)
        sListBest.append(sum(s_times_best) / trials)
        sListWorst.append(sum(s_times_worst) / trials)
        qListAvg.append(sum(q_times) / trials)
        qListBest.append(sum(q_times_best) / trials)
        qListWorst.append(sum(q_times_worst) / trials)
        mListAvg.append(sum(m_times) / trials)
        mListBest.append(sum(m_times_best) / trials)
        mListWorst.append(sum(m_times_worst) / trials)
        numList.append(i)
    print(f"{'Array Size':<15} {'Bubble Sort':<15} {'Insertion Sort':<15} {'Selection Sort':<15} {'Quick Sort':<15} {'Merge Sort':<15}")
    print("-" * 90)
    for i, (b, ins, s, q, m) in enumerate(zip(bListAvg, iListAvg, sListAvg, qListAvg, mListAvg)):
        print(f"{numList[i]:<15} {b:<15.6f} {ins:<15.6f} {s:<15.6f} {q:<15.6f} {m:<15.6f}")
    return (bListAvg, bListBest, bListWorst,
            iListAvg, iListBest, iListWorst,
            sListAvg, sListBest, sListWorst,
            qListAvg, qListBest, qListWorst,
            mListAvg, mListBest, mListWorst,
            numList)
        
          
def graphPlot(bListAvg, bListBest, bListWorst, iListAvg, iListBest, iListWorst, sListAvg, sListBest, sListWorst, qListAvg, qListBest, qListWorst, mListAvg, mListBest, mListWorst, numList, log_scale=False):
    fig, axs = plt.subplots(3, 1, figsize=(12, 14))
    # Ensure axs is always a list of Axes
    import numpy as np
    if isinstance(axs, np.ndarray):
        axs = list(axs.ravel())
    else:
        axs = [axs]
    cases = [
        ("Average Case", bListAvg, iListAvg, sListAvg, qListAvg, mListAvg),
        ("Best Case", bListBest, iListBest, sListBest, qListBest, mListBest),
        ("Worst Case", bListWorst, iListWorst, sListWorst, qListWorst, mListWorst)
    ]
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    labels = ['Bubble Sort', 'Insertion Sort', 'Selection Sort', 'Quick Sort', 'Merge Sort']
    for i, (title, bData, iData, sData, qData, mData) in enumerate(cases):
        axs[i].plot(numList, bData, label=labels[0], color=colors[0])
        axs[i].plot(numList, iData, label=labels[1], color=colors[1])
        axs[i].plot(numList, sData, label=labels[2], color=colors[2])
        axs[i].plot(numList, qData, label=labels[3], color=colors[3])
        axs[i].plot(numList, mData, label=labels[4], color=colors[4])
        axs[i].set_title(f'{title} Time Complexity')
        axs[i].set_xlabel('Array Size')
        axs[i].set_ylabel('Time (seconds)')
        axs[i].grid(True, which="both", ls="--", alpha=0.5)
        if log_scale:
            axs[i].set_yscale('log')
        axs[i].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"compare{'_log' if log_scale else ''}.png"))
    plt.show()
    plt.close()
           

def parse_args():
    parser = argparse.ArgumentParser(description="Compare sorting algorithm performance.")
    parser.add_argument("--min-size", type=int, default=100, help="Minimum array size")
    parser.add_argument("--max-size", type=int, default=2100, help="Maximum array size")
    parser.add_argument("--step", type=int, default=200, help="Step size for array sizes")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials for averaging")
    parser.add_argument("--log-scale", action="store_true", help="Use logarithmic scale for plots")
    args = parser.parse_args()
    if args.min_size <= 0 or args.max_size <= 0 or args.step <= 0 or args.trials <= 0:
        parser.error("All arguments must be positive")
    if args.min_size >= args.max_size:
        parser.error("min-size must be less than max-size")
    return args

if __name__ == "__main__":
    print("Running...")
    args = parse_args()
    bListAvg, bListBest, bListWorst, iListAvg, iListBest, iListWorst, sListAvg, sListBest, sListWorst, qListAvg, qListBest, qListWorst, mListAvg, mListBest, mListWorst, numList = calculate(args.min_size, args.max_size, args.step, args.trials)
    graphPlot(bListAvg, bListBest, bListWorst, iListAvg, iListBest, iListWorst, sListAvg, sListBest, sListWorst, qListAvg, qListBest, qListWorst, mListAvg, mListBest, mListWorst, numList, args.log_scale)
