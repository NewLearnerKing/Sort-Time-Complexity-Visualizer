# Copyright (c) 2025 Abhishek Patel
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import time
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def createArray(n):
    if n <= 0:
        raise ValueError("Array size must be positive")
    try:
        avg = np.random.randint(0, 1000, n)
        best = np.sort(avg)
        worst = best[::-1]
        return avg, best, worst
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
    
    Args:
        arr (list): The input array to be sorted.
    
    Returns:
        float: Time taken to sort the array in seconds.
    """
    strTime = time.perf_counter()
    for i in range(1, len(arr)):
        key = arr[i]
        j = i-1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
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

def calculate(min_size=100, max_size=2101, step=200, trials=1):
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
    numList = []
    for i in range(min_size, max_size, step):
        b_times, b_times_best, b_times_worst = [], [], []
        i_times, i_times_best, i_times_worst = [], [], []
        s_times, s_times_best, s_times_worst = [], [], []
        for _ in range(trials):
            avg, best, worst = createArray(i)
            b_times.append(bubbleSort(avg.copy()))
            b_times_best.append(bubbleSort(best.copy()))
            b_times_worst.append(bubbleSort(worst.copy()))
            i_times.append(insertionSort(avg.copy()))
            i_times_best.append(insertionSort(best.copy()))
            i_times_worst.append(insertionSort(worst.copy()))
            s_times.append(selectionSort(avg.copy()))
            s_times_best.append(selectionSort(best.copy()))
            s_times_worst.append(selectionSort(worst.copy()))
        bListAvg.append(sum(b_times) / trials)
        bListBest.append(sum(b_times_best) / trials)
        bListWorst.append(sum(b_times_worst) / trials)
        iListAvg.append(sum(i_times) / trials)
        iListBest.append(sum(i_times_best) / trials)
        iListWorst.append(sum(i_times_worst) / trials)
        sListAvg.append(sum(s_times) / trials)
        sListBest.append(sum(s_times_best) / trials)
        sListWorst.append(sum(s_times_worst) / trials)
        numList.append(i)
    print(f"{'Array Size':<15} {'Bubble Sort':<15} {'Insertion Sort':<15} {'Selection Sort':<15}")
    print("-" * 60)
    for i, (b, i_, s) in enumerate(zip(bListAvg, iListAvg, sListAvg)):
        print(f"{numList[i]:<15} {b:<15.6f} {i_:<15.6f} {s:<15.6f}")
    return bListAvg, bListBest, bListWorst, iListAvg, iListBest, iListWorst, sListAvg, sListBest, sListWorst, numList
        
          
def graphPlot(bListAvg, bListBest, bListWorst, iListAvg, iListBest, iListWorst, sListAvg, sListBest, sListWorst, numList, log_scale=False):
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    cases = [("Average Case", bListAvg, iListAvg, sListAvg),
             ("Best Case", bListBest, iListBest, sListBest),
             ("Worst Case", bListWorst, iListWorst, sListWorst)]
    
    for i, (title, bData, iData, sData) in enumerate(cases):
        axs[i].plot(numList, bData, label='Bubble Sort', color='blue')
        axs[i].plot(numList, iData, label='Insertion Sort', color='green')
        axs[i].plot(numList, sData, label='Selection Sort', color='red')
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
    bListAvg, bListBest, bListWorst, iListAvg, iListBest, iListWorst, sListAvg, sListBest, sListWorst, numList = calculate(args.min_size, args.max_size, args.step, args.trials)
    graphPlot(bListAvg, bListBest, bListWorst, iListAvg, iListBest, iListWorst, sListAvg, sListBest, sListWorst, numList, args.log_scale)
