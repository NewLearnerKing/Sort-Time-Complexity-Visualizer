import time
import numpy as np
import matplotlib.pyplot as plt

def createArray(n):
    """Generate arrays for average, best, and worst case scenarios."""
    averg = np.random.randint(0, 1000, n)
    best = np.sort(averg)
    worst = best[::-1]
    return averg, best, worst


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

def calculate():
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
    i = 100
    while i<=2100:
        average, best, worst = createArray(i)
        bTime = bubbleSort(average.copy())
        bTime2 = bubbleSort(best.copy())
        bTime3 = bubbleSort(worst.copy())
        iTime = insertionSort(average.copy())
        iTime2 = insertionSort(best.copy())
        iTime3 = insertionSort(worst.copy())
        sTime = selectionSort(average.copy())
        sTime2 = selectionSort(best.copy())
        sTime3 = selectionSort(worst.copy())
        print(f"Array size: {i}")
        print(f"Bubble sort time: {bTime} seconds")
        print(f"Insertion sort time: {iTime} seconds")
        print(f"Selection sort time: {sTime} seconds")
        print()
        bListAvg.append(bTime)
        bListBest.append(bTime2)
        bListWorst.append(bTime3)
        iListAvg.append(iTime)
        iListBest.append(iTime2)
        iListWorst.append(iTime3)
        sListAvg.append(sTime)
        sListBest.append(sTime2)
        sListWorst.append(sTime3)
        numList.append(i)
        i = i+200
    return bListAvg, bListBest, bListWorst, iListAvg, iListBest, iListWorst, sListAvg, sListBest, sListWorst, numList
        

def graphPlot(bListAvg, bListBest, bListWorst, iListAvg, iListBest, iListWorst, sListAvg, sListBest, sListWorst, numList):
    """
    This function plots the time taken by each sorting algorithm for different array sizes.

    Args:
        bListAvg (list): List of time taken for bubble sort for random numbers
        bListBest (list): List of time taken for bubble sort for ascending numbers
        bListWorst (list): List of time taken for bubble sort for descending numbers
        iListAvg (list): List of time taken for insertion sort for random numbers
        iListBest (list): List of time taken for insertion sort for ascending numbers
        iListWorst (list): List of time taken for insertion sort for descending numbers
        sListAvg (list): List of time taken for selection sort for random numbers
        sListBest (list): List of time taken for selection sort for ascending numbers
        sListWorst (list): List of time taken for selection sort for descending numbers
        numList (list): List of numbers for size of arrays
    """
    plt.figure(figsize=(10,6))
    plt.plot(numList, bListAvg, label='Bubble sort average time')
    plt.plot(numList, iListAvg, label='Insertion sort average time')
    plt.plot(numList, sListAvg, label='Selection sort average time')
    plt.title('Average time complexity of sorting algorithms')
    plt.xlabel('Array size')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(10,6))
    plt.plot(numList, bListBest, label='Bubble sort best time')
    plt.plot(numList, iListBest, label='Insertion sort best time')
    plt.plot(numList, sListBest, label='Selection sort best time')
    plt.title('Best time complexity of sorting algorithms')
    plt.xlabel('Array size')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(10,6))
    plt.plot(numList, bListWorst, label='Bubble sort worst time')
    plt.plot(numList, iListWorst, label='Insertion sort worst time')
    plt.plot(numList, sListWorst, label='Selection sort worst time')
    plt.title('Worst time complexity of sorting algorithms')
    plt.xlabel('Array size')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.show()
        

if __name__ == "__main__":
    bListAvg, bListBest, bListWorst, iListAvg, iListBest, iListWorst, sListAvg, sListBest, sListWorst, numList = calculate()
    graphPlot(bListAvg, bListBest, bListWorst, iListAvg, iListBest, iListWorst, sListAvg, sListBest, sListWorst, numList)
