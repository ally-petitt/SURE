import torch
import time
import csv

def invert_matrix_and_measure_time(size):
    # Generate a random square matrix of given size
    matrix = torch.randn(size, size)

    # Measure the time taken to invert the matrix
    start_time = time.time()
    inverted_matrix = torch.inverse(matrix)
    end_time = time.time()

    # Calculate the time taken in seconds (with high precision)
    inversion_time = end_time - start_time

    return inversion_time

def main():
    # Define matrix sizes from 50 to 2650 (step size 50)
    matrix_sizes = list(range(50, 2700, 50))

    # Open a CSV file to save the results
    with open('pytorch_matrix_inversion_times.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Matrix Size', 'Time (seconds)'])

        # Iterate over each matrix size
        for size in matrix_sizes:
            time_taken = invert_matrix_and_measure_time(size)
            writer.writerow([size, f'{time_taken:.6f}'])

    print("Matrix inversion times saved to 'matrix_inversion_times.csv'.")

if __name__ == "__main__":
    main()

