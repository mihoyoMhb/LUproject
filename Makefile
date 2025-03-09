# Makefile for Matrix Decomposition Project
# Last updated: 2025-03-09 by mihoyoMhb

# Compiler and flags
CC = gcc
# Added proper OpenMP flags for parallel execution
CFLAGS = -Wall -Wextra -O2 -fopenmp -ftree-vectorize -march=native -fopt-info-vec-optimized
LDFLAGS = -lm -fopenmp

# Target executable
TARGET = matrix_test

# Source files and object files
SRC = main.c LU_decomposition.c LU_optimized.c  Test_LU.c
OBJ = $(SRC:.c=.o)
HEADERS = LU_decomposition.h LU_optimized.h  Test_LU.h

# Default target
all: $(TARGET)

# Link object files to create executable
$(TARGET): $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

# Compile source files to object files
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up generated files
clean:
	rm -f $(OBJ) $(TARGET)

# Clean and rebuild
rebuild: clean all

# Run the program
run: $(TARGET)
	./$(TARGET)

# Performance test
perf: $(TARGET)
	./$(TARGET)

# Phony targets
.PHONY: all clean rebuild run perf