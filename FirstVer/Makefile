CC = gcc
CFLAGS = -O3 -march=native -fopenmp -ffast-math -ftree-vectorize -fopt-info-vec
TARGET = Parallel
SRC = Parallel.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) -o $(TARGET) $(SRC) $(CFLAGS)

clean:
	rm -f $(TARGET)