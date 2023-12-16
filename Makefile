CC = gcc
CFLAGS = -Wall -Wextra -g -O3
LDFLAGS = -lm -lsodium
TARGET = main
SRC = main.c mymodel.c
OBJ = $(SRC:.c=.o)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ) $(LDFLAGS)
	@rm -f $(OBJ)

%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f $(TARGET)

.PHONY: clean
