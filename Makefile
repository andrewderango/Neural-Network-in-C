######### WSL Makefile #########

# CC = gcc
# CFLAGS = -Wall -Wextra -g -O3
# LDFLAGS = -lm -lsodium
# TARGET = ANN
# SRC = main.c mymodel.c
# OBJ = $(SRC:.c=.o)

# $(TARGET): $(OBJ)
# 	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ) $(LDFLAGS)
# 	@rm -f $(OBJ)

# %.o: %.c
# 	$(CC) $(CFLAGS) -c $<

# clean:
# 	rm -f $(TARGET)

# .PHONY: clean

######## MacOS Makefile #########

CC = gcc
CFLAGS = -Wall -Wextra -g -O3 -I/opt/homebrew/Cellar/libsodium/1.0.19/include
LDFLAGS = -L/opt/homebrew/Cellar/libsodium/1.0.19/lib -lm -lsodium
TARGET = ANN
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