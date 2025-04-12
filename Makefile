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
CFLAGS = -Wall -Wextra -g -O3 -I/opt/homebrew/opt/libsodium/include
LDFLAGS = -L/opt/homebrew/opt/libsodium/lib -lm -lsodium
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