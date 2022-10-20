COMPILER=g++

# Flags
CFLAGS=-O3 -Wall

# Files to compile
OBJECTS=solution.cpp

# Executable name
EXE=solution


all: $(OBJECTS)
	$(COMPILER) $(OBJECTS) $(CFLAGS) -o $(EXE)

run:
	./$(EXE)

clean:
	rm $(EXE)
