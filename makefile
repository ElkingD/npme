.SECONDARY:

# Compiler and base flags
CPP = icpx
BASEFLAGS = -ffast-math -qopenmp -funroll-loops -qmkl -Wno-nan-infinity-disabled -MMD -MP
CFLAGS = -O3 -march=native $(BASEFLAGS)

# Directories
SRC_DIR = src
APP_DIR = app
OBJ_DIR = obj
EXE_DIR = exe

# Ensure output directories exist
$(shell mkdir -p $(OBJ_DIR) $(EXE_DIR))

# App executables (adjust as needed)
APPS = npme makeRandomBox laplaceDM RalphaDM helmholtzDM userDefineFunc compareV

# Object files
APP_OBJS = $(addprefix $(OBJ_DIR)/, $(addsuffix .o, $(APPS)))
LIB_SRCS = $(wildcard $(SRC_DIR)/*.cpp)
LIB_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(LIB_SRCS))

# Default target
all: $(addprefix $(EXE_DIR)/, $(APPS))

# AVX2 build target
avx2: CFLAGS := -O3 -march=native -mavx2 $(BASEFLAGS)
avx2: clean all

# AVX-512 build target
avx512: CFLAGS := -O3 -xCORE-AVX512 $(BASEFLAGS)
avx512: clean all

# Build each executable by linking app object with library objects
$(EXE_DIR)/%: $(OBJ_DIR)/%.o $(LIB_OBJS)
	$(CPP) $(CFLAGS) -o $@ $^ -lm

# Compile application source files
$(OBJ_DIR)/%.o: $(APP_DIR)/%.cpp
	$(CPP) $(CFLAGS) -I$(SRC_DIR) -c $< -o $@

# Compile library source files, with a special case for PotentialLaplace.cpp using -O1
obj/PotentialLaplace.o: src/PotentialLaplace.cpp
	$(CPP) -O1 $(filter-out -O3,$(CFLAGS)) -I$(SRC_DIR) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CPP) $(CFLAGS) -I$(SRC_DIR) -c $< -o $@

# Clean target
clean:
	rm -f $(OBJ_DIR)/*.o $(OBJ_DIR)/*.d $(EXE_DIR)/*

# Automatically include dependencies
-include $(OBJ_DIR)/*.d

