# Compiler and flags
CPP = icpc
CFLAGS = -Ofast -fopenmp -march=native -ffast-math -funroll-all-loops -Werror-all -mkl

# Directories
TOP_DIR := $(CURDIR)
SRC_DIR := $(TOP_DIR)/src
APP_DIR := $(TOP_DIR)/app
OBJ_DIR := $(TOP_DIR)/obj
EXE_DIR := $(TOP_DIR)/exe

# Ensure obj and exe directories exist
$(shell mkdir -p $(OBJ_DIR) $(EXE_DIR))

# Include directories
INCLUDES := -I$(SRC_DIR)

# Library source and object files
LIB_SRC := $(wildcard $(SRC_DIR)/NPME_*.cpp)
LIB_OBJ := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(LIB_SRC))

# Application source/object/executables
APP_SRC := $(wildcard $(APP_DIR)/*.cpp)
APP_OBJ := $(patsubst $(APP_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(APP_SRC))
APP_NAMES := npme makeRandomBox laplaceDM RalphaDM helmholtzDM userDefineFunc compareV
APP_EXE := $(patsubst %,$(EXE_DIR)/%,$(APP_NAMES))

# Build all
all: $(APP_EXE)

# Linking rules for each app
$(EXE_DIR)/%: $(OBJ_DIR)/%.o $(LIB_OBJ)
	$(CPP) $(CFLAGS) -o $@ $^ -lm

# Compile rules
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CPP) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/%.o: $(APP_DIR)/%.cpp
	$(CPP) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Cleanup
clean:
	rm -f $(OBJ_DIR)/*.o $(EXE_DIR)/*

