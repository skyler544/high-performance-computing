CC = g++
OPENCL = -D CL_TARGET_OPENCL_VERSION=120
LINK = -lOpenCL
TARGET = vector-add
SRCS = main.cpp cl_utils.cpp

all: $(TARGET)

run: all
	./$(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(OPENCL) $(SRCS) -o $(TARGET) $(LINK)

clean:
	rm $(TARGET)
