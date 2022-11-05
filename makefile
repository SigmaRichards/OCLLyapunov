TARGET="DO-THE-ROAR"
LINK_IM2=`pkg-config --cflags opencv4`
LINK_TAR=-lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui

$(TARGET): src/main.cpp src/logmap_kernel.cl
	g++ -o $(TARGET) src/main.cpp -lOpenCL $(LINK_IM2) $(LINK_TAR)

clean:
	rm *.o $(TARGET)
