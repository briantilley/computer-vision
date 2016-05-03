# compiler
CPP=g++

all:
	$(CPP) driver.cpp -o a.out --std=gnu++11

clean:
	rm -rf *.o a.out