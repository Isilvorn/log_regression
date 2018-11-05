all: logr

logr: temp/logr.o temp/dvect.o
	g++ -std=c++11 -g temp/logr.o temp/dvect.o
	mv a.out logr

temp/logr.o: src/main.cpp
	g++ -std=c++11 -c src/main.cpp
	mv main.o temp/logr.o

temp/dvect.o: src/dvect.cpp
	g++ -std=c++11 -c src/dvect.cpp
	mv dvect.o temp/dvect.o

clean:
	rm -f *~
	rm temp/*.o
