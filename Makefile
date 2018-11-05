all: logr

logr: temp/logr.o
	g++ -std=c++11 -g temp/logr.o
	mv a.out logr

temp/logr.o: src/main.cpp
	g++ -std=c++11 -c src/main.cpp
	mv main.o temp/logr.o

clean:
	rm -f *~
	rm temp/*
