CPUCPP    = gcc -pg
CPPFLAGS  = -O3
DEFS      =  
INCLUDES  = -I.
LIBDIR    = -L/usr/lib 
LIBS      =  
LINK      =  $(LIBDIR) $(LIBS)
CPU_COMPILE_CPP = $(CPUCPP) $(DEFS) $(INCLUDES) $(CPPFLAGS)


all:
	$(CPU_COMPILE_CPP)  bucketsort.c  $(LINK) -o bucketsort

clean:
	rm bucketsort

profiling:
	$(CPU_COMPILE_CPP)  bucketsort.c  $(LINK) -o bucketsort
	./bucketsort < message.txt
#Profiling file
	gprof ./bucketsort gmon.out > BenchMark_serial.txt

#profiling dot-scheme
	gprof2dot BenchMark_serial.txt > BenchMark_serial.dot
	dot -Tpng -o profile.png BenchMark_serial.dot 