CPUCPP    = gcc
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
