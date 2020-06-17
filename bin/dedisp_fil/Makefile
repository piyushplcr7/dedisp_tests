DEDISP_HOME ?= $(HOME)/opt/dedisp
include Makefile.inc

dedisp_fil: dedisp_fil.c
	$(NVCC) -Xcompiler -Wall -O3 -I$(DEDISP_HOME)/include -o dedisp_fil dedisp_fil.c -L$(DEDISP_HOME)/lib -ldedisp -lm

clean:
	rm -f testdedisp
