# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc
METAL_CC ?= clang
METAL_CXX ?= clang++
XCRUN ?= xcrun
LLAMA_CPP_SOURCE ?= $(shell sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' ../llama_cpp_convert_build/CMakeCache.txt 2>/dev/null)
LLAMA_CPP_ESCARDO_BUILD ?= ../llama_cpp_escardo_build

# the most basic way of building that is most likely to work on most systems
.PHONY: run
run: run.c
	$(CC) -O3 -o run run.c -lm
	$(CC) -O3 -o runq runq.c -lm

.PHONY: runescardo
runescardo: run_escardo.c atkey_term_c.c atkey_term_c.h run.c
	$(CC) -std=c11 -O3 -Wall -Wextra -Werror \
		-Wno-deprecated-declarations -Wno-sign-compare \
		-Wno-unused-variable run_escardo.c atkey_term_c.c \
		-lm -o run_escardo

.PHONY: runescardometal
runescardometal: run_escardo_metal

metal_kernels.air: metal_kernels.metal
	$(XCRUN) -sdk macosx metal -O3 -c metal_kernels.metal -o $@

metal_kernels.metallib: metal_kernels.air
	$(XCRUN) -sdk macosx metallib metal_kernels.air -o $@

run_escardo_metal_main.o: run_escardo.c atkey_term_c.h run.c
	$(METAL_CC) -std=c11 -O3 -Wall -Wextra -Werror \
		-Wno-deprecated-declarations -Wno-sign-compare \
		-Wno-unused-variable -DATKEY_METAL -c run_escardo.c -o $@

atkey_term_c_metal.o: atkey_term_c.c atkey_term_c.h metal_backend.h run.c
	$(METAL_CC) -std=c11 -O3 -Wall -Wextra -Werror \
		-Wno-deprecated-declarations -Wno-sign-compare \
		-Wno-unused-variable -DATKEY_METAL -c atkey_term_c.c -o $@

metal_backend.o: metal_backend.mm metal_backend.h
	$(METAL_CXX) -std=c++17 -O3 -Wall -Wextra -Werror \
		-Wno-deprecated-declarations -fobjc-arc -c metal_backend.mm -o $@

run_escardo_metal: run_escardo_metal_main.o atkey_term_c_metal.o \
		metal_backend.o metal_kernels.metallib
	$(METAL_CXX) run_escardo_metal_main.o atkey_term_c_metal.o \
		metal_backend.o -framework Foundation -framework Metal \
		-framework MetalPerformanceShaders -lm -o $@

.PHONY: runescardogguf
runescardogguf:
	cmake -S llama_cpp_escardo -B $(LLAMA_CPP_ESCARDO_BUILD) \
		-DLLAMA_CPP_SOURCE=$(LLAMA_CPP_SOURCE) \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_C_COMPILER=$(METAL_CC) \
		-DCMAKE_CXX_COMPILER=$(METAL_CXX)
	cmake --build $(LLAMA_CPP_ESCARDO_BUILD) \
		--target run_escardo_gguf -j 12

.PHONY: runhidden
runhidden: run_hidden_feedback.c
	$(CC) -O3 -o run_hidden_feedback run_hidden_feedback.c -lm

.PHONY: reversecompany
reversecompany: reverse_company_readout.c run.c
	$(CC) -std=c11 -O3 -Wall -Wextra -Werror \
		-Wno-sign-compare -Wno-unused-variable \
		reverse_company_readout.c -lm -o reverse_company_readout

.PHONY: runhiddenselect
runhiddenselect: run_hidden_feedback_select.c llama_company.c llama_company.h atkey_term_c.c atkey_term_c.h run.c
	$(CC) -std=c11 -O3 -Wall -Wextra -Werror \
		-Wno-sign-compare -Wno-unused-variable -Wno-unused-function \
		-o run_hidden_feedback_select run_hidden_feedback_select.c \
		llama_company.c atkey_term_c.c -lm

.PHONY: testcompany
testcompany: test_llama_company.c llama_company.c llama_company.h atkey_term_c.c atkey_term_c.h run.c
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror -DATKEY_REFERENCE_TEST_API \
		-c test_llama_company.c -o test_llama_company.o
	$(CC) -std=c11 -O2 -Wall -Wextra -Werror \
		-c llama_company.c -o llama_company.o
	$(CC) -std=c11 -O3 -Wno-deprecated-declarations \
		-DATKEY_REFERENCE_TEST_API -c atkey_term_c.c -o atkey_term_c_reference.o
	$(CC) test_llama_company.o llama_company.o atkey_term_c_reference.o \
		-lm -o test_llama_company
	./test_llama_company test/stories260K.bin test/tok512.bin

.PHONY: candidateprobe
candidateprobe: candidate_probe.c run.c
	$(CC) -std=c11 -O3 -Wall -Wextra -Werror \
		-Wno-sign-compare -Wno-unused-variable \
		candidate_probe.c -lm -o candidate_probe

.PHONY: candidatedump
candidatedump: candidate_dump.c candidate_probe.c run.c
	$(CC) -std=c11 -O3 -Wall -Wextra -Werror \
		-Wno-sign-compare -Wno-unused-variable -Wno-unused-function \
		candidate_dump.c -lm -o candidate_dump

.PHONY: forcedphraseprobe
forcedphraseprobe: forced_phrase_probe.c candidate_probe.c run.c
	$(CC) -std=c11 -O3 -Wall -Wextra -Werror \
		-Wno-sign-compare -Wno-unused-variable -Wno-unused-function \
		forced_phrase_probe.c -lm -o forced_phrase_probe

.PHONY: companyprobe
companyprobe: company_probe.c candidate_probe.c run.c
	$(CC) -std=c11 -O3 -Wall -Wextra -Werror \
		-Wno-sign-compare -Wno-unused-variable -Wno-unused-function \
		company_probe.c -lm -o company_probe

.PHONY: cpsfixedpoints
cpsfixedpoints: cps_fixed_points.c run.c
	$(CC) -std=c11 -O3 -Wall -Wextra -Werror \
		-Wno-sign-compare -Wno-unused-variable -Wno-unused-function \
		cps_fixed_points.c -lm -o cps_fixed_points

.PHONY: cpsaffinespectrum
cpsaffinespectrum: cps_affine_spectrum.c cps_fixed_points.c run.c
	$(CC) -std=c11 -O3 -Wall -Wextra -Werror \
		-Wno-sign-compare -Wno-unused-variable -Wno-unused-function \
		cps_affine_spectrum.c -lm -framework Accelerate \
		-o cps_affine_spectrum

.PHONY: cpspullbackspectrum
cpspullbackspectrum: cps_pullback_spectrum.c cps_affine_spectrum.c cps_fixed_points.c run.c
	$(CC) -std=c11 -O3 -Wall -Wextra -Werror \
		-Wno-sign-compare -Wno-unused-variable -Wno-unused-function \
		cps_pullback_spectrum.c -lm -framework Accelerate \
		-o cps_pullback_spectrum

.PHONY: cpsgrammaractions
cpsgrammaractions: cps_grammar_actions.c cps_fixed_points.c run.c
	$(CC) -std=c11 -O3 -Wall -Wextra -Werror \
		-Wno-sign-compare -Wno-unused-variable -Wno-unused-function \
		cps_grammar_actions.c -lm -o cps_grammar_actions

.PHONY: longcontextprofiles
longcontextprofiles: companyprobe
	python3 gather_company_traces.py

.PHONY: exhaustivescaleprobe
exhaustivescaleprobe: exhaustive_scale_probe.c candidate_probe.c run.c
	$(CC) -std=c11 -O3 -Wall -Wextra -Werror \
		-Wno-sign-compare -Wno-unused-variable -Wno-unused-function \
		exhaustive_scale_probe.c -lm -o exhaustive_scale_probe

.PHONY: scalerewardaudit
scalerewardaudit: scale_reward_audit.c exhaustive_scale_probe.c candidate_probe.c run.c
	$(CC) -std=c11 -O3 -Wall -Wextra -Werror \
		-Wno-sign-compare -Wno-unused-variable -Wno-unused-function \
		scale_reward_audit.c -lm -o scale_reward_audit

# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./run out/model.bin -n 3
rundebug: run.c
	$(CC) -g -o run run.c -lm
	$(CC) -g -o runq runq.c -lm

# https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
# https://simonbyrne.github.io/notes/fastmath/
# -Ofast enables all -O3 optimizations.
# Disregards strict standards compliance.
# It also enables optimizations that are not valid for all standard-compliant programs.
# It turns on -ffast-math, -fallow-store-data-races and the Fortran-specific
# -fstack-arrays, unless -fmax-stack-var-size is specified, and -fno-protect-parens.
# It turns off -fsemantic-interposition.
# In our specific application this is *probably* okay to use
.PHONY: runfast
runfast: run.c
	$(CC) -Ofast -o run run.c -lm
	$(CC) -Ofast -o runq runq.c -lm

# additionally compiles with OpenMP, allowing multithreaded runs
# make sure to also enable multiple threads when running, e.g.:
# OMP_NUM_THREADS=4 ./run out/model.bin
.PHONY: runomp
runomp: run.c
	$(CC) -Ofast -fopenmp -march=native run.c  -lm  -o run
	$(CC) -Ofast -fopenmp -march=native runq.c  -lm  -o runq

.PHONY: win64
win64:
	x86_64-w64-mingw32-gcc -Ofast -D_WIN32 -o run.exe -I. run.c win.c
	x86_64-w64-mingw32-gcc -Ofast -D_WIN32 -o runq.exe -I. runq.c win.c

# compiles with gnu99 standard flags for amazon linux, coreos, etc. compatibility
.PHONY: rungnu
rungnu:
	$(CC) -Ofast -std=gnu11 -o run run.c -lm
	$(CC) -Ofast -std=gnu11 -o runq runq.c -lm

.PHONY: runompgnu
runompgnu:
	$(CC) -Ofast -fopenmp -std=gnu11 run.c  -lm  -o run
	$(CC) -Ofast -fopenmp -std=gnu11 runq.c  -lm  -o runq

# run all tests
.PHONY: test
test:
	pytest

# run only tests for run.c C implementation (is a bit faster if only C code changed)
.PHONY: testc
testc:
	pytest -k runc

# run the C tests, without touching pytest / python
# to increase verbosity level run e.g. as `make testcc VERBOSITY=1`
VERBOSITY ?= 0
.PHONY: testcc
testcc:
	$(CC) -DVERBOSITY=$(VERBOSITY) -O3 -o testc test.c -lm
	./testc

.PHONY: clean
clean:
	rm -f run
	rm -f runq
	rm -f run_escardo
	rm -f run_escardo_metal
	rm -f run_escardo_metal_main.o
	rm -f atkey_term_c_metal.o
	rm -f metal_backend.o
	rm -f metal_kernels.air
	rm -f metal_kernels.metallib
	rm -f run_hidden_feedback
	rm -f reverse_company_readout
	rm -f run_hidden_feedback_select
	rm -f run_atkey_term
	rm -f run_atkey_term_strict
	rm -f candidate_probe
	rm -f candidate_dump
	rm -f forced_phrase_probe
	rm -f company_probe
	rm -f cps_fixed_points
	rm -f cps_affine_spectrum
	rm -f cps_pullback_spectrum
	rm -f cps_grammar_actions
	rm -f exhaustive_scale_probe
	rm -f scale_reward_audit
	rm -f atkey_term.o
	rm -f atkey_term_c.o
	rm -f atkey_term_c_reference.o
	rm -f llama_company.o
	rm -f test_llama_company.o
	rm -f test_llama_company
