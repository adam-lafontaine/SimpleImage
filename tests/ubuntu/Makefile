GPP := g++-11


build := ./build_files
tests := ..
src := ../../src
util := $(src)/util


tests_exe := $(build)/tests_exe


# main
main_c        := $(tests)/tests_main.cpp
main_o        := $(build)/tests_main.o
object_files  := $(main_o)


defines_h := $(src)/defines.hpp

stopwatch_h := $(util)/stopwatch.hpp

tests_include_h := $(tests)/tests_include.hpp

execute_h := $(util)/execute.hpp
execute_h += $(defines_h)


execute_c := $(util)/execute.cpp
execute_o := $(build)/execute.o
object_files += $(execute_o)


execute_tests_c := $(tests)/execute_tests.cpp
execute_tests_o := $(build)/execute_tests.o
object_files += $(execute_tests_o)


# flags
INCLUDES  :=
LIBRARIES := -ltbb
CCFLAGS := -std=c++17

CCFLAGS += -O3
#CCFLAGS += -DNDEBUG


# build rules

$(main_o): $(main_c) $(tests_include_h)
	@echo "\n main"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)


$(execute_o): $(execute_c) $(execute_h)
	@echo "\n execute"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)
	

$(execute_tests_o): $(execute_tests_c) $(execute_h) $(stopwatch_h)
	@echo "\n execute_tests"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)

$(tests_exe): $(object_files)
	@echo "\n tests_exe"
	$(GPP) $(CCFLAGS) -o $@ $+ $(LIBRARIES)


build: $(tests_exe)

run: build
	$(tests_exe)

clean:
	rm -rfv $(build)/*

setup:
	mkdir -p $(build)