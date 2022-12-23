GPP := g++-11


build := ./build_files
out := ../TestImages/out_files

tests := ..
src := ../../src
util := $(src)/util
simage := $(src)/simage
stb := $(simage)/stb


tests_exe := $(build)/tests_exe

defines_h := $(src)/defines.hpp

stopwatch_h := $(util)/stopwatch.hpp


# main
main_c        := $(tests)/tests_main.cpp
main_o        := $(build)/tests_main.o
object_files  := $(main_o)


# execute
execute_h := $(util)/execute.hpp
execute_h += $(defines_h)

execute_c := $(util)/execute.cpp
execute_o := $(build)/execute.o
object_files += $(execute_o)


# stb
stb_h := $(stb)/stb_include.hpp
stb_h += $(stb)/stb_image_resize.h
stb_h += $(stb)/stb_image_write.h
stb_h += $(stb)/stb_image.h
stb_h += $(defines_h)

stb_simage_c := $(stb)/stb_simage.cpp
stb_simage_o := $(build)/stb_simage.object_files
object_files += $(stb_simage_o)


#simage
simage_platform_h := $(simage)/simage_platform.hpp
simage_platform_h += $(defines_h)

simage_h := $(simage)/simage.hpp
simage_h += $(simage_platform_h)

simage_c := $(simage)/simage.cpp
simage_o := $(build)/simage.o
object_files += $(simage_o)


# tests
tests_include_h := $(tests)/tests_include.hpp
tests_include_h += $(simage_h)

execute_tests_c := $(tests)/execute_tests.cpp
execute_tests_o := $(build)/execute_tests.o
object_files += $(execute_tests_o)

stb_simage_tests_c := $(tests)/stb_simage_tests.cpp
stb_simage_tests_o := $(build)/stb_simage_tests.o
object_files += $(stb_simage_tests_o)


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

$(stb_simage_o): $(stb_simage_c) $(stb_h) $(simage_platform_h)
	@echo "\n stb_simage"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)

$(simage_o): $(simage_c) $(simage_h)
	@echo "\n simage"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)



$(execute_tests_o): $(execute_tests_c) $(execute_h) $(stopwatch_h)
	@echo "\n execute_tests"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)

$(stb_simage_tests_o): $(stb_simage_tests_c) $(tests_include_h)
	@echo "\n stb_simage_tests"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)


$(tests_exe): $(object_files)
	@echo "\n tests_exe"
	$(GPP) $(CCFLAGS) -o $@ $+ $(LIBRARIES)


build: $(tests_exe)

run: build
	$(tests_exe)

clean:
	rm -rfv $(build)/*
	rm -rfv $(out)/*

setup:
	mkdir -p $(build)
	mkdir -p $(out)