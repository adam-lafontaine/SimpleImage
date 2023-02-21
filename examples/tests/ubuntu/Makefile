GPP := g++-11


build := ./build_files
out := ../TestImages/out_files

tests  := ..
src    := ../../src
util   := $(src)/util
simage := $(src)/simage
stb    := $(src)/stb


tests_exe := $(build)/tests_exe

# headers
defines_h := $(src)/defines.hpp
stopwatch_h := $(util)/stopwatch.hpp

# execute
execute_h := $(util)/execute.hpp
execute_h += $(defines_h)

# memory buffer
memory_buffer_h := $(util)/memory_buffer.hpp
memory_buffer_h += $(defines_h)

# color_space
color_space_h := $(util)/color_space.hpp
color_space_h += $(defines_h)

# stb
stb_h := $(stb)/stb_include.hpp
stb_h += $(stb)/stb_image_resize.h
stb_h += $(stb)/stb_image_write.h
stb_h += $(stb)/stb_image.h
stb_h += $(defines_h)

# simage
simage_platform_h := $(simage)/simage_platform.hpp
simage_platform_h += $(defines_h)

simage_h := $(simage)/simage.hpp
simage_h += $(simage_platform_h)
simage_h += $(memory_buffer_h)

#tests
tests_include_h := $(tests)/tests_include.hpp
tests_include_h += $(simage_h)


# main
main_c := $(tests)/tests_main.cpp
main_c += $(tests_include_h)
main_o := $(build)/tests_main.o
object_files := $(main_o)


# execute
execute_c := $(util)/execute.cpp
execute_c += $(execute_h)
execute_o := $(build)/execute.o
object_files += $(execute_o)


# memory buffer
memory_buffer_c := $(util)/memory_buffer.cpp
memory_buffer_c += $(memory_buffer_h)
memory_buffer_o := $(build)/memory_buffer.o
object_files += $(memory_buffer_o)


# stb_simage
stb_simage_c := $(stb)/stb_simage.cpp
stb_simage_c += $(stb_h)
stb_simage_c += $(simage_platform_h)
stb_simage_o := $(build)/stb_simage.o
object_files += $(stb_simage_o)


# simage_platform
simage_platform_c := $(simage)/simage_platform.cpp
simage_platform_c += $(simage_platform_h)
simage_platform_c += $(execute_h)
simage_platform_c += $(color_space_h)
simage_platform_o := $(build)/simage_platform.o
object_files += $(simage_platform_o)


# simage
simage_c := $(simage)/simage.cpp
simage_c += $(simage_h)
simage_c += $(memory_buffer_h)
simage_c += $(execute_h)
simage_c += $(color_space_h)
simage_o := $(build)/simage.o
object_files += $(simage_o)


# tests
execute_tests_c := $(tests)/execute_tests.cpp
execute_tests_c += $(execute_h)
execute_tests_c += $(stopwatch_h)
execute_tests_o := $(build)/execute_tests.o
object_files += $(execute_tests_o)

memory_buffer_tests_c := $(tests)/memory_buffer_tests.cpp
memory_buffer_tests_c += $(memory_buffer_h);
memory_buffer_tests_o := $(build)/memory_buffer_tests.o
object_files += $(memory_buffer_tests_o)

create_image_tests_c := $(tests)/create_image_tests.cpp
create_image_tests_c += $(tests_include_h)
create_image_tests_o := $(build)/create_image_tests.o
object_files += $(create_image_tests_o)

make_view_tests_c := $(tests)/make_view_tests.cpp
make_view_tests_c += $(tests_include_h)
make_view_tests_o := $(build)/make_view_tests.o
object_files += $(make_view_tests_o)

stb_simage_tests_c := $(tests)/stb_simage_tests.cpp
stb_simage_tests_c += $(tests_include_h)
stb_simage_tests_o := $(build)/stb_simage_tests.o
object_files += $(stb_simage_tests_o)

map_tests_c := $(tests)/map_tests.cpp
map_tests_c += $(tests_include_h)
map_tests_o := $(build)/map_tests.o
object_files += $(map_tests_o)

map_rgb_tests_c := $(tests)/map_rgb_tests.cpp
map_rgb_tests_c += $(tests_include_h)
map_rgb_tests_o := $(build)/map_rgb_tests.o
object_files += $(map_rgb_tests_o)

map_rgb_hsv_tests_c := $(tests)/map_rgb_hsv_tests.cpp
map_rgb_hsv_tests_c += $(tests_include_h)
map_rgb_hsv_tests_c += $(color_space_h)
map_rgb_hsv_tests_c += $(execute_h)
map_rgb_hsv_tests_o := $(build)/map_rgb_hsv_tests.o
object_files += $(map_rgb_hsv_tests_o)

map_rgb_lch_tests_c := $(tests)/map_rgb_lch_tests.cpp
map_rgb_lch_tests_c += $(tests_include_h)
map_rgb_lch_tests_c += $(color_space_h)
map_rgb_lch_tests_c += $(execute_h)
map_rgb_lch_tests_o := $(build)/map_rgb_lch_tests.o
object_files += $(map_rgb_lch_tests_o)

map_rgb_yuv_tests_c := $(tests)/map_rgb_yuv_tests.cpp
map_rgb_yuv_tests_c += $(tests_include_h)
map_rgb_yuv_tests_c += $(color_space_h)
map_rgb_yuv_tests_c += $(execute_h)
map_rgb_yuv_tests_o := $(build)/map_rgb_yuv_tests.o
object_files += $(map_rgb_yuv_tests_o)

sub_view_tests_c := $(tests)/sub_view_tests.cpp
sub_view_tests_c += $(tests_include_h)
sub_view_tests_o := $(build)/sub_view_tests.o
object_files += $(sub_view_tests_o)

copy_tests_c := $(tests)/copy_tests.cpp
copy_tests_c += $(tests_include_h)
copy_tests_o := $(build)/copy_tests.o
object_files += $(copy_tests_o)

fill_tests_c := $(tests)/fill_tests.cpp
fill_tests_c += $(tests_include_h)
fill_tests_o := $(build)/fill_tests.o
object_files += $(fill_tests_o)

histogram_tests_c := $(tests)/histogram_tests.cpp
histogram_tests_c += $(tests_include_h)
histogram_tests_o := $(build)/histogram_tests.o
object_files += $(histogram_tests_o)

gradients_tests_c := $(tests)/gradients_tests.cpp
gradients_tests_c += $(tests_include_h)
gradients_tests_o := $(build)/gradients_tests.o
object_files += $(gradients_tests_o)

blur_tests_c := $(tests)/blur_tests.cpp
blur_tests_c += $(tests_include_h)
blur_tests_o := $(build)/blur_tests.o
object_files += $(blur_tests_o)



# flags
INCLUDES  :=
LIBRARIES := -ltbb
CCFLAGS := -std=c++20

CCFLAGS += -O3
CCFLAGS += -DNDEBUG


# build rules

$(main_o): $(main_c)
	@echo "\n main"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)


$(execute_o): $(execute_c)
	@echo "\n execute"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)

$(memory_buffer_o): $(memory_buffer_c)
	@echo "\n memory_buffer"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)


$(stb_simage_o): $(stb_simage_c)
	@echo "\n stb_simage"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)

$(simage_platform_o): $(simage_platform_c)
	@echo "\n simage_platform"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)

$(simage_o): $(simage_c)
	@echo "\n simage"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)


$(execute_tests_o): $(execute_tests_c)
	@echo "\n execute_tests"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)

$(memory_buffer_tests_o): $(memory_buffer_tests_c)
	@echo "\n memory_buffer_tests"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)

$(stb_simage_tests_o): $(stb_simage_tests_c)
	@echo "\n stb_simage_tests"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)

$(create_image_tests_o): $(create_image_tests_c)
	@echo "\n create_image_tests"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)

$(make_view_tests_o): $(make_view_tests_c)
	@echo "\n make_view_tests"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)

$(map_tests_o): $(map_tests_c)
	@echo "\n map_tests"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)

$(map_rgb_tests_o): $(map_rgb_tests_c)
	@echo "\n map_rgb_tests"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)

$(map_rgb_hsv_tests_o): $(map_rgb_hsv_tests_c)
	@echo "\n map_rgb_hsv_tests"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)

$(map_rgb_lch_tests_o): $(map_rgb_lch_tests_c)
	@echo "\n map_rgb_lch_tests"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)

$(map_rgb_yuv_tests_o): $(map_rgb_yuv_tests_c)
	@echo "\n map_rgb_yuv_tests"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)

$(sub_view_tests_o): $(sub_view_tests_c)
	@echo "\n sub_view_tests"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)

$(copy_tests_o): $(copy_tests_c)
	@echo "\n copy_tests"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)

$(fill_tests_o): $(fill_tests_c)
	@echo "\n fill_tests"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)

$(histogram_tests_o): $(histogram_tests_c)
	@echo "\n histogram_tests"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)

$(gradients_tests_o): $(gradients_tests_c)
	@echo "\n gradients_tests"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)

$(blur_tests_o): $(blur_tests_c)
	@echo "\n blur_tests"
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