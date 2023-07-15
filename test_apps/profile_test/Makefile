GPP := g++-11

SDL2 := `sdl2-config --cflags --libs`
LIBUSB := `pkg-config --libs --cflags libusb-1.0` -ljpeg

build := ./build_files

app := ../app
tests_src := ./tests_src

simage := ../../simage
stb := $(simage)/libs/stb
uvc := $(simage)/libs/uvc
src := $(simage)/src
impl := $(src)/impl
util := $(src)/util

exe := $(build)/test_app

simage_h := $(simage)/simage.hpp
simage_h += $(util)/types.hpp
simage_h += $(simage)/defines.hpp

tests_include_h := ../tests_include.hpp



# main
main_dep := $(app_h)
main_dep += $(tests_include_h)

main_c := $(tests_src)/profile_test_main.cpp
main_o := $(build)/main.o
object_files := $(main_o)


# simage
simage_dep := $(simage_h)

simage_dep += $(impl)/channel_pixels.cpp
simage_dep += $(impl)/platform_image.cpp
simage_dep += $(impl)/row_begin.cpp
simage_dep += $(impl)/make_view.cpp
simage_dep += $(impl)/sub_view.cpp
simage_dep += $(impl)/fill.cpp
simage_dep += $(impl)/convolve.cpp
simage_dep += $(impl)/gradients.cpp
simage_dep += $(impl)/blur.cpp
simage_dep += $(impl)/rotate.cpp
simage_dep += $(impl)/split_channels.cpp
simage_dep += $(impl)/copy.cpp
simage_dep += $(impl)/map_channels.cpp
simage_dep += $(impl)/alpha_blend.cpp
simage_dep += $(impl)/for_each_pixel.cpp
simage_dep += $(impl)/transform.cpp
simage_dep += $(impl)/centroid.cpp
simage_dep += $(impl)/skeleton.cpp
simage_dep += $(impl)/histogram.cpp
simage_dep += $(impl)/select_channel.cpp
simage_dep += $(stb)/stb_image_resize.h
simage_dep += $(stb)/stb_image_write.h
simage_dep += $(stb)/stb_image.h
simage_dep += $(stb)/stb_include.hpp
simage_dep += $(stb)/stb_simage.cpp
simage_dep += $(uvc)/libuvc2.hpp
simage_dep += $(uvc)/uvc_simage.cpp
simage_dep += $(util)/color_space.hpp
simage_dep += $(util)/execute.hpp
simage_dep += $(util)/execute.cpp

simage_c := $(simage)/simage.cpp
simage_o := $(build)/simage.o
object_files += $(simage_o)


# tests
tests_dep := $(tests_include_h)
tests_dep += $(util)/profiler.hpp
tests_dep += $(util)/profiler.cpp

tests_c := $(tests_src)/profile_tests.cpp
tests_o := $(build)/tests.o
object_files += $(tests_o)


# temp overwrite
#main_c := $(tests_src)/temp_main.cpp
#tests_c := $(tests_src)/temp_tests.cpp


# flags
INCLUDES  := 
#INCLUDES += -I /usr/include/libusb-1.0

LIBRARIES := -ltbb -pthread
LIBRARIES += $(LIBUSB)
#LIBRARIES += $(SDL2)


CCFLAGS := -std=c++17
CCFLAGS += -DSIMAGE_NO_CUDA
CCFLAGS += -DNDEBUG
CCFLAGS += -O3

# build rules

$(main_o): $(main_c) $(main_dep)
	@echo "\n main"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)


$(simage_o): $(simage_c) $(simage_dep)
	@echo "\n simage"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)


$(tests_o): $(tests_c) $(tests_dep)
	@echo "\n tests"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)


$(exe): $(object_files)
	@echo "\n exe"
	$(GPP) $(CCFLAGS) -o $@ $+ $(LIBRARIES)


build: $(exe)

run: build
	$(exe)

clean:
	rm -rfv $(build)/*

setup:
	mkdir -p $(build)