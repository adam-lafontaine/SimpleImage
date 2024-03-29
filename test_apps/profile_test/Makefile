GPP := g++-11

SDL2 := `sdl2-config --cflags --libs`
LIBUSB := `pkg-config --libs --cflags libusb-1.0` -ljpeg

build := ./build_files

tests_src := ./tests_src

simage := ../../simage
stb := $(simage)/libs/stb
uvc := $(simage)/libs/uvc
src := $(simage)/src
cpp := $(src)/cpp
util := $(src)/util

exe := $(build)/test_app

simage_h := $(simage)/simage.hpp
simage_h += $(util)/types.hpp
simage_h += $(simage)/defines.hpp
simage_h += $(simage)/simage_types.hpp

tests_include_h := ../tests_include.hpp



# main
main_dep := $(app_h)
main_dep += $(tests_include_h)

main_c := $(tests_src)/profile_test_main.cpp
main_o := $(build)/main.o
object_files := $(main_o)


# simage
simage_dep := $(simage_h)

simage_dep += $(cpp)/alpha_blend.cpp
simage_dep += $(cpp)/blur.cpp
simage_dep += $(cpp)/centroid.cpp
simage_dep += $(cpp)/channel_pixels.cpp
simage_dep += $(cpp)/convolve.cpp
simage_dep += $(cpp)/copy.cpp
simage_dep += $(cpp)/fill.cpp
simage_dep += $(cpp)/for_each_pixel.cpp
simage_dep += $(cpp)/gradients.cpp
simage_dep += $(cpp)/histogram.cpp
simage_dep += $(cpp)/make_view.cpp
simage_dep += $(cpp)/map_color.cpp
simage_dep += $(cpp)/map_gray.cpp
simage_dep += $(cpp)/map_rgb.cpp
simage_dep += $(cpp)/platform_image.cpp
simage_dep += $(cpp)/rotate.cpp
simage_dep += $(cpp)/row_begin.cpp
simage_dep += $(cpp)/select_channel.cpp
simage_dep += $(cpp)/simd.cpp
simage_dep += $(cpp)/skeleton.cpp
simage_dep += $(cpp)/split_channels.cpp
simage_dep += $(cpp)/sub_view.cpp
simage_dep += $(cpp)/transform.cpp
simage_dep += $(cpp)/verify.cpp

simage_dep += $(stb)/stb_image_resize.h
simage_dep += $(stb)/stb_image_write.h
simage_dep += $(stb)/stb_image.h
simage_dep += $(stb)/stb_include.hpp
simage_dep += $(stb)/stb_simage.cpp
simage_dep += $(uvc)/libuvc2.hpp
simage_dep += $(uvc)/uvc_simage.cpp
simage_dep += $(util)/color_space.hpp
simage_dep += $(util)/memory_buffer.hpp

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

LIBRARIES := -pthread
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