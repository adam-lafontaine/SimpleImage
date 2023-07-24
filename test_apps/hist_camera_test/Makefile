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
cpp := $(src)/cpp
util := $(src)/util

exe := $(build)/test_app

simage_h := $(simage)/simage.hpp
simage_h += $(util)/types.hpp
simage_h += $(simage)/defines.hpp
simage_h += $(simage)/simage_types.hpp

app_h := $(app)/app.hpp
tests_include_h := ../tests_include.hpp



# main
main_dep := $(app_h)
main_dep += $(tests_include_h)

main_c := $(tests_src)/hist_camera_test_main.cpp
main_o := $(build)/main.o
object_files := $(main_o)


# app
app_dep := $(app_h)
app_dep += $(simage_h)
app_dep += $(util)/types.hpp
app_dep += $(simage)/defines.hpp
app_dep += $(app)/icon_64.h
app_dep += $(app)/input_state.hpp
app_dep += $(app)/input.hpp
app_dep += $(app)/sdl_include.hpp
app_dep += $(app)/stopwatch.hpp
app_dep += $(app)/sdl_input.cpp
app_dep += $(app)/sdl_render_run.cpp

app_c := $(app)/app.cpp
app_o := $(build)/app.o
object_files += $(app_o)


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
simage_dep += $(cpp)/map_channels.cpp
simage_dep += $(cpp)/platform_image.cpp
simage_dep += $(cpp)/rotate.cpp
simage_dep += $(cpp)/row_begin.cpp
simage_dep += $(cpp)/select_channel.cpp
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
simage_dep += $(util)/execute.hpp
simage_dep += $(util)/execute.cpp
simage_dep += $(util)/memory_buffer.hpp

simage_c := $(simage)/simage.cpp
simage_o := $(build)/simage.o
object_files += $(simage_o)


# tests
tests_dep := $(tests_include_h)


tests_c := $(tests_src)/tests.cpp
tests_o := $(build)/tests.o
object_files += $(tests_o)


# flags
INCLUDES  := 
#INCLUDES += -I /usr/include/libusb-1.0

LIBRARIES := $(SDL2) -ltbb -pthread
LIBRARIES += $(LIBUSB)

CCFLAGS := -std=c++17
CCFLAGS += -DSIMAGE_NO_CUDA

#CCFLAGS += -O3 -DNDEBUG

# build rules

$(main_o): $(main_c) $(main_dep)
	@echo "\n main"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)


$(app_o): $(app_c) $(app_dep)
	@echo "\n app"
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