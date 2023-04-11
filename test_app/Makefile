GPP := g++-11

SDL2 := `sdl2-config --cflags --libs`
LIBUSB := `pkg-config --libs --cflags libusb-1.0` -ljpeg

build := ./build_files

app := ./app
tests_src := ./tests_src

simage := ../simage
stb := $(simage)/libs/stb
uvc := $(simage)/libs/uvc
src := $(simage)/src
util := $(src)/util

exe := $(build)/test_app

simage_h := $(simage)/simage.hpp
app_h := $(app)/app.hpp
tests_h := $(tests_src)/tests_include.hpp



# main
main_dep := $(app_h)
main_dep += $(tests_h)

main_c := $(tests_src)/test_app_main.cpp
main_o := $(build)/main.o
object_files := $(main_o)


# app
app_dep := $(app_h)
app_dep += $(simage_h)
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
simage_dep += $(src)/simage_impl.cpp
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
tests_dep := $(tests_h)
tests_dep += $(tests_src)/fill_tests.cpp
tests_dep += $(tests_src)/copy_tests.cpp
tests_dep += $(tests_src)/resize_image_tests.cpp
tests_dep += $(tests_src)/split_channels_tests.cpp
tests_dep += $(tests_src)/alpha_blend_tests.cpp

tests_c := $(tests_src)/tests.cpp
tests_o := $(build)/tests.o
object_files += $(tests_o)


# flags
INCLUDES  := 
#INCLUDES += -I /usr/include/libusb-1.0

LIBRARIES := $(SDL2) -ltbb -pthread
LIBRARIES += $(LIBUSB)

CCFLAGS := -std=c++20 

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