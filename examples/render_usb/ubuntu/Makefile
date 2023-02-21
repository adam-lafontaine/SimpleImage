GPP := g++-11

SDL2 := `sdl2-config --cflags --libs`
LIBUSB := `pkg-config --libs --cflags libusb-1.0` -ljpeg

build := ./build_files

procs  := ..
src    := ../../../src
util   := $(src)/util
app    := $(src)/app
input  := $(src)/input
sdl    := $(src)/sdl
simage := $(src)/simage
stb    := $(src)/stb
opencv := $(src)/opencv
uvc    := $(src)/uvc
res    := $(src)/resources

render_exe := $(build)/render_exe


# resources
icon_h := $(res)/icon_64.h

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

# uvc
uvc_h := $(uvc)/libuvc.h

# simage
simage_platform_h := $(simage)/simage_platform.hpp
simage_platform_h += $(defines_h)

simage_h := $(simage)/simage.hpp
simage_h += $(simage_platform_h)
simage_h += $(memory_buffer_h)

# input
input_h := $(input)/input.hpp

input_state_h := $(input)/input_state.hpp
input_state += $(input_h)

# sdl
sdl_include_h := $(sdl)/sdl_include.hpp
sdl_include_h += $(input_h)
sdl_include_h += $(icon_h)

# app
app_h := $(app)/app.hpp
app_h += $(simage_platform_h)
app_h += $(input_h)

# procs
proc_def_h := $(procs)/proc_def.hpp
proc_def_h += $(simage_platform_h)


# main
main_c := $(procs)/render_usb_main.cpp
main_c += $(proc_def_h)
main_o := $(build)/render_usb_main.o
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


# uvc_simage
uvc_simage_c := $(uvc)/uvc_simage.cpp
uvc_simage_c += $(simage_platform_h)
uvc_simage_c += $(uvc_h)
uvc_simage_o := $(build)/uvc_simage.o
object_files += $(uvc_simage_o)


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


# sdl_input
sdl_input_c := $(sdl)/sdl_input.cpp
sdl_input_c += $(sdl_include_h)
sdl_input_c += $(input_state_h)
sdl_input_o := $(build)/sdl_input.o
object_files += $(sdl_input_o)


# sdl_render
sdl_render_c := $(sdl)/sdl_render_run.cpp
sdl_render_c += $(sdl_include_h)
sdl_render_c += $(app_h)
sdl_render_c += $(stopwatch_h)
sdl_render_o := $(build)/sdl_render.o
object_files += $(sdl_render_o)


# render_usb
render_usb_c := $(procs)/render_usb.cpp
render_usb_c += $(proc_def_h)
render_usb_c += $(simage_h)
render_usb_o := $(build)/render_usb.o
object_files += $(render_usb_o)


# flags
INCLUDES  := 
INCLUDES += -I /usr/include/libusb-1.0

LIBRARIES := $(SDL2) -ltbb -pthread
LIBRARIES += $(LIBUSB)

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

$(uvc_simage_o): $(uvc_simage_c)
	@echo "\n uvc_simage"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)


$(simage_platform_o): $(simage_platform_c)
	@echo "\n simage_platform"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)

$(simage_o): $(simage_c)
	@echo "\n simage"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)


$(sdl_input_o): $(sdl_input_c)
	@echo "\n sdl_input"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)

$(sdl_render_o): $(sdl_render_c)
	@echo "\n sdl_render"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)


$(render_usb_o): $(render_usb_c)
	@echo "\n render_usb"
	$(GPP) $(CCFLAGS) -o $@ -c $< $(LIBRARIES)


$(render_exe): $(object_files)
	@echo "\n render_exe"
	$(GPP) $(CCFLAGS) -o $@ $+ $(LIBRARIES)


build: $(render_exe)

run: build
	$(render_exe)

clean:
	rm -rfv $(build)/*

setup:
	mkdir -p $(build)