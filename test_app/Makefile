GPP := g++-11

SDL2 := `sdl2-config --cflags --libs`

build := ./build_files

app := ./app
src := ./src

exe := $(build)/test_app


app_h := $(app)/app.hpp



# main
main_dep := $(app_h)

main_c := $(src)/test_app_main.cpp
main_o := $(build)/main.o
object_files := $(main_o)


# app
app_dep := $(app_h)
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


# build rules

$(app_o): $(app_c) $(app_dep)
	@echo "\n app"
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