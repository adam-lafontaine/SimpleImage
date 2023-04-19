#pragma once

#include "../../simage/simage.hpp"
#include "input.hpp"


#include <functional>


namespace img = simage;


namespace app
{
    class WindowSettings
    {
    public:
        const char* app_title;
        const char* version;

        u32 screen_width;
        u32 screen_height;
    };


    class DebugInfo
    {
    public:
        u32 n_controllers = 0;
    };


    class AppState
    {
    public:

        bool signal_stop = false;

        DebugInfo dgb;

        img::View screen_buffer[2];
        int read_index = 0;
    };
}



// sdl_render_run.cpp

bool render_init(app::WindowSettings const& window_settings, app::AppState& app_settings);

void render_close();

void render_once();

void render_run(app::AppState& app_state, std::function<void(Input const&)> const& on_input);