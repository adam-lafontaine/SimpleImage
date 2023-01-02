#pragma once

#include "../simage/simage_platform.hpp"


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


    class AppSettings
    {
    public:
        img::View screen_pixels;

        bool signal_stop = false;

        DebugInfo dgb;
    };
}


bool render_init(app::WindowSettings const& window_settings, app::AppSettings& app_settings);

void render_run(app::AppSettings& app_settings);