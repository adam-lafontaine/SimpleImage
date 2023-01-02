// D-Bus not build with -rdynamic...
// sudo killall ibus-daemon

#include "sdl_include.hpp"
#include "../simage/simage_platform.hpp"
#include "../util/stopwatch.hpp"

namespace img = simage;

#include <cstdio>
#include <thread>


namespace app
{
    constexpr auto APP_TITLE = "simage render";
    constexpr auto VERSION = "1.0";


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
        int placeholder;
    };


    class AppSettings
    {
    public:
        img::View screen_pixels;

        bool signal_stop = false;

        DebugInfo dgb;
    };
}


constexpr int WINDOW_WIDTH = 600;
constexpr int WINDOW_HEIGHT = 800;

// control the framerate of the application
constexpr r32 TARGET_FRAMERATE_HZ = 60.0f;
constexpr r32 TARGET_MS_PER_FRAME = 1000.0f / TARGET_FRAMERATE_HZ;

static bool g_running = false;


static void set_app_screen_buffer(ScreenMemory const& screen, img::View& app_screen)
{
    app_screen.width = screen.image_width;
    app_screen.height = screen.image_height;
    app_screen.image_width = screen.image_width;
    app_screen.image_data = (img::Pixel*)screen.image_data;
    app_screen.range = make_range(app_screen);
}


static void handle_sdl_event(SDL_Event const& event)
{
    switch(event.type)
    {
        case SDL_WINDOWEVENT:
        {
            handle_sdl_window_event(event.window);
        } break;
        case SDL_QUIT:
        {
            printf("SDL_QUIT\n");
            g_running = false;
        } break;
        case SDL_KEYDOWN:
        case SDL_KEYUP:
        {
            auto key_code = event.key.keysym.sym;
            auto alt = event.key.keysym.mod & KMOD_ALT;
            if(key_code == SDLK_F4 && alt)
            {
                printf("ALT F4\n");
                g_running = false;
            }
            else if(key_code == SDLK_ESCAPE)
            {
                printf("ESC\n");
                g_running = false;
            }

        } break;
        
    }
}


bool render_run(app::WindowSettings const& window_settings, app::AppSettings& app_settings)
{
    char WINDOW_TITLE[50] = { 0 };
    snprintf(WINDOW_TITLE, 50, "%s v%s", window_settings.app_title, window_settings.version);

    printf("\n%s\n", WINDOW_TITLE);
    if(!init_sdl())
    {        
        return false;
    }

    ScreenMemory screen{};
    if(!create_screen_memory(screen, WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT))
    {
        return false;
    }

    Input input[2] = {};
    SDLControllerInput controller_input = {};

    auto const cleanup = [&]()
    {
        app_settings.signal_stop = true;
        close_game_controllers(controller_input, input[0]);
        destroy_screen_memory(screen);
        close_sdl();
    };

    open_game_controllers(controller_input, input[0]);
    input[1].num_controllers = input[0].num_controllers;
    printf("controllers = %d\n", input[0].num_controllers);

    set_app_screen_buffer(screen, app_settings.screen_pixels);

    g_running = true;      
    
    bool in_current = 0;
    bool in_old = 1;
    Stopwatch sw;
    r64 frame_ms_elapsed = TARGET_MS_PER_FRAME;
    char dbg_title[50] = { 0 };
    r64 ms_elapsed = 0.0;
    r64 title_refresh_ms = 500.0;

    auto const wait_for_framerate = [&]()
    {
        frame_ms_elapsed = sw.get_time_milli();

        if(ms_elapsed >= title_refresh_ms)
        {
            ms_elapsed = 0.0;
            #ifndef NDEBUG
            snprintf(dbg_title, 50, "%s (%d)", WINDOW_TITLE, (int)frame_ms_elapsed);
            SDL_SetWindowTitle(screen.window, dbg_title);
            #endif
        }

        auto sleep_ms = (u32)(TARGET_MS_PER_FRAME - frame_ms_elapsed);
        if (frame_ms_elapsed < TARGET_MS_PER_FRAME && sleep_ms > 0)
        { 
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
            while (frame_ms_elapsed < TARGET_MS_PER_FRAME)
            {
                frame_ms_elapsed = sw.get_time_milli();
            }        
        }

        ms_elapsed += frame_ms_elapsed;        

        sw.start();
    };
    
    sw.start();
    while(g_running)
    {
        SDLEventInfo evt{};
        evt.first_in_queue = true;
        evt.has_event = false;

        while (SDL_PollEvent(&evt.event))
        {
            evt.has_event = true;
            handle_sdl_event(evt.event);
            process_keyboard_input(evt, input[in_old].keyboard, input[in_current].keyboard);
            process_mouse_input(evt, input[in_old].mouse, input[in_current].mouse);
            evt.first_in_queue = false;
        }

        if (!evt.has_event)
        {
            process_keyboard_input(evt, input[in_old].keyboard, input[in_current].keyboard);
            process_mouse_input(evt, input[in_old].mouse, input[in_current].mouse);
        }

        process_controller_input(controller_input, input[in_old], input[in_current]);

        // does not miss frames but slows animation
        input[in_current].dt_frame = TARGET_MS_PER_FRAME / 1000.0f;

        // animation speed maintained but frames missed
        //input[in_current].dt_frame = frame_ms_elapsed / 1000.0f; // TODO:  

        wait_for_framerate();
        render_screen(screen);

        // swap inputs
        in_current = in_old;
        in_old = !in_old;        
    }

    cleanup();

    return true;
}