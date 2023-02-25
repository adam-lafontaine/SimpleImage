// D-Bus not build with -rdynamic...
// sudo killall ibus-daemon


#include "sdl_include.hpp"
#include "../app/app.hpp"
#include "../util/stopwatch.hpp"

#include <thread>


// control the framerate of the application
constexpr r32 TARGET_FRAMERATE_HZ = 60.0f;
constexpr r32 TARGET_MS_PER_FRAME = 1000.0f / TARGET_FRAMERATE_HZ;

static bool g_running = false;

ScreenMemory g_screen;
Input g_input[2];
SDLControllerInput g_controller_input = {};

char WINDOW_TITLE[50] = { 0 };


static void set_app_screen_buffer(ScreenMemory const& screen, img::View& app_screen)
{
    app_screen.width = screen.image_width;
    app_screen.height = screen.image_height;
    app_screen.matrix_width = screen.image_width;
    app_screen.matrix_data_ = (img::Pixel*)screen.image_data;
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
            print_message("SDL_QUIT\n");
            g_running = false;
        } break;
        case SDL_KEYDOWN:
        case SDL_KEYUP:
        {
            auto key_code = event.key.keysym.sym;
            auto alt = event.key.keysym.mod & KMOD_ALT;
            if(key_code == SDLK_F4 && alt)
            {
                print_message("ALT F4\n");
                g_running = false;
            }
            else if(key_code == SDLK_ESCAPE)
            {
                print_message("ESC\n");
                g_running = false;
            }

        } break;
        
    }
}


bool render_init(app::WindowSettings const& window_settings, app::AppState& app_state)
{
    snprintf(WINDOW_TITLE, 50, "%s v%s", window_settings.app_title, window_settings.version);

    if (!init_sdl())
    {
        app_state.signal_stop = true;
        return false;
    }

    if (!create_screen_memory(g_screen, WINDOW_TITLE, (int)window_settings.screen_width, (int)window_settings.screen_height))
    {
        app_state.signal_stop = true;
        return false;
    }

    open_game_controllers(g_controller_input, g_input[0]);
    g_input[1].num_controllers = g_input[0].num_controllers;

    app_state.dgb.n_controllers = g_input[0].num_controllers;

    set_app_screen_buffer(g_screen, app_state.screen_pixels);

    return true;
}


void render_run(app::AppState& app_state, std::function<void(Input const&)> const& on_input)
{

    auto const cleanup = [&]()
    {
        app_state.signal_stop = true;
        close_game_controllers(g_controller_input, g_input[0]);
        destroy_screen_memory(g_screen);
        close_sdl();
    };    

    g_running = true;      
    
    int in_current = 0;
    int in_old = 1;
    Stopwatch sw;
    r64 frame_ms_elapsed = TARGET_MS_PER_FRAME;
    char dbg_title[80] = { 0 };
    r64 ms_elapsed = 0.0;
    r64 title_refresh_ms = 500.0;    

    auto const wait_for_framerate = [&]()
    {
        frame_ms_elapsed = sw.get_time_milli();

        if(ms_elapsed >= title_refresh_ms)
        {
            ms_elapsed = 0.0;
            #ifndef NDEBUG
            snprintf(dbg_title, 80, "%s (%d)", WINDOW_TITLE, (int)frame_ms_elapsed);
            SDL_SetWindowTitle(g_screen.window, dbg_title);
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
    while(g_running && !app_state.signal_stop)
    {
        SDLEventInfo evt{};
        evt.first_in_queue = true;
        evt.has_event = false;

        while (SDL_PollEvent(&evt.event))
        {
            evt.has_event = true;
            handle_sdl_event(evt.event);
            process_keyboard_input(evt, g_input[in_old].keyboard, g_input[in_current].keyboard);
            process_mouse_input(evt, g_input[in_old].mouse, g_input[in_current].mouse);
            evt.first_in_queue = false;
        }

        if (!evt.has_event)
        {
            process_keyboard_input(evt, g_input[in_old].keyboard, g_input[in_current].keyboard);
            process_mouse_input(evt, g_input[in_old].mouse, g_input[in_current].mouse);
        }

        process_controller_input(g_controller_input, g_input[in_old], g_input[in_current]);

        // does not miss frames but slows animation
        g_input[in_current].dt_frame = TARGET_MS_PER_FRAME / 1000.0f;

        // animation speed maintained but frames missed
        //g_input[in_current].dt_frame = frame_ms_elapsed / 1000.0f; // TODO:

        on_input(g_input[in_current]);

        wait_for_framerate();
        render_screen(g_screen);

        // swap inputs
        in_current = in_current ? 0 : 1;
        in_old = in_current ? 0 : 1;
    }

    cleanup();
}