#include "../simage/simage_platform.hpp"

#define LIBUVC_IMPLEMENTATION 1
#include "../uvc/libuvc2.hpp"

#include <vector>
#include <algorithm>

#ifndef NDEBUG
#include <cstdio>
#endif

//using namespace uvc;

namespace img = simage;

/*

libuvc requires RW permissions for opening capturing devices, so you must
create the following .rules file:

/etc/udev/rules.d/99-uvc.rules

Then, for each webcam add the following line:

SUBSYSTEMS=="usb", ENV{DEVTYPE}=="usb_device", ATTRS{idVendor}=="XXXX", ATTRS{idProduct}=="YYYY", MODE="0666"

Replace XXXX and YYYY for the 4 hexadecimal characters corresponding to the
vendor and product ID of your webcams.

*/

constexpr auto DEVICE_PERMISSION_MSG = 
"libuvc requires RW permissions for opening capturing devices, so you must create the following .rules file:"
"\n\n"
"/etc/udev/rules.d/99-uvc.rules"
"\n\n"
"Then, for each webcam add the following line:"
"\n\n"
"SUBSYSTEMS==\"usb\", ENV{DEVTYPE}==\"usb_device\", ATTRS{idVendor}==\"XXXX\", ATTRS{idProduct}==\"YYYY\", MODE=\"0666\"\n\n"
"Replace XXXX and YYYY for the 4 hexadecimal characters corresponding to the vendor and product ID of your webcams."
"\n\n"
"Restart the computer for the changes to take effect."
;

constexpr u8 EXPOSURE_MODE_AUTO = 2;
constexpr u8 EXPOSURE_MODE_APERTURE = 8;


/* verify */

#ifndef NDEBUG

namespace simage
{
	template <typename T>
	static bool verify(MatrixView<T> const& view)
	{
		return view.matrix_width && view.width && view.height && view.matrix_data;
	}


	static bool verify(CameraUSB const& camera)
	{
		return camera.image_width && camera.image_height && camera.max_fps && camera.device_id >= 0;
	}


	template <typename T>
	static bool verify(CameraUSB const& camera, MatrixView<T> const& view)
	{
		return verify(camera) && verify(view) &&
			camera.image_width == view.width &&
			camera.image_height == view.height;
	}
}

#endif





class DeviceUVC
{
public:
    uvc::device* p_device = nullptr;
    uvc::device_handle* h_device = nullptr;
    uvc::stream_ctrl* ctrl = nullptr;
    uvc::stream_handle* h_stream = nullptr;

    uvc::frame* rgb_frame = nullptr;
    img::ViewRGB rgb_view;

    int device_id = -1;

    int product_id = -1;
    int vendor_id = -1;
    
    int frame_width = -1;
    int frame_height = -1;
    int fps = -1;

    const char* format_code;

    bool is_connected = false;
    bool is_streaming = false;
};


class DeviceListUVC
{
public:
    uvc::context* context = nullptr;
    uvc::device** device_list = nullptr;

    std::vector<uvc::stream_ctrl> stream_ctrl_list;

    std::vector<DeviceUVC> devices;

    bool is_connected = false;
};


static DeviceListUVC g_device_list;


static void free_device_frame(DeviceUVC& device)
{
    if (device.rgb_frame)
    {
        uvc::uvc_free_frame(device.rgb_frame);
        device.rgb_frame = nullptr;
    }
}


static void print_uvc_error(uvc::error err, const char* msg)
{
#ifndef NDEBUG

    uvc::uvc_perror(err, msg);

#endif
}


static void print_error(const char* msg)
{
#ifndef NDEBUG

    printf("%s\n", msg);

#endif
}


static void print_device_permissions_msg()
{
#ifndef NDEBUG

printf("\n********** LINUX PERMISSIONS ERROR **********\n\n");

printf("%s", DEVICE_PERMISSION_MSG);

printf("\n\n********** LINUX PERMISSIONS ERROR **********\n\n");

#endif
}


static void print_device_error_info(DeviceListUVC const& list)
{
#ifndef NDEBUG

    if (list.devices.empty())
    {
        printf("No connected devices found\n");
        return;
    }

    for (auto const& dev : list.devices)
    {
        printf("Device %d:\n", dev.device_id);
        printf("  Vendor ID:   0x%04x\n", dev.vendor_id);
        printf("  Product ID:  0x%04x\n", dev.product_id);
    }

    print_device_permissions_msg();

#endif
}


static void print_device_list(DeviceListUVC const& list)
{
#ifndef NDEBUG

    printf("\nFound %u cameras\n", (u32)list.devices.size());
    for (auto const& cam : list.devices)
    {
        printf("Device %d:\n", cam.device_id);
        printf("  Vendor ID:   0x%04x\n", cam.vendor_id);
        printf("  Product ID:  0x%04x\n", cam.product_id);
        printf("  Status: %s\n", (cam.is_connected ? "OK" : "ERROR"));
        if (cam.is_connected)
        {
            printf("  Format:      (%4s) %dx%d %dfps\n", cam.format_code, cam.frame_width, cam.frame_height, cam.fps);
        }        
    }

#endif
}


static void print_uvc_stream_info(DeviceUVC const& device)
{
#ifndef NDEBUG
    //printf("\n");
    //uvc_print_stream_ctrl(device.ctrl, stdout);
#endif
}


static DeviceUVC* get_default_device(DeviceListUVC& list)
{
    auto& devices = list.devices;
    if(devices.empty())
    {
        return nullptr;
    }

    // return camera with highest index
    for (int id = (int)devices.size() - 1; id >= 0; --id)
    {
        if (devices[id].is_connected)
        {
            return devices.data() + id;
        }
    }

    return nullptr;
}


static void disconnect_device(DeviceUVC& device)
{
    if (device.is_connected)
    {
        uvc::uvc_close(device.h_device);
        device.h_device = nullptr;

        uvc::uvc_unref_device(device.p_device);
        device.p_device = nullptr;

        device.frame_width = -1;
        device.frame_height = -1;
        device.fps = -1;
        device.is_connected = false;        
    }
}


static bool connect_device(DeviceUVC& device, uvc::stream_ctrl* ctrl)
{
    device.ctrl = ctrl;

    auto res = uvc::uvc_open(device.p_device, &device.h_device);
    if (res != uvc::SUCCESS)
    {
        print_uvc_error(res, "uvc_open");
        if (res == uvc::ERROR_ACCESS)
        {
            print_device_permissions_msg();
        }
        
        return false;
    }        

    const uvc::format_desc* format_desc = uvc::uvc_get_format_descs(device.h_device);
    const uvc::frame_desc* frame_desc = format_desc->frame_descs;
    uvc::frame_format frame_format;
    int width = 640;
    int height = 480;
    int fps = 30;

    switch (format_desc->bDescriptorSubtype) 
    {
    case uvc::VS_FORMAT_MJPEG:
        frame_format = uvc::FRAME_FORMAT_MJPEG;
        break;
    case uvc::VS_FORMAT_FRAME_BASED:
        frame_format = uvc::FRAME_FORMAT_H264;
        break;
    default:
        frame_format = uvc::FRAME_FORMAT_YUYV;
        break;
    }

    device.format_code = (char*)format_desc->fourccFormat;

    if (frame_desc) 
    {
        width = frame_desc->wWidth;
        height = frame_desc->wHeight;
        fps = 10000000 / frame_desc->dwDefaultFrameInterval;
    }

    device.frame_width = width;
    device.frame_height = height;
    device.fps = fps;
    device.is_connected = true;

    res = uvc::uvc_get_stream_ctrl_format_size(
        device.h_device, device.ctrl, /* result stored in ctrl */
        frame_format,
        width, height, fps /* width, height, fps */
    );

    if (res != uvc::SUCCESS)
    {
        print_uvc_error(res, "uvc_get_stream_ctrl_format_size");
        disconnect_device(device);
        return false;
    }
    else
    {
        print_uvc_stream_info(device);
    }    

    return true;
}


static bool enumerate_devices(DeviceListUVC& list)
{
    uvc::device_descriptor* desc;

    auto res = uvc::uvc_init(&list.context, NULL);
    if (res != uvc::SUCCESS)
    {
        print_uvc_error(res, "uvc_init");
        uvc::uvc_exit(list.context);
        return false;
    }

    res = uvc::uvc_get_device_list(list.context, &list.device_list);
    if (res != uvc::SUCCESS)
    {
        print_uvc_error(res, "uvc_get_device_list");
        uvc::uvc_exit(list.context);
        return false;
    }

    if (!g_device_list.device_list[0])
    {
        uvc::uvc_exit(list.context);
        return false;
    }

    for (int i = 0; list.device_list[i]; ++i) 
    {
        DeviceUVC device;

        device.p_device = list.device_list[i];

        res = uvc::uvc_get_device_descriptor(device.p_device, &desc);
        if (res != uvc::SUCCESS)
        {
            print_uvc_error(res, "uvc_get_device_descriptor");
            continue;
        }

        device.device_id = i;
        device.product_id = desc->idProduct;
        device.vendor_id = desc->idVendor;

        list.devices.push_back(std::move(device));
        uvc::uvc_free_device_descriptor(desc);
    }

    if (list.devices.empty())
    {
        uvc::uvc_exit(list.context);
        return false;
    }

    // allocate for stream info
    std::vector<uvc::stream_ctrl> ctrls(list.devices.size());
    list.stream_ctrl_list = std::move(ctrls);

    list.is_connected = false;

    for (size_t i = 0; i < list.devices.size(); ++i)
    {        
        auto& dev = list.devices[i];
        auto ctrl = list.stream_ctrl_list.data() + i;
        list.is_connected |= connect_device(dev, ctrl);
    }

    return true;
}


static void stop_device(DeviceUVC& device)
{    
    if (device.is_streaming)
    {
        uvc::uvc_stop_streaming(device.h_device);
        device.h_stream = nullptr;
    }
}


static bool start_device_single_frame(DeviceUVC& device)
{ 
    if (device.is_streaming)
    {
        return true;
    }

    auto res = uvc::uvc_stream_open_ctrl(device.h_device, &device.h_stream, device.ctrl);
    if (res != uvc::SUCCESS)
    {
        print_uvc_error(res, "uvc_stream_open_ctrl");
        return false;
    }

    res = uvc::uvc_stream_start(device.h_stream, 0, (void*)12345, 0);
    if (res != uvc::SUCCESS)
    {
        print_uvc_error(res, "uvc_stream_start");
        uvc::uvc_stream_close(device.h_stream);
        return false;
    }

    device.is_streaming = true;

    return true;
}


static void enable_exposure_mode(DeviceUVC const& device)
{
    auto res = uvc::uvc_set_ae_mode(device.h_device, EXPOSURE_MODE_AUTO);
    if (res == uvc::SUCCESS)
    {
        return;
    }

    print_uvc_error(res, "uvc_set_ae_mode... auto");

    if (res == uvc::ERROR_PIPE)
    {
        res = uvc::uvc_set_ae_mode(device.h_device, EXPOSURE_MODE_APERTURE);
        if (res != uvc::SUCCESS)
        {
            print_uvc_error(res, "uvc_set_ae_mode... aperture");
        }
    }
}


static void close_all_devices()
{
    if (!g_device_list.is_connected)
    {
        return;
    }

    for (auto& device : g_device_list.devices)
    {
        stop_device(device);
        disconnect_device(device);
        free_device_frame(device);
    }
    
    g_device_list.is_connected = false;
    g_device_list.devices.clear();
    g_device_list.stream_ctrl_list.clear();

    uvc::uvc_free_device_list(g_device_list.device_list, 0);
    g_device_list.device_list = nullptr;

    uvc::uvc_exit(g_device_list.context);
    g_device_list.context = nullptr;
}


static bool grab_and_convert_frame(DeviceUVC& device)
{
    uvc::frame* frame;

    auto res = uvc::uvc_stream_get_frame(device.h_stream, &frame, 0);
    if (res != uvc::SUCCESS)
    {
        print_uvc_error(res, "uvc_stream_get_frame");
        return false;
    }

    auto rgb = device.rgb_frame;

    res = uvc::uvc_any2rgb(frame, rgb);
    if (res != uvc::SUCCESS)
    {  
        print_uvc_error(res, "uvc_any2rgb");
        return false;
    }

    return true;
}


namespace simage
{
    static bool camera_is_initialized(CameraUSB const& camera)
    {
        return camera.is_open
            && camera.device_id >= 0
            && camera.device_id < (int)g_device_list.devices.size();
    }


    bool open_camera(CameraUSB& camera)
    {
        if (!enumerate_devices(g_device_list))
        {
            print_device_error_info(g_device_list);
            return false;
        }

        print_device_list(g_device_list);

        auto result = get_default_device(g_device_list);
        if (!result)
        {
            print_error("No connected devices available");
            return false;
        }

        auto& device = *result;

        camera.device_id = device.device_id;
        camera.image_width = device.frame_width;
        camera.image_height = device.frame_height;
        camera.max_fps = device.fps;

        if (!create_image(camera.latest_frame, camera.image_width, camera.image_height))
        {
            uvc::uvc_free_device_list(g_device_list.device_list, 0);
            uvc::uvc_exit(g_device_list.context);
            return false;
        }

        camera.frame_roi = img::make_view(camera.latest_frame);

        free_device_frame(device);

        size_t frame_bytes = device.frame_width * device.frame_height * 3;

        device.rgb_frame = uvc::uvc_allocate_frame(frame_bytes);
        if (!device.rgb_frame)
        {
            print_error("Error allocating frame memory");
            free_device_frame(device);
            return false;
        }

        ImageRGB rgb;
        rgb.width = device.frame_width;
        rgb.height = device.frame_height;
        rgb.data_ = (RGBu8*)device.rgb_frame->data;

        device.rgb_view = img::make_view(rgb);

        if (!start_device_single_frame(device))
        {
            return false;
        }
        
        camera.is_open = true;

        enable_exposure_mode(device);

        return true;        
    }


    void close_camera(CameraUSB& camera)
    {
        camera.is_open = false;
        destroy_image(camera.latest_frame);

        if (camera.device_id < 0 || camera.device_id >= (int)g_device_list.devices.size())
		{
			return;
		}

        /*auto& device = g_device_list.devices[camera.device_id];
        if (device.is_streaming)
        {
            stop_device(device);
        }

        if (device.is_connected)
        {
            disconnect_device(device);
        }

        if (device.rgb_frame)
        {
            uvc::uvc_free_frame(device.rgb_frame);
            device.rgb_frame = nullptr;
        }*/

        close_all_devices();
    }


    bool grab_image(CameraUSB const& camera)
    {
        assert(verify(camera));
        
        if (!camera_is_initialized(camera))
        {
            return false;
        }
        
        auto& device = g_device_list.devices[camera.device_id];

        if (!grab_and_convert_frame(device))
        {
            return false;
        }

        auto roi = make_range(camera.frame_roi.width, camera.frame_roi.height);
        auto device_view = sub_view(device.rgb_view, roi);
        
        map_rgb(device_view, camera.frame_roi);

        return true;
    }


    bool grab_image(CameraUSB const& camera, View const& dst)
    {
        assert(verify(camera));
        
        if (!camera_is_initialized(camera))
        {
            return false;
        }
        
        auto& device = g_device_list.devices[camera.device_id];

        if (!grab_and_convert_frame(device))
        {
            return false;
        }

        auto roi = make_range(camera.frame_roi.width, camera.frame_roi.height);
        auto device_view = sub_view(device.rgb_view, roi);

        map_rgb(device_view, dst);

        return true;
    }


	bool grab_image(CameraUSB const& camera, view_callback const& grab_cb)
    {
        if (!camera.is_open || camera.device_id < 0 || camera.device_id >= (int)g_device_list.devices.size())
		{
			return false;
		}

        auto& device = g_device_list.devices[camera.device_id];

        if (!grab_and_convert_frame(device))
        {
            return false;
        }

        auto roi = make_range(camera.frame_roi.width, camera.frame_roi.height);
        auto device_view = sub_view(device.rgb_view, roi);

        map_rgb(device_view, camera.frame_roi);
        grab_cb(camera.frame_roi);

        return true;
    }


    bool grab_continuous(CameraUSB const& camera, view_callback const& grab_cb, bool_f const& grab_condition)
    {
        assert(verify(camera));
        
        if (!camera_is_initialized(camera))
        {
            return false;
        }

        auto& device = g_device_list.devices[camera.device_id];

        auto roi = make_range(camera.frame_roi.width, camera.frame_roi.height);
        auto device_view = sub_view(device.rgb_view, roi);

        auto frame_view = make_view(camera.latest_frame);
        
        while (grab_condition())
        {
            if (grab_and_convert_frame(device))
            {               
                map_rgb(device_view, camera.frame_roi);
				grab_cb(camera.frame_roi);
            }
        }

        return true;
    }
}