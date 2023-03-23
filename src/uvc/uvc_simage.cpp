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
		return camera.frame_width && camera.frame_height && camera.max_fps && camera.device_id >= 0;
	}


	template <typename T>
	static bool verify(CameraUSB const& camera, MatrixView<T> const& view)
	{
		return verify(camera) && verify(view) &&
			camera.frame_width == view.width &&
			camera.frame_height == view.height;
	}
}

#endif


typedef uvc::uvc_error_t(convert_frame_callback_t)(uvc::frame *in, uvc::frame *out);

static uvc::uvc_error_t convert_frame_error(uvc::frame *in, uvc::frame *out)
{
    return uvc::UVC_ERROR_NOT_SUPPORTED;
}


class DeviceUVC
{
public:
    uvc::device* p_device = nullptr;
    uvc::device_handle* h_device = nullptr;
    uvc::stream_ctrl* ctrl = nullptr;
    uvc::stream_handle* h_stream = nullptr;

    convert_frame_callback_t* convert_rgb = convert_frame_error;
    convert_frame_callback_t* convert_gray = convert_frame_error;

    uvc::frame* uvc_frame = nullptr;

    img::ViewRGB rgb_view;
    img::ViewGray gray_view;

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
    if (device.uvc_frame)
    {
        uvc::uvc_free_frame(device.uvc_frame);
        device.uvc_frame = nullptr;
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
    if (res != uvc::UVC_SUCCESS)
    {
        print_uvc_error(res, "uvc_open");
        if (res == uvc::UVC_ERROR_ACCESS)
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
    case uvc::UVC_VS_FORMAT_MJPEG:
        frame_format = uvc::UVC_FRAME_FORMAT_MJPEG;        
        break;
    case uvc::UVC_VS_FORMAT_FRAME_BASED:
        frame_format = uvc::UVC_FRAME_FORMAT_H264;
        break;
    default:
        frame_format = uvc::UVC_FRAME_FORMAT_YUYV;
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

    if (res != uvc::UVC_SUCCESS)
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
    if (res != uvc::UVC_SUCCESS)
    {
        print_uvc_error(res, "uvc_init");
        uvc::uvc_exit(list.context);
        return false;
    }

    res = uvc::uvc_get_device_list(list.context, &list.device_list);
    if (res != uvc::UVC_SUCCESS)
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
        if (res != uvc::UVC_SUCCESS)
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


static void enable_exposure_mode(DeviceUVC const& device)
{
    auto res = uvc::uvc_set_ae_mode(device.h_device, EXPOSURE_MODE_AUTO);
    if (res == uvc::UVC_SUCCESS)
    {
        return;
    }

    print_uvc_error(res, "uvc_set_ae_mode... auto");

    if (res == uvc::UVC_ERROR_PIPE)
    {
        res = uvc::uvc_set_ae_mode(device.h_device, EXPOSURE_MODE_APERTURE);
        if (res != uvc::UVC_SUCCESS)
        {
            print_uvc_error(res, "uvc_set_ae_mode... aperture");
        }
    }
}


static bool set_frame_formats(DeviceUVC& device)
{
    uvc::frame* frame;

    auto res = uvc::uvc_stream_get_frame(device.h_stream, &frame, 0);
    if (res != uvc::UVC_SUCCESS)
    {
        print_uvc_error(res, "uvc_stream_get_frame");
        return false;
    }

    switch(frame->frame_format)
    {
    case uvc::UVC_FRAME_FORMAT_YUYV:
        device.convert_rgb = uvc::xtra::uvc_yuyv2rgb;
        device.convert_gray = uvc::xtra::uvc_yuyv2y;
        break;
    case uvc::UVC_FRAME_FORMAT_UYVY:
        device.convert_rgb = uvc::xtra::uvc_uyvy2rgb;
        device.convert_gray = uvc::xtra::uvc_uyvy2y;
        break;
    case uvc::UVC_FRAME_FORMAT_MJPEG:
        device.convert_rgb = uvc::xtra::uvc_mjpeg2rgb;
        device.convert_gray = uvc::xtra::uvc_mjpeg2gray;
        break;
    case uvc::UVC_FRAME_FORMAT_RGB:
        device.convert_rgb = uvc::xtra::uvc_duplicate_frame;
        // rgb2gray
        break;
    case uvc::UVC_FRAME_FORMAT_BGR:
        device.convert_rgb = uvc::xtra::uvc_bgr2rgb;
        // bgr2gray
        break;
    case uvc::UVC_FRAME_FORMAT_GRAY8:
        device.convert_rgb = uvc::xtra::uvc_gray2rgb;
        device.convert_gray = uvc::xtra::uvc_duplicate_frame;
        break;
    case uvc::UVC_FRAME_FORMAT_GRAY16:

        break;
    case uvc::UVC_FRAME_FORMAT_NV12:

        break;
    default:
        break;
    }

    return true;
}


static bool start_device_single_frame(DeviceUVC& device)
{ 
    if (device.is_streaming)
    {
        return true;
    }

    auto res = uvc::uvc_stream_open_ctrl(device.h_device, &device.h_stream, device.ctrl);
    if (res != uvc::UVC_SUCCESS)
    {
        print_uvc_error(res, "uvc_stream_open_ctrl");
        return false;
    }

    res = uvc::uvc_stream_start(device.h_stream, 0, (void*)12345, 0);
    if (res != uvc::UVC_SUCCESS)
    {
        print_uvc_error(res, "uvc_stream_start");
        uvc::uvc_stream_close(device.h_stream);
        return false;
    }

    device.is_streaming = true;

    enable_exposure_mode(device);

    return true;
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


static bool grab_and_convert_frame_rgb(DeviceUVC& device)
{
    uvc::frame* in_frame;

    auto res = uvc::uvc_stream_get_frame(device.h_stream, &in_frame, 0);
    if (res != uvc::UVC_SUCCESS)
    {
        print_uvc_error(res, "uvc_stream_get_frame");
        return false;
    }

    auto out_frame = device.uvc_frame;
    
    res = device.convert_rgb(in_frame, out_frame);
    if (res != uvc::UVC_SUCCESS)
    {  
        print_uvc_error(res, "device.convert_rgb");
        return false;
    }

    return true;
}


static bool grab_and_convert_frame_gray(DeviceUVC& device)
{
    uvc::frame* in_frame;

    auto res = uvc::uvc_stream_get_frame(device.h_stream, &in_frame, 0);
    if (res != uvc::UVC_SUCCESS)
    {
        print_uvc_error(res, "uvc_stream_get_frame");
        return false;
    }

    auto out_frame = device.uvc_frame;
    
    res = device.convert_gray(in_frame, out_frame);
    if (res != uvc::UVC_SUCCESS)
    {  
        print_uvc_error(res, "device.convert_gray");
        return false;
    }

    return true;
}


namespace simage
{
    static Image make_frame_rgb(CameraUSB const& camera)
    {
        Image rgb;
        rgb.width = camera.frame_width;
        rgb.height = camera.frame_height;
        rgb.data_ = (Pixel*)camera.frame_data;

        return rgb;
    }


    static ImageGray make_frame_gray(CameraUSB const& camera)
    {
        ImageGray gray;
        gray.width = camera.frame_width;
        gray.height = camera.frame_height;
        gray.data_ = camera.frame_data;

        return gray;
    }


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
        camera.frame_width = device.frame_width;
        camera.frame_height = device.frame_height;
        camera.max_fps = device.fps;

        camera.frame_data = (u8*)malloc(camera.frame_width * camera.frame_height * sizeof(Pixel));

        if (!camera.frame_data)
        {
            uvc::uvc_free_device_list(g_device_list.device_list, 0);
            uvc::uvc_exit(g_device_list.context);
            return false;
        }

        auto roi = make_range(camera.frame_width, camera.frame_height);       
        set_roi(camera, roi);

        free_device_frame(device);
        size_t frame_max_bytes = device.frame_width * device.frame_height * 3;

        device.uvc_frame = uvc::uvc_allocate_frame(frame_max_bytes);
        if (!device.uvc_frame)
        {
            print_error("Error allocating frame memory");
            free_device_frame(device);
            return false;
        }

        ImageRGB rgb;
        rgb.width = device.frame_width;
        rgb.height = device.frame_height;
        rgb.data_ = (RGBu8*)device.uvc_frame->data;

        device.rgb_view = img::make_view(rgb);

        ImageGray gray;
        gray.width = rgb.width;
        gray.height = rgb.height;
        gray.data_ = (u8*)rgb.data_;

        device.gray_view = img::make_view(gray);

        if (!start_device_single_frame(device))
        {
            return false;
        }        

        if (!set_frame_formats(device))
        {
            return false;
        }
        
        camera.is_open = true;

        return true;        
    }


    void close_camera(CameraUSB& camera)
    {
        camera.is_open = false;

        auto rgb = make_frame_rgb(camera);

        destroy_image(rgb);

        if (camera.device_id < 0 || camera.device_id >= (int)g_device_list.devices.size())
		{
			return;
		}

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

        if (!grab_and_convert_frame_rgb(device))
        {
            return false;
        }

        auto roi = make_range(camera.rgb_roi.width, camera.rgb_roi.height);
        auto device_view = sub_view(device.rgb_view, roi);
        
        map_rgb(device_view, camera.rgb_roi);

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

        if (!grab_and_convert_frame_rgb(device))
        {
            return false;
        }

        auto roi = make_range(camera.rgb_roi.width, camera.rgb_roi.height);
        auto device_view = sub_view(device.rgb_view, roi);

        map_rgb(device_view, dst);

        return true;
    }


	bool grab_image(CameraUSB const& camera, rgb_callback const& grab_cb)
    {
        if (!camera.is_open || camera.device_id < 0 || camera.device_id >= (int)g_device_list.devices.size())
		{
			return false;
		}

        auto& device = g_device_list.devices[camera.device_id];

        if (!grab_and_convert_frame_rgb(device))
        {
            return false;
        }

        auto roi = make_range(camera.rgb_roi.width, camera.rgb_roi.height);
        auto device_view = sub_view(device.rgb_view, roi);

        map_rgb(device_view, camera.rgb_roi);
        grab_cb(camera.rgb_roi);

        return true;
    }


    bool grab_continuous(CameraUSB const& camera, rgb_callback const& grab_cb, bool_f const& grab_condition)
    {
        assert(verify(camera));
        
        if (!camera_is_initialized(camera))
        {
            return false;
        }

        auto& device = g_device_list.devices[camera.device_id];

        auto roi = make_range(camera.rgb_roi.width, camera.rgb_roi.height);
        auto device_view = sub_view(device.rgb_view, roi);
        
        while (grab_condition())
        {
            if (grab_and_convert_frame_rgb(device))
            {               
                map_rgb(device_view, camera.rgb_roi);
				grab_cb(camera.rgb_roi);
            }
        }

        return true;
    }


    void set_roi(CameraUSB& camera, Range2Du32 const& roi)
    {
        auto rgb = make_frame_rgb(camera);
        camera.rgb_roi = sub_view(rgb, roi);

        auto gray = make_frame_gray(camera);
        camera.gray_roi = sub_view(gray, roi);
    }
}