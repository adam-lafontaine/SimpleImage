#include "../../simage.hpp"
#include "../../src/util/execute.hpp"
#include "../../src/util/color_space.hpp"

#define LIBUVC_IMPLEMENTATION 1
#include "libuvc2.hpp"

#include <vector>
#include <algorithm>

#ifndef NDEBUG
#include <cstdio>
#endif

namespace img = simage;


constexpr u8 EXPOSURE_MODE_AUTO = 2;
constexpr u8 EXPOSURE_MODE_APERTURE = 8;


/* verify */

#ifndef NDEBUG

namespace simage
{
	static bool verify(CameraUSB const& camera)
	{
		return camera.frame_width && camera.frame_height && camera.max_fps && camera.device_id >= 0;
	}


	template <typename T>
	static bool verify(CameraUSB const& camera, SubMatrix2D<T> const& view)
	{
		return verify(camera) && verify(view) &&
			camera.frame_width == view.width &&
			camera.frame_height == view.height;
	}
}

#endif


typedef uvc::uvc_error_t(convert_rgb_callback_t)(uvc::frame* in, img::Image const& dst);
typedef uvc::uvc_error_t(convert_gray_callback_t)(uvc::frame* in, img::ImageGray const& dst);


namespace convert
{
    static uvc::uvc_error_t rgb_error(uvc::frame* in, img::Image const& dst)
    {
        return uvc::UVC_ERROR_NOT_SUPPORTED;
    }


    static uvc::uvc_error_t gray_error(uvc::frame* in, img::ImageGray const& dst)
    {
        return uvc::UVC_ERROR_NOT_SUPPORTED;
    }


    class YUYV // UNTESTED
    {
    public:
        u8 y1;
        u8 u;
        u8 y2;
        u8 v;
    };


    static uvc::uvc_error_t yuyv_to_rgba(uvc::frame* in, img::Image const& dst)
    {
        auto src = (YUYV*)in->data;
        auto const convert_2_pixels = [&](u32 i)
        {
            auto s = src[i];
            auto& d1 = dst.data_[2 * i].rgba;
            auto& d2 = dst.data_[2 * i + 1].rgba;

            auto rgb = yuv::u8_to_rgb_u8(s.y1, s.u, s.v);
            d1.red = rgb.red;
            d1.green = rgb.green;
            d1.blue = rgb.blue;
            d1.alpha = 255;

            rgb = yuv::u8_to_rgb_u8(s.y2, s.u, s.v);
            d2.red = rgb.red;
            d2.green = rgb.green;
            d2.blue = rgb.blue;
            d2.alpha = 255;
        };

        process_range(0, dst.width * dst.height / 2, convert_2_pixels);

        return uvc::UVC_SUCCESS;
    }


    class YUV // UNTESTED
    {
    public:
        u8 y;
        u8 uv;
    };


    static uvc::uvc_error_t yuyv_to_gray(uvc::frame* in, img::ImageGray const& dst)
    {
        auto src = (YUV*)in->data;

        auto const convert_pixel = [&](u32 i)
        {
            dst.data_[i] = src[i].y;
        };

        process_range(0, dst.width * dst.height, convert_pixel);

        return uvc::UVC_SUCCESS;
    }


    class UYVY // UNTESTED
    {
    public:
        u8 y1;
        u8 u;
        u8 v;
        u8 y2;
    };


    static uvc::uvc_error_t uyvy_to_rgba(uvc::frame* in, img::Image const& dst)
    {
        auto src = (UYVY*)in->data;
        auto const convert_2_pixels = [&](u32 i)
        {
            auto s = src[i];
            auto& d1 = dst.data_[2 * i].rgba;
            auto& d2 = dst.data_[2 * i + 1].rgba;

            auto rgb = yuv::u8_to_rgb_u8(s.y1, s.u, s.v);
            d1.red = rgb.red;
            d1.green = rgb.green;
            d1.blue = rgb.blue;
            d1.alpha = 255;

            rgb = yuv::u8_to_rgb_u8(s.y2, s.u, s.v);
            d2.red = rgb.red;
            d2.green = rgb.green;
            d2.blue = rgb.blue;
            d2.alpha = 255;
        };

        process_range(0, dst.width * dst.height / 2, convert_2_pixels);

        return uvc::UVC_SUCCESS;
    }


    class UVY
    {
    public:
        u8 uv;
        u8 y;
    };


    static uvc::uvc_error_t uyvy_to_gray(uvc::frame* in, img::ImageGray const& dst)
    {
        auto src = (UVY*)in->data;

        auto const convert_pixel = [&](u32 i)
        {   
            dst.data_[i] = src[i].y;
        };

        process_range(0, dst.width * dst.height, convert_pixel);

        return uvc::UVC_SUCCESS;
    }


    static uvc::uvc_error_t mjpeg_to_rgba(uvc::frame* in, img::Image const& dst)
    {
        return uvc::opt::mjpeg2rgba(in, (u8*)dst.data_);
    }


    static uvc::uvc_error_t mjpeg_to_gray(uvc::frame* in, img::ImageGray const& dst)
    {
        return uvc::opt::mjpeg2gray(in, (u8*)dst.data_);
    }


    static uvc::uvc_error_t rgb_to_rgba(uvc::frame* in, img::Image const& dst)
    {
        auto src = (img::RGBu8*)in->data;

        auto const convert_pixel = [&](u32 i)
        {
            auto s = src[i];
            auto& d = dst.data_[i].rgba;

            d.red = s.red;
            d.green = s.green;
            d.blue = s.blue;
            d.alpha = 255;
        };

        process_range(0, dst.width * dst.height, convert_pixel);

        return uvc::UVC_SUCCESS;
    }


    static uvc::uvc_error_t rgb_to_gray(uvc::frame* in, img::ImageGray const& dst)
    {
        auto src = (img::RGBu8*)in->data;

        auto const convert_pixel = [&](u32 i)
        {
            auto s = src[i];
            auto& d = dst.data_[i];

            d = gray::u8_from_rgb_u8(s.red, s.green, s.blue);
        };

        process_range(0, dst.width * dst.height, convert_pixel);

        return uvc::UVC_SUCCESS;
    }


    static uvc::uvc_error_t bgr_to_rgba(uvc::frame* in, img::Image const& dst)
    {
        auto src = (img::BGRu8*)in->data;

        auto const convert_pixel = [&](u32 i)
        {
            auto s = src[i];
            auto& d = dst.data_[i].rgba;

            d.red = s.red;
            d.green = s.green;
            d.blue = s.blue;
            d.alpha = 255;
        };

        process_range(0, dst.width * dst.height, convert_pixel);

        return uvc::UVC_SUCCESS;
    }


    static uvc::uvc_error_t bgr_to_gray(uvc::frame* in, img::ImageGray const& dst)
    {
        auto src = (img::BGRu8*)in->data;

        auto const convert_pixel = [&](u32 i)
        {
            auto s = src[i];
            auto& d = dst.data_[i];

            d = gray::u8_from_rgb_u8(s.red, s.green, s.blue);
        };

        process_range(0, dst.width * dst.height, convert_pixel);

        return uvc::UVC_SUCCESS;
    }


    static uvc::uvc_error_t gray_to_rgba(uvc::frame* in, img::Image const& dst)
    {
        auto src = (u8*)in->data;

        auto const convert_pixel = [&](u32 i)
        {
            auto s = src[i];
            auto& d = dst.data_[i].rgba;

            d.red = s;
            d.green = s;
            d.blue = s;
            d.alpha = 255;
        };

        process_range(0, dst.width * dst.height, convert_pixel);

        return uvc::UVC_SUCCESS;
    }


    static uvc::uvc_error_t gray_to_gray(uvc::frame* in, img::ImageGray const& dst)
    {
        auto src = (u8*)in->data;

        auto const convert_pixel = [&](u32 i)
        {
            dst.data_[i] = src[i];
        };

        process_range(0, dst.width * dst.height, convert_pixel);

        return uvc::UVC_SUCCESS;
    }
}


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


class DeviceUVC
{
public:
    uvc::device* p_device = nullptr;
    uvc::device_handle* h_device = nullptr;
    uvc::stream_ctrl* ctrl = nullptr;
    uvc::stream_handle* h_stream = nullptr;

    convert_rgb_callback_t* convert_rgb = convert::rgb_error;
    convert_gray_callback_t* convert_gray = convert::gray_error;

    img::Image frame_rgb;
    img::ImageGray frame_gray;

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
        device.convert_rgb = convert::yuyv_to_rgba;
        device.convert_gray = convert::yuyv_to_gray;
        break;
    case uvc::UVC_FRAME_FORMAT_UYVY:
        device.convert_rgb = convert::uyvy_to_rgba;
        device.convert_gray = convert::uyvy_to_gray;
        break;
    case uvc::UVC_FRAME_FORMAT_MJPEG:
        device.convert_rgb = convert::mjpeg_to_rgba;
        device.convert_gray = convert::mjpeg_to_gray;
        break;
    case uvc::UVC_FRAME_FORMAT_RGB:
        device.convert_rgb = convert::rgb_to_rgba;
        device.convert_gray = convert::rgb_to_gray;
        break;
    case uvc::UVC_FRAME_FORMAT_BGR:
        device.convert_rgb = convert::bgr_to_rgba;;
        device.convert_gray = convert::bgr_to_gray;
        break;
    case uvc::UVC_FRAME_FORMAT_GRAY8:
        device.convert_rgb = convert::gray_to_rgba;
        device.convert_gray = convert::gray_to_gray;
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
        img::destroy_image(device.frame_rgb);
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
    
    res = device.convert_rgb(in_frame, device.frame_rgb);
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
    
    res = device.convert_gray(in_frame, device.frame_gray);
    if (res != uvc::UVC_SUCCESS)
    {  
        print_uvc_error(res, "device.convert_gray");
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

        auto const fail = [&]()
        {
            uvc::uvc_free_device_list(g_device_list.device_list, 0);
            uvc::uvc_exit(g_device_list.context);
            destroy_image(camera.frame_image);
            destroy_image(device.frame_rgb);
            return false;
        };

        auto width = device.frame_width;
        auto height = device.frame_height;

        camera.device_id = device.device_id;
        camera.max_fps = device.fps;
        camera.frame_width = width;
        camera.frame_height = height;        

        if (!create_image(camera.frame_image, width, height))
        {
            return fail();
        }

        if (!create_image(device.frame_rgb, width, height))
        {
            return fail();
        }

        device.frame_gray.width = width;
        device.frame_gray.height = height;
        device.frame_gray.data_ = (u8*)device.frame_rgb.data_;

        auto roi = make_range(width, height);       
        set_roi(camera, roi);

        if (!start_device_single_frame(device))
        {
            return fail();
        }        

        if (!set_frame_formats(device))
        {
            return fail();
        }
        
        camera.is_open = true;

        return true;        
    }


    void close_camera(CameraUSB& camera)
    {
        camera.is_open = false;

        destroy_image(camera.frame_image);

        if (camera.device_id < 0 || camera.device_id >= (int)g_device_list.devices.size())
		{
			return;
		}

        close_all_devices();
    }
    

    bool grab_rgb(CameraUSB const& camera, View const& dst)
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

        auto device_view = sub_view(device.frame_rgb, camera.roi);

        copy(device_view, dst);

        return true;
    }


	bool grab_rgb(CameraUSB const& camera, rgb_callback const& grab_cb)
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

        auto device_view = sub_view(device.frame_rgb, camera.roi);
        auto camera_view = sub_view(camera.frame_image, camera.roi);

        copy(device_view, camera_view);
        grab_cb(camera_view);

        return true;
    }


    bool grab_rgb_continuous(CameraUSB const& camera, rgb_callback const& grab_cb, bool_f const& grab_condition)
    {
        assert(verify(camera));
        
        if (!camera_is_initialized(camera))
        {
            return false;
        }

        auto& device = g_device_list.devices[camera.device_id];

        auto device_view = sub_view(device.frame_rgb, camera.roi);
        auto camera_view = sub_view(camera.frame_image, camera.roi);
        
        while (grab_condition())
        {
            if (grab_and_convert_frame_rgb(device))
            {               
                copy(device_view, camera_view);
                grab_cb(camera_view);
            }
        }

        return true;
    }


	bool grab_gray(CameraUSB const& camera, ViewGray const& dst)
    {
        assert(verify(camera));
        
        if (!camera_is_initialized(camera))
        {
            return false;
        }
        
        auto& device = g_device_list.devices[camera.device_id];

        if (!grab_and_convert_frame_gray(device))
        {
            return false;
        }

        auto device_view = sub_view(device.frame_gray, camera.roi);

        copy(device_view, dst);

        return true;
    }


	bool grab_gray(CameraUSB const& camera, gray_callback const& grab_cb)
    {
        if (!camera.is_open || camera.device_id < 0 || camera.device_id >= (int)g_device_list.devices.size())
		{
			return false;
		}

        auto& device = g_device_list.devices[camera.device_id];

        if (!grab_and_convert_frame_gray(device))
        {
            return false;
        }

        auto device_view = sub_view(device.frame_gray, camera.roi);

        grab_cb(device_view);

        return true;
    }


	bool grab_gray_continuous(CameraUSB const& camera, gray_callback const& grab_cb, bool_f const& grab_condition)
    {
        assert(verify(camera));
        
        if (!camera_is_initialized(camera))
        {
            return false;
        }

        auto& device = g_device_list.devices[camera.device_id];
        
        auto device_view = sub_view(device.frame_gray, camera.roi);
        
        while (grab_condition())
        {
            if (grab_and_convert_frame_gray(device))
            { 
				grab_cb(device_view);
            }
        }

        return true;
    }


    void set_roi(CameraUSB& camera, Range2Du32 roi)
    {
        if (roi.x_end <= camera.frame_image.width &&
            roi.x_begin < roi.x_end &&
            roi.y_end <= camera.frame_height &&
            roi.y_begin < roi.y_end)
        {
            camera.roi = roi;
        }        
    }
}