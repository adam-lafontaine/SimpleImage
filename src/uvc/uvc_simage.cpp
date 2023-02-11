#include "../simage/simage_platform.hpp"
#include "../util/stopwatch.hpp"
#include "../util/execute.hpp"
#include "../uvc/libuvc.h"

#include <array>
#include <thread>
#include <vector>
#include <algorithm>

#ifndef NDEBUG
#include <cstdio>
#endif

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
"Replace XXXX and YYYY for the 4 hexadecimal characters corresponding to the vendor and product ID of your webcams.";

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
		return camera.image_width && camera.image_height && camera.max_fps && camera.id >= 0;
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
    uvc_device_t* p_device;
    uvc_device_handle_t* h_device;
    uvc_stream_ctrl_t* ctrl;

    int device_id = -1;

    int product_id = -1;
    int vendor_id = -1;
    
    int frame_width = -1;
    int frame_height = -1;
    int fps = -1;

    const char* format_code;

    bool is_connected = false;
    bool is_streaming = false;

    bool grab_single = false;

    uvc_frame_t* rgb_frames[2];
    u32 frame_curr = 0;
	u32 frame_prev = 1;
};


class DeviceListUVC
{
public:
    uvc_context_t *context;
    uvc_device_t** device_list;

    std::vector<uvc_stream_ctrl_t> stream_ctrl_list;

    std::vector<DeviceUVC> devices;

    bool is_connected = false;
};


static DeviceListUVC g_device_list;

class RGB
{
public:
    u8 red;
    u8 green;
    u8 blue;
};


static img::Pixel to_pixel(RGB const& rgb)
{
    img::Pixel p{};

    p.rgba.red = rgb.red;
    p.rgba.green = rgb.green;
    p.rgba.blue = rgb.blue;
    p.rgba.alpha = 255;

    return p;
}


static void print_uvc_error(uvc_error_t err, const char* msg)
{
#ifndef NDEBUG

    uvc_perror(err, msg);

#endif
}


static void print_error(const char* msg)
{
#ifndef NDEBUG

    printf("%s\n", msg);

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

    printf("\n%s\n", DEVICE_PERMISSION_MSG);

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
        uvc_close(device.h_device);
        uvc_unref_device(device.p_device);
        device.frame_width = -1;
        device.frame_height = -1;
        device.fps = -1;
        device.is_connected = false;        
    }
}


static bool connect_device(DeviceUVC& device, uvc_stream_ctrl_t* ctrl)
{
    device.ctrl = ctrl;

    auto res = uvc_open(device.p_device, &device.h_device);
    if (res != UVC_SUCCESS)
    {
        print_uvc_error(res, "uvc_open");
        return false;
    }        

    const uvc_format_desc_t* format_desc = uvc_get_format_descs(device.h_device);
    const uvc_frame_desc_t* frame_desc = format_desc->frame_descs;
    enum uvc_frame_format frame_format;
    int width = 640;
    int height = 480;
    int fps = 30;

    switch (format_desc->bDescriptorSubtype) 
    {
    case UVC_VS_FORMAT_MJPEG:
        frame_format = UVC_FRAME_FORMAT_MJPEG;
        break;
    case UVC_VS_FORMAT_FRAME_BASED:
        frame_format = UVC_FRAME_FORMAT_H264;
        break;
    default:
        frame_format = UVC_FRAME_FORMAT_YUYV;
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

    res = uvc_get_stream_ctrl_format_size(
        device.h_device, device.ctrl, /* result stored in ctrl */
        frame_format,
        width, height, fps /* width, height, fps */
    );

    if (res != UVC_SUCCESS)
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
    uvc_device_descriptor_t* desc;

    auto res = uvc_init(&list.context, NULL);
    if (res != UVC_SUCCESS)
    {
        print_uvc_error(res, "uvc_init");
        uvc_exit(list.context);
        return false;
    }

    res = uvc_get_device_list(list.context, &list.device_list);
    if (res != UVC_SUCCESS)
    {
        print_uvc_error(res, "uvc_get_device_list");
        uvc_exit(list.context);
        return false;
    }

    if (!g_device_list.device_list[0])
    {
        uvc_exit(list.context);
        return false;
    }

    for (int i = 0; list.device_list[i]; ++i) 
    {
        DeviceUVC device;

        device.p_device = list.device_list[i];

        res = uvc_get_device_descriptor(device.p_device, &desc);
        if (res != UVC_SUCCESS)
        {
            print_uvc_error(res, "uvc_get_device_descriptor");
            continue;
        }

        device.device_id = i;
        device.product_id = desc->idProduct;
        device.vendor_id = desc->idVendor;

        list.devices.push_back(std::move(device));
        uvc_free_device_descriptor(desc);
    }

    if (list.devices.empty())
    {
        uvc_exit(list.context);
        return false;
    }

    // allocate for stream info
    std::vector<uvc_stream_ctrl_t> ctrls(list.devices.size());
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


static void uvc_single_frame_callback(uvc_frame_t* frame, void* data)
{
    auto& device = *(DeviceUVC*)data;

    if (!device.grab_single)
    {
        return;
    }

    auto rgb = device.rgb_frames[device.frame_curr];    

    auto res = uvc_any2rgb(frame, rgb);
    if (res != UVC_SUCCESS)
    {
        print_uvc_error(res, "uvc_any2rgb");
        device.grab_single = false;
        return;
    }
    
    device.grab_single = false;
}


static void stop_device(DeviceUVC& device)
{    
    uvc_stop_streaming(device.h_device);
    device.is_streaming = false;

    uvc_free_frame(device.rgb_frames[0]);
    uvc_free_frame(device.rgb_frames[1]);
}


static bool start_device(DeviceUVC& device, uvc_frame_callback_t callback)
{
    size_t frame_bytes = device.frame_width * device.frame_height * 3;
    device.rgb_frames[0] = uvc_allocate_frame(frame_bytes);
    device.rgb_frames[1] = uvc_allocate_frame(frame_bytes);

    if (!device.rgb_frames[0] || !device.rgb_frames[1])
    {
        print_error("Error allocating frame memory");
        return false;
    }   
    
    auto res = uvc_start_streaming(device.h_device, device.ctrl, callback, (void *)(&device), 0);

    if (res != UVC_SUCCESS)
    {
        print_uvc_error(res, "uvc_start_streaming");
        return false;
    }

    device.is_streaming = true;

    return true;
}


static void enable_exposure_mode(DeviceUVC const& device)
{
    auto res = uvc_set_ae_mode(device.h_device, EXPOSURE_MODE_AUTO);
    if (res == UVC_SUCCESS)
    {
        return;
    }

    print_uvc_error(res, "uvc_set_ae_mode... auto");

    if (res == UVC_ERROR_PIPE)
    {
        res = uvc_set_ae_mode(device.h_device, EXPOSURE_MODE_APERTURE);
        if (res != UVC_SUCCESS)
        {
            print_uvc_error(res, "uvc_set_ae_mode... aperture");
        }
    }
}


namespace simage
{
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

        camera.id = device.device_id;
        camera.image_width = device.frame_width;
        camera.image_height = device.frame_height;
        camera.max_fps = device.fps;

        if (!start_device(device, uvc_single_frame_callback))
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

        if (camera.id < 0 || camera.id >= (int)g_device_list.devices.size())
		{
			return;
		}

        auto& device = g_device_list.devices[camera.id];
        if (device.is_streaming)
        {
            stop_device(device);
        }

        if (device.is_connected)
        {
            disconnect_device(device);
        }
    }


    void close_all_cameras()
    {
        if (!g_device_list.is_connected)
        {
            return;
        }

        for (auto& device : g_device_list.devices)
        {
            if (device.is_streaming)
            {
                stop_device(device);
            }

            if (device.is_connected)
            {
                disconnect_device(device);
            }
        }

        uvc_exit(g_device_list.context);
        g_device_list.is_connected = false;  

        g_device_list.devices.clear();
        g_device_list.stream_ctrl_list.clear();

        uvc_free_device_list(g_device_list.device_list, 0);
    }


    bool grab_image(CameraUSB const& camera, View const& dst)
    {
        assert(verify(camera, dst));

        if (!camera.is_open || camera.id < 0 || camera.id >= (int)g_device_list.devices.size())
		{
			return false;
		}

        auto& device = g_device_list.devices[camera.id];
        if (!device.is_streaming)
        {
            return false;
        }

        device.grab_single = true;
        while (device.grab_single)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        auto& frame = *(device.rgb_frames[device.frame_curr]);
        auto src = (RGB*)frame.data;

        for (u32 y = 0; y < dst.height; ++y)
        {
            auto d = img::row_begin(dst, y);
            for (u32 x = 0; x < dst.width; ++x)
            {
                d[x] = to_pixel(*src);
                ++src;
            }
        }

        device.frame_curr = device.frame_curr ? 0 : 1;
        device.frame_prev = device.frame_curr ? 0 : 1;

        return true;
    }


    bool grab_image(CameraUSB const& camera, view_callback const& grab_cb)
    {
        if (!camera.is_open || camera.id < 0 || camera.id >= (int)g_device_list.devices.size())
		{
			return false;
		}

        auto& device = g_device_list.devices[camera.id];
        if (!device.is_streaming)
        {
            return false;
        }



        return false;
    }


    bool grab_continuous(CameraUSB const& camera, view_callback const& grab_cb, bool_f const& grab_condition)
    {
        if (!camera.is_open || camera.id < 0 || camera.id >= (int)g_device_list.devices.size())
		{
			return false;
		}

        auto& device = g_device_list.devices[camera.id];
        if (!device.is_streaming)
        {
            return false;
        }

        return false;
    }


    
}


static void print_connected_device_info_raw(uvc_context_t *ctx)
{
    uvc_device_t** list;
    uvc_device_t* dev;
    uvc_device_descriptor_t* desc;
    uvc_error_t res;

    printf("Finding connected UVC devices\n");

    res = uvc_get_device_list(ctx, &list);
    if (res != UVC_SUCCESS)
    {
        printf("Unable to get devices\n");
        return;
    }

    if (!list[0])
    {
        printf("No connected cameras found\n");
    }

    for (int i = 0; list[i]; ++i) 
    {
        dev = list[i];

        // Retrieve the device descriptor
        res = uvc_get_device_descriptor(dev, &desc);
        if (res != UVC_SUCCESS)
        {
            printf("Error retrieving device descriptor: %d\n", res);
            continue;
        }

        // Print the device information
        printf("Device %d:\n", i);
        printf("  Vendor ID:      0x%04x\n", desc->idVendor);
        printf("  Product ID:     0x%04x\n", desc->idProduct);
        printf("  Manufacturer:   %s\n", desc->manufacturer);
        printf("  Product:        %s\n", desc->product);
        printf("  Serial Number:  %s\n", desc->serialNumber);

        uvc_free_device_descriptor(desc);
    }

    uvc_free_device_list(list, 1);
    printf("\n%s\n", DEVICE_PERMISSION_MSG);
}


static bool example()
{
    uvc_context_t *ctx;
    uvc_device_t *dev;
    uvc_device_handle_t *devh;
    uvc_stream_ctrl_t ctrl;
    uvc_error_t res;

    res = uvc_init(&ctx, NULL);
    if (res != UVC_SUCCESS)
    {
        uvc_perror(res, "uvc_init");
        //return res;

        return false;
    }

    int vendor_id = 0;
    int product_id = 0;
    const char* serial_number = NULL;

    res = uvc_find_device(ctx, &dev, vendor_id, product_id, serial_number);
    if (res != UVC_SUCCESS)
    {
        uvc_perror(res, "uvc_find_device");
        return false;
    }

    res = uvc_open(dev, &devh);
    if (res != UVC_SUCCESS)
    {
        uvc_perror(res, "uvc_open"); /* unable to open device */
        print_connected_device_info_raw(ctx);
        return false;
    }        

    const uvc_format_desc_t *format_desc = uvc_get_format_descs(devh);
    const uvc_frame_desc_t *frame_desc = format_desc->frame_descs;
    enum uvc_frame_format frame_format;
    int width = 640;
    int height = 480;
    int fps = 30;

    switch (format_desc->bDescriptorSubtype) 
    {
    case UVC_VS_FORMAT_MJPEG:
        frame_format = UVC_FRAME_FORMAT_MJPEG;
        break;
    case UVC_VS_FORMAT_FRAME_BASED:
        frame_format = UVC_FRAME_FORMAT_H264;
        break;
    default:
        frame_format = UVC_FRAME_FORMAT_YUYV;
        break;
    }

    if (frame_desc) 
    {
        width = frame_desc->wWidth;
        height = frame_desc->wHeight;
        fps = 10000000 / frame_desc->dwDefaultFrameInterval;
    }

    printf("\nFirst format: (%4s) %dx%d %dfps\n", format_desc->fourccFormat, width, height, fps);

    res = uvc_get_stream_ctrl_format_size(
        devh, &ctrl, /* result stored in ctrl */
        frame_format,
        width, height, fps /* width, height, fps */
    );

    /* Print out the result */
    uvc_print_stream_ctrl(&ctrl, stdout);

    if (res != UVC_SUCCESS)
    {
        uvc_perror(res, "get_mode"); /* device doesn't provide a matching stream */

        uvc_close(devh);
        uvc_unref_device(dev);
        uvc_exit(ctx);
        return false;
    }

    uvc_close(devh);
    uvc_unref_device(dev);
    uvc_exit(ctx);

    return true;
}
