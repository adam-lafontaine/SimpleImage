#include "../simage/simage_platform.hpp"
#include "../util/stopwatch.hpp"
#include "../util/execute.hpp"
#include "../uvc/libuvc.h"

#include <array>
#include <thread>
#include <vector>

#include <cstdio>

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


static void print_uvc_error(uvc_error_t err, const char* msg)
{
    #ifndef NDEBUG

    uvc_perror(err, msg);

    #endif
}


class CameraUVC
{
public:
    uvc_device_t* device;
    uvc_device_handle_t* h_device;
    uvc_stream_ctrl_t ctrl;

    int product_id = -1;
    int vendor_id = -1;
    
    int frame_width = -1;
    int frame_height = -1;
    int fps = -1;

    bool is_open = false;
};


class CameraListUVC
{
public:
    uvc_context_t *context;
    uvc_device_t** device_list;

    std::vector<CameraUVC> cameras;

    bool is_open = false;
};


static CameraListUVC g_camera_list;


static bool enumerate_cameras(CameraListUVC& list)
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

    if (!g_camera_list.device_list[0])
    {
        uvc_exit(list.context);
        return false;
    }

    for (int i = 0; list.device_list[i]; ++i) 
    {
        CameraUVC camera;

        camera.device = list.device_list[i];

        res = uvc_get_device_descriptor(camera.device, &desc);
        if (res != UVC_SUCCESS)
        {
            print_uvc_error(res, "uvc_get_device_descriptor");
            continue;
        }

        camera.product_id = desc->idProduct;
        camera.vendor_id = desc->idVendor;

        list.cameras.push_back(std::move(camera));
        uvc_free_device_descriptor(desc);
    }

    if (list.cameras.empty())
    {
        uvc_exit(list.context);
        return false;
    }

    list.is_open = true;

    return true;
}


static void close_camera(CameraUVC& camera)
{
    if (camera.is_open)
    {
        uvc_close(camera.h_device);
        uvc_unref_device(camera.device);
        camera.frame_width = -1;
        camera.frame_height = -1;
        camera.fps = -1;
        camera.is_open = false;
    }
}


static bool open_camera(CameraUVC& camera)
{
    auto res = uvc_open(camera.device, &camera.h_device);
    if (res != UVC_SUCCESS)
    {
        print_uvc_error(res, "uvc_open");
        return false;
    }        

    const uvc_format_desc_t *format_desc = uvc_get_format_descs(camera.h_device);
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
        camera.h_device, &camera.ctrl, /* result stored in ctrl */
        frame_format,
        width, height, fps /* width, height, fps */
    );

    if (res != UVC_SUCCESS)
    {
        print_uvc_error(res, "uvc_get_stream_ctrl_format_size");
        close_camera(camera);
        return false;
    }

    #ifndef NDEBUG
    /* Print out the result */
    uvc_print_stream_ctrl(&camera.ctrl, stdout);
    #endif

    camera.frame_width = width;
    camera.frame_height = height;
    camera.fps = fps;
    camera.is_open = true;

    return true;
}


static void print_connected_camera_info(CameraListUVC const& list)
{
    for (int i = 0; i < list.cameras.size(); ++i)
    {
        auto const& cam = list.cameras[i];
        printf("Device %d:\n", i);
        printf("  Vendor ID:      0x%04x\n", cam.vendor_id);
        printf("  Product ID:     0x%04x\n", cam.product_id);
    }

    printf("\n%s\n", DEVICE_PERMISSION_MSG);
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


namespace simage
{
    bool open_camera(CameraUSB& camera)
    {
        if (!enumerate_cameras(g_camera_list))
        {
            return false;
        }

        auto& cam = g_camera_list.cameras.back();

        if (!open_camera(cam))
        {
            print_connected_camera_info(g_camera_list);
            return false;
        }

        return true;        
    }


    void close_all_cameras()
    {
        if (!g_camera_list.is_open)
        {
            return;
        }

        for (auto& camera : g_camera_list.cameras)
        {
            close_camera(camera);
        }

        uvc_exit(g_camera_list.context);
        g_camera_list.is_open = false;      
    }


    bool grab_image(CameraUSB const& camera, View const& dst)
    {

        return false;
    }


    bool grab_image(CameraUSB const& camera, bgr_callback const& grab_cb)
    {
        return false;
    }


    bool grab_continuous(CameraUSB const& camera, bgr_callback const& grab_cb, bool_f const& grab_condition)
    {
        return false;
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
}