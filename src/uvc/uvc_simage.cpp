#include "../simage/simage_platform.hpp"
#include "../util/stopwatch.hpp"
#include "../util/execute.hpp"
#include "../uvc/libuvc.h"

#include <array>
#include <thread>

#include <cstdio>

/*

sudo apt-get install v4l-utils
v4l2-ctl --list-devices

*/

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


static void print_connected_device_info(uvc_context_t *ctx)
{
    uvc_device_t** list;
    uvc_device_t* dev;
    uvc_device_descriptor_t* desc;
    uvc_error_t res;

    printf("Finding connected UVC devices\n");

    res = uvc_get_device_list(ctx, &list);
    if ((int)res < 0)
    {
        printf("Unable to get devices\n");
        return;
    }

    for (int i = 0; list[i]; ++i) 
    {
        dev = list[i];

        // Retrieve the device descriptor
        res = uvc_get_device_descriptor(dev, &desc);
        if ((int)res < 0)
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
        uvc_context_t *ctx;
        uvc_device_t *dev;
        uvc_device_handle_t *devh;
        uvc_stream_ctrl_t ctrl;
        uvc_error_t res;

        res = uvc_init(&ctx, NULL);
        if ((int)res < 0)
        {
            uvc_perror(res, "uvc_init");
            //return res;

            return false;
        }

        int vendor_id = 0;
        int product_id = 0;
        const char* serial_number = NULL;

        res = uvc_find_device(ctx, &dev, vendor_id, product_id, serial_number);
        if ((int)res < 0)
        {
            uvc_perror(res, "uvc_find_device"); /* no devices found */
            return false;
        }

        res = uvc_open(dev, &devh);
        if ((int)res < 0)
        {
            uvc_perror(res, "uvc_open"); /* unable to open device */
            print_connected_device_info(ctx);
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
            frame_format = UVC_COLOR_FORMAT_MJPEG;
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

        if ((int)res < 0)
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


    void close_all_cameras()
    {

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
}