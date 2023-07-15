# SimpleImage

A simple image processing library written in C++17.

## Definitions

### Image

* A contiguous block of memory containing pixel data
* Owns the image memory and must be destroyed
* Read from file, grabbed by camera etc.
* Not processed directly
* Processing is done via a "View" to the image data

### View

* Provides access to all or part of an image
* Represents an entire image or a rectangular sub-section
* Does not own the memory
* Its memory can be that of an "Image" or part of a "MemoryBuffer"

### MemoryBuffer

* A pre-allocated stack for storing image view data
* Owns the memory and must be destroyed

### Interleaved

* | R0 | G0 | B0 | A0 | R1 | G1 | B1 | A1 | R2 | G2 | B2 | A2 |...

### Planar

* | R0 | R1 | R2 |..., | G0 | G1 | G2 |..., | B0 | B1 | B2 |..., | A0 | A1 | A2 |...,

### Channel View

* A view to multiple channels of planar image data

### Platform Image/View

An interleaved image or view.  It is in the format suitable for interacting with the "platform", such as writing to file or rendering in a window.

## Types

* `Image`: 4 byte RGBA interleaved image data
* `View`: A view to 4 byte image data
* `ImageGray`: 1 byte image data
* `ViewGray`: A view to 1 byte image data
* `View1f32`: Single channel float view
* `View2f32`, `View3f32`, `View4f32`: Multi-channel float view
* `Buffer32`: Allocates data for 4 byte pixel or float channel data
* `Buffer8`: Allocates data for 1 byte pixel data
* `DeviceImage` (CUDA): todo
* `DeviceView` (CUDA): todo
* `DeviceBuffer` (CUDA): todo

## API Overview

**See the `/test_apps/` directory for complete examples**

### Interleaved / Platform

Read, resize, write

```cpp
namespace img = simage;


img::Image image;

auto success = img::read_image_from_file("file_path", image);
if (!success)
{
    // error
}

auto new_width = image.width * 2;
auto new_height = image.height / 2;

img::Image image2;

success = img::resize_image(image, image2, width, height);
if (!success)
{
    // error
}

success = img::write_image(image2, "new_file_path");
if (!success)
{
    // error
}

img::destroy_image(image);
img::destroy_image(image2);
```

Make a view from an image

```cpp
namespace img = simage;


img::Image image;
auto success = img::read_image_from_file("file_path", image);
if (!success)
{
    // error
}

auto view = img::make_view(image);

// ...

img::destroy_image(image);
```

Using a MemoryBuffer

```cpp
namespace img = simage;


u32 width = 1080;
u32 height = 720;

u32 n_views = 2;

auto n_pixels = width * height * n_views;

auto buffer = img::create_buffer32(n_pixels);

img::Image image;
auto src = img::make_view_resized_from_file("file_path", image, width, height, buffer);
auto dst = img::make_view(width, height, buffer);

img::blur(src, dst);

// ...

img::destroy_image(image);
img::destroy_buffer(buffer);
```

Grayscale images

```cpp
namespace img = simage;


u32 width = 1080;
u32 height = 720;

auto buffer = img::create_buffer8(width * height * 2);

img::ImageGray image;
auto src = img::make_view_resized_from_file("file_path", image, width, height, buffer);
auto dst = img::make_view(width, height, buffer);

img::gradients(src, dst);

img::destroy_image(image);
img::destroy_buffer(buffer);
```

Isolate a rectagular region of an image for processing, without making a copy.

```cpp
namespace img = simage;


img::Image image;
auto success = img::read_image_from_file("file_path", image);
if (!success)
{
    // error
}

auto w = image.width;
auto h = image.height;

// upper left region of image
auto region = make_range(w / 2, h / 2);

auto view = img::sub_view(image, region);

// ...

auto w2 = view.width;
auto h2 = view.height;

// center of the upper left region
region.x_begin = w2 / 4;
region.x_end = w2 * 3 / 4;
region.y_begin = h2 / 4;
region.y_end = h2 * 3 / 4;

auto view2 = img::sub_view(view, region);

// ...

img::destroy_image(image);
```

Convert between different image formats

```cpp
View view;
ViewGray gray;
ViewYUV yuv;

// ...

img::map_gray(view, gray); // RGBA to grayscale
img::map_yuv(yuv, view);   // YUYV to RGBA
```

Create custom image transforms

```cpp
namespace img = simage;


Image image;
auto src = img::make_view_from_file("file_path", image);

auto width = src.width;
auto height = src.height;

auto buffer = img::create_buffer32(width * height);

auto dst = img::make_view(width, height, buffer);

auto const invert = [](img::Pixel p)
{
    p.rgba.red = 255 - p.rgba.red;
    p.rgba.green = 255 - p.rgba.green;
    p.rgba.blue = 255 - p.rgba.blue;

    return p;
};

img::transform(src, dst, invert);

// copy the transformed data back to the original image memory
img::copy(dst, src);

img::write_image(image, "new_file_path");

img::destroy_image(image);
img::destroy_buffer(buffer);
```

Other functions

```cpp
fill()
copy()
alpha_blend()
for_each_pixel()
threshold()
binarize()
blur()
gradients()
split_rgb()
rotate()
centroid()
skeleton()
make_histograms()
```

**See `/test_apps/interleaved_tests/`**

### Planar / Channel

Make a view with up to 4 channels using a MemoryBuffer

```cpp
namespace img = simage;


u32 width = 1080;
u32 height = 720;
u32 n_channels = 10;

auto buffer     = img::create_buffer32(width * height * n_channels);

auto gray       = img::make_view_1(width, height, buffer);

auto gray_alpha = img::make_view_2(width, height, buffer);

auto rgb        = img::make_view_3(width, height, buffer);

auto rgba       = img::make_view_4(width, height, buffer);

// ...

img::destroy_buffer(buffer);
```

Convert between platform view and channel view 

```cpp
namespace img = simage;


Image image;
auto view = img::make_view_from_file("file_path", image);

auto width = view.width;
auto height = view.height;

auto buffer = img::create_buffer32(width * height * 3);

auto rgb = img::make_view_3(width, height, buffer);

img::map_rgb(view, rgb);

// ...

img::map_rgb(rgb, view);

// ...

img::destroy_image(image);
img::destroy_buffer(buffer);
```

Convert between color spaces

```cpp
namespace img = simage;


Image image;
auto view = img::make_view_from_file("file_path", image);

auto width = view.width;
auto height = view.height;

auto buffer = img::create_buffer32(width * height * 3);

auto hsv = img::make_view_3(width, height, buffer);

img::map_rgb_hsv(view, hsv); // supports HSV, YUV, LCH

// ...

img::destroy_image(image);
img::destroy_buffer(buffer);
```

Select a single channel as a separate view

```cpp
namespace img = simage;


u32 width = 1080;
u32 height = 720;

auto buffer = img::create_buffer(width * height * 3);

auto rgb = img::make_view_3(width, height, buffer);

auto green = img::select_channel(rgb, img::RGB::G);

// ...

img::destroy_buffer(buffer);
```

Other functions

```cpp
sub_view()
select_rgb()
map_gray()
fill()
copy()
transform()
threshold()
binarize()
alpha_blend()
rotate()
blur()
gradients()
```

**See `/test_apps/planar_tests/`**

### USB Camera

Grab an image

```cpp
namespace img = simage;


CameraUSB camera;

auto success = img::open_camera(camera);

if (!success)
{
    // error
}

auto width = camera.frame_width;
auto height = camera.frame_height;

auto buffer = img::make_buffer32(width * height);

auto view = img::make_view(width, height, buffer);

success = grab_rgb(camera, view);
if (!success)
{
    // error
}

// ...

img::destroy_buffer(buffer);
img::close_camera(camera);
```

Grab with a callback

```cpp
namespace img = simage;


CameraUSB camera;
if (!img::open_camera(camera))
{
    // error
}

auto width = camera.frame_width;
auto height = camera.frame_height;

img::Image image;
if (!img::create_image(image, width, height))
{
    // error
}

auto view = img::make_view(image);


int id = 0;
f32 angle = 0.0f;
Point2Du32 center = { width / 2, height / 2 };

auto const rotate_and_save = [&](img::View const& frame) 
{
    img::rotate(frame, view, center, angle);
    angle += 0.25f;
    img::write_image(image, make_file_path_by_id(id++));
};

img::grab _rgb(camera, rotate_and_save);
img::grab _rgb(camera, rotate_and_save);
img::grab _rgb(camera, rotate_and_save);

// ...

img::close_camera(camera);
img::destroy_image(image);
```

Grab continuous

```cpp
#include <thread>

namespace img = simage;


CameraUSB camera;
if (!img::open_camera(camera))
{
    // error
}

int id = 0;

// will keep grabbing while this function returns true
auto const grab_cond = [&id]() { return id < 100; };

auto const process_frame = [](img::View const& frame) { ... };

// grab_continuous() is blocking.  Start in a separate thread
std::thread th([&]() { img::grab_continuous(camera, process_frame, grab_cond); });

// ...

th.join();

img::close_camera(camera);
```

Other functions

```cpp
grab_gray()
grab_gray_continuous()
set_roi()
```

**See `/test_apps/usb_camera_tests/`**

### Histograms

The namespace simage::hist contains functionality for creating histograms of various color spaces.  Still not sure if it belongs here or somewhere else.

**See `/test_apps/hist_camera_test/`**

### CUDA

Basic implementation for processing images on Nvidia GPUs is on the way.

### Settings

**See `/simage/defines.hpp`**

```cpp
// Support .png image files
#define SIMAGE_PNG

// Support .bmp image files
#define SIMAGE_BMP

// Disable multithreaded image processing
#define SIMAGE_NO_PARALLEL

// Disable std::filesystem file paths as an alternative to const char*
// Uses std::string instead
#define SIMAGE_NO_FILESYSTEM

// Disable USB camera support
#define SIMAGE_NO_USB_CAMERA

// Disable CUDA GPU support
#define SIMAGE_NO_CUDA
```

## Credits (dependencies)

* [stb_image](https://github.com/nothings/stb): Read, write, resize images (included)
* [libuvc](https://github.com/libuvc/libuvc): Webcam support - Linux (included, requires libusb-1.0)
* [OpenCV](https://opencv.org/): Wecam support - Windows (requires install)
* [SDL2](https://www.libsdl.org/): Rendering test application examples (requires install)

## Compile Instructions

* Copy the /simage directory to your project
* #include "{your_path}/simage/simage.hpp" for the api
* Compile/link with {your_path}/simage/simage.cpp
* Note: For USB camera functionality, install libusb for Linux or OpenCV for windows
    * Or #define SIMAGE_NO_USB_CAMERA

## Run the examples

Install the required libraries

* libusb (USB camera, Linux)
* OpenCV (USB camera, Windows)

Edit the `ROOT_DIR` variable in `/test_apps/tests_include.hpp` to where your project is.

Windows

* Use the Visual Studio solution provided

Linux

* Navigate to one of the test app directories
* Run `make setup` to create a build directory
* Run `make build` / `make run`
