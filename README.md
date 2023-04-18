# SimpleImage

A simple image processing library written in C++17.

## Definitions

### Image

* A contiguous block of memory containing the pixel data
* Owns the image memory and must be destroyed
* Read from file, grabbed by camera etc.
* Not processed directly
* Processing is done via a "view" to the image data

### View

* Provides access to all or part of an image
* Represents an entire image or a rectangular sub-section
* Does not own memory

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

* Image: 4 byte RGBA interleaved image data
* View: A view to 4 byte image data
* ImageGray: 1 byte image data
* ViewGray: A view to 1 byte image data
* View1f32: Single channel float view
* View2f32, View3f32, View4f32: Multi-channel float view
* Buffer32: Allocates data for 4 byte pixel or float channel data
* Buffer8: Allocates data for 1 byte pixel data
* DeviceImage (CUDA): todo
* DeviceView (CUDA): todo
* DeviceBuffer (CUDA): todo

## Compile Instructions

* Copy the /simage directory to your project
* #include "{path}/simage/simage.hpp" for the api
* Compile/link with {path}/simage/simage.cpp
* Note: For USB camera functionality, install libuvc for Linux or OpenCV for windows

## API Overview

**See the /test_apps directory for complete examples**

### Interleaved / Plaform

Read, resize, write

```
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

```
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

```
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

```
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

```
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

```
View view;
ViewGray gray;
ViewYUV yuv;

// ...

img::map_gray(view, gray); // RGBA to grayscale
img::map_yuv(yuv, view);   // YUYV to RGBA
```

Create custom image tranforms

```
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

```
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
```

### Planar / Channel

Make a view with up to 4 channels using a MemoryBuffer

```
namespace img = simage;


u32 width = 1080;
u32 height = 720;
u32 n_channels = 10;

auto buffer = img::create_buffer32(width * height * n_channels);

auto gray = img::make_view_1(width, height, buffer);

auto gray_alpha = img::make_view_2(width, height, buffer);

auto rgb = img::make_view_3(width, height, buffer);

auto rgba = img::make_view_4(width, height, buffer);

// ...

img::destroy_buffer(buffer);
```

Convert between platform view and channel view 

```
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

```
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

```
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

```
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

### Credits (dependencies)

* stb_image: Read, write, resize images
* libuvc: Webcam support - Linux (requires libusb-1.0)
* opencv: Wecam support - Windows
* SDL2: Rendering test application examples

What problem are you solving?
Can you state the problem clearly?
Have you experienced it yourself?
Can you define the problem narrowly?

Who is your customer?
how often do they have the problem?
How intense is the problem?
Are they willing to pay?
How easy are they to find?

Does your MVP actually solve the problem?

Which customers should you go after first?
* The easy ones
* The most desparate
* Who's business will go out of business without this?

Which customers should you run away from?
* Complaining
* Exploit

Should you discount your product?
* No

Analytics
* Mix panel
* 5 - 10 stats
* Metrics part of build spec

KPI
* Revenue
* Usage

Braistorm
* Features
* Bug fix
* Maintenance
* Tests
* Easy/medium/hard
* Write specs

