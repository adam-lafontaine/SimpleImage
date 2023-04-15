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

### ChannelView

* A view to multiple channels of planar image data

### Types

* Image: 4 byte RGBA interleaved image data
* View: A view to 4 byte image data
* ImageGray: 1 byte image data
* ViewGray: A view to 1 byte image data
* View1f32: Single channel float view
* View2f32, View3f32, View4f32: Multi-channel float view
* Buffer32: Allocates data for 4 byte pixel and float channel data
* Buffer8: Allocates data for 1 byte pixel data
* DeviceImage (CUDA): todo
* DeviceView (CUDA): todo
* DeviceBuffer (CUDA): todo

### Compile Instructions

* Copy the /simage directory to your project
* #include "{path}/simage/simage.hpp" for the api
* Compile/link with {path}/simage/simage.cpp

### API Overview

**Examples available in the /test_apps directory**

Read, resize, write.

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

### Credits

* Boost GIL: The inspiration to start this project
* stb_image: Read, write, resize images
* libuvc: Webcam support - Linux
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

