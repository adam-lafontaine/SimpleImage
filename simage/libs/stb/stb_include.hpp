#pragma once

#include "../../defines.hpp"


#define STBI_NO_GIF
#define STBI_NO_PSD
#define STBI_NO_PIC
#define STBI_NO_PNM
#define STBI_NO_HDR
#define STBI_NO_TGA

#define STBI_NO_JPEG

#ifndef SIMAGE_PNG
#define STBI_NO_PNG
#endif // !SIMAGE_PNG

#ifndef SIMAGE_BMP
#define STBI_NO_BMP
#endif // !SIMAGE_BMP

#ifdef SIMAGE_NO_SIMD
#define STBI_NO_SIMD
#endif // SIMAGE_NO_SIMD

#ifdef SIMD_ARM_NEON
#define STBI_NEON
#endif // SIMD_ARM_NEON





#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


#ifndef SIMAGE_NO_WRITE
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif // !SIMAGE_NO_WRITE


#ifndef SIMAGE_NO_RESIZE
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"
#endif // !SIMAGE_NO_RESIZE
