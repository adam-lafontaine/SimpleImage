#include "src/simage_impl.cpp"
#include "libs/stb/stb_simage.cpp"
#include "src/util/execute.cpp"

#ifndef SIMAGE_NO_USB_CAMERA

#include "libs/uvc/uvc_simage.cpp"
#include "libs/pthread-win32/pthread.c"



#ifdef _WIN32

//#include "libs/opencv/opencv_simage.cpp"

#else

#include "libs/uvc/uvc_simage.cpp"

#endif // _WIN32

#endif // !SIMAGE_NO_USB_CAMERA