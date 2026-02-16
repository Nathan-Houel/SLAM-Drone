cmake_minimum_required(VERSION 3.0.2)
project(tellovs LANGUAGES CXX)

# This example uses the library with the CMake INTERFACE library.
# Alternatively you can just copy and paste 'tello.hpp' into your include folder.
# In this case, omit the 'add_subdirectory' and 'target_link_libraries' below.

# add_subdirectory(.. bin)    # Path to the library's CMakeLists.txt.

set(SOURCES
    src/main.cpp
)

add_executable(tellovs ${SOURCES})
target_include_directories(tellovs PUBLIC include)
# target_link_libraries(tellovs tello)                    # Depends on the tello INTERFACE library (consisting only of the header)

if (NOT MSVC)
    target_link_libraries(tellovs pthread opencv_stitching opencv_alphamat opencv_aruco opencv_barcode opencv_bgsegm opencv_bioinspired opencv_ccalib opencv_dnn_objdetect opencv_dnn_superres opencv_dpm opencv_face opencv_freetype opencv_fuzzy opencv_hdf opencv_hfs opencv_img_hash opencv_intensity_transform opencv_line_descriptor opencv_mcc opencv_quality opencv_rapid opencv_reg opencv_rgbd opencv_saliency opencv_shape opencv_stereo opencv_structured_light opencv_phase_unwrapping opencv_superres opencv_optflow opencv_surface_matching opencv_tracking opencv_highgui opencv_datasets opencv_text opencv_plot opencv_ml opencv_videostab opencv_videoio opencv_viz opencv_wechat_qrcode opencv_ximgproc opencv_video opencv_xobjdetect opencv_objdetect opencv_calib3d opencv_imgcodecs opencv_features2d opencv_dnn opencv_flann opencv_xphoto opencv_photo opencv_imgproc opencv_core)    # pthread library must almost always be linked on linux
endif()

if (MSVC)
    target_compile_options(tellovs PUBLIC /std:c++14)       # C++14 standard required
else()
    target_compile_options(tellovs PUBLIC --std=c++14 -I/usr/include/opencv4/)      # C++14 standard required
endif()
