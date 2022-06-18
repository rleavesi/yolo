cmake_minimum_required(VERSION 3.5)
project(yolo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_INSTALL_PREFIX "/usr/local/ev_sdk")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(ADDITIONAL_RPATH "/opt/intel/openvino_2021/deployment_tools/ngraph/lib:\
                    /opt/intel/opencl:\
                    /opt/intel/openvino_2021/deployment_tools/inference_engine/external/hddl/lib:\
                    /opt/intel/openvino_2021/deployment_tools/inference_engine/external/gna/lib:\
                    /opt/intel/openvino_2021/deployment_tools/inference_engine/external/mkltiny_lnx/lib:\
                    /opt/intel/openvino_2021/deployment_tools/inference_engine/external/tbb/lib:\
                    /opt/intel/openvino_2021/deployment_tools/inference_engine/lib/intel64:\
                    /usr/local/ev_sdk/build:\
                    /usr/local/ev_sdk/3rd/cJSON/build")

#以下指令用于去除警告信息
add_definitions(-w)
#以下指令用于开启eigen的openmp优化
add_definitions(-fopenmp)
FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND") 
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}") 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}") 
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}") 
    set(CMAKE_SHARE_LINKER_FLAGS "${CMAKE_SHARE_LINKER_FLAGS} ${OpenMP_SHARE_LINKER_FLAGS}") 
endif()

include_directories(
    /opt/intel/openvino_2021/deployment_tools/inference_engine/include/
    /opt/intel/openvino_2021/deployment_tools/ngraph/include/
)
# 设置RPATH，使编译好的libji.so在运行时能够找到openvino_2021的依赖库
# 设置RPATH，使编译好的libji.so在运行时能够找到依赖库
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
if (DEFINED ADDITIONAL_RPATH)
    set(CMAKE_INSTALL_RPATH "${ADDITIONAL_RPATH}:${CMAKE_INSTALL_RPATH}")
endif ()
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

set(OpenCV_DIR "/usr")
find_package(OpenCV 4)

set(InferenceEngine_DIR "/opt/intel/openvino_2021/deployment_tools/inference_engine/share")
find_package(InferenceEngine)

add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/Yolo.cpp)
target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${InferenceEngine_LIBRARIES})
