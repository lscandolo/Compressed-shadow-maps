#pragma once

constexpr int SUCCESS                       = int(  0);
constexpr int ERROR_GENERIC                 = int( -1);
constexpr int ERROR_FILE_NOT_FOUND          = int( -2);
constexpr int ERROR_READING_FILE            = int( -3);
constexpr int ERROR_RESOURCE_NOT_FOUND      = int( -4);
constexpr int ERROR_LOADING_RESOURCE        = int( -5);
constexpr int ERROR_INVALID_POINTER         = int( -6);
constexpr int ERROR_INCORRECT_NAME          = int( -7);
constexpr int ERROR_INVALID_PARAMETER       = int( -8);
constexpr int ERROR_INVALID_OPERATION       = int( -9);
constexpr int ERROR_OPENGL_FAILURE          = int(-10);
constexpr int ERROR_UNINITIALIZED_OBJECT    = int(-11);
constexpr int ERROR_UNINITIALIZED_PARAMETER = int(-12);
constexpr int ERROR_OPTIX_FAILURE           = int(-13);
constexpr int ERROR_EXTERNAL_LIB            = int(-14);
constexpr int ERROR_EXTERNAL_PROCESS        = int(-15);
constexpr int ERROR_CUDA                    = int(-16);
