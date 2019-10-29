-- Premake file
--
local path = require ("path")

-- Small hack to know whether this is linux or windows
local isWindows = (package.config:sub(1,1) == '\\');

-- Function to obtaim latest SDK version to use in win10 VS
function os.winSdkVersion()
    local reg_arch = iif( os.is64bit(), "\\Wow6432Node\\", "\\" )
    local sdk_version = os.getWindowsRegistry( "HKLM:SOFTWARE" .. reg_arch .."Microsoft\\Microsoft SDKs\\Windows\\v10.0\\ProductVersion" )
    if sdk_version ~= nil then return sdk_version end
end

-- Function to add " at beginning and end of string
function quot(s)
	return '"'..s..'"'
end

workspace "Compressed shadow maps"
    location "build/"
    platforms { "x64" }
    configurations { "release", "debug" }
	startproject "Compressed shadow maps"
    
	language "C++"
	cppdialect "C++17"

    flags "NoPCH"
    vectorextensions "AVX"
    flags "MultiProcessorCompile"

    objdir "build/%{cfg.buildcfg}-%{cfg.platform}-%{cfg.toolset}"
    targetsuffix "-%{cfg.buildcfg}-%{cfg.platform}-%{cfg.toolset}"

    newoption {
        trigger = "toolset",
        description = "Select toolset on Linux / MacOS",
        allowed = {
            { "gcc", "Use GCC" },
            { "clang", "Use Clang" }
        }
    };

    -- Workaround empty "toolset"
    filter "system:linux or system:macos"
        toolset( _OPTIONS["toolset"] or "gcc" );
    filter "system:windows"
        toolset( "msc" );
    filter "*"

    -- default compiler flags
    filter "toolset:gcc or toolset:clang"
        linkoptions { "-pthread" }
        buildoptions { "-march=native", "-Wall", "-pthread" }

    filter "toolset:msc"
        defines { "_CRT_SECURE_NO_WARNINGS=1", "NOMINMAX" }

    filter "action:vs2015"
        buildoptions { "/utf-8" }
    
    filter "action:vs2017"
        buildoptions { "/utf-8" }
        --buildoptions { "/std:c++latest" }
    
    filter "*"

    -- default libraries
    filter "system:linux"
        links "GL"
        links "GLU"

    filter "system:windows"
        links "OpenGL32"

        libdirs "%{wks.location}/freeglut/lib/x64"
        includedirs "%{wks.location}/freeglut/include"

    filter "*"

    -- default outputs
    filter "kind:StaticLib"
        targetdir "lib/"

    filter { "kind:ConsoleApp", "configurations:release" }
        targetdir "bin/release"
        targetextension ".exe"
    filter { "kind:ConsoleApp", "configurations:debug" }
        targetdir "bin/debug"
        targetextension ".exe"
    
    filter "*"

    --configurations
    configuration "debug"
        symbols "On" -- new, but doesn't work?
        --flags "Symbols" -- deprecated, but works
        defines { "_DEBUG=1" }

    configuration "release"
        optimize "On"
        defines { "NDEBUG=1" }
        flags "LinkTimeOptimization"

    configuration "*"

project( "imgui" )
    kind "StaticLib"
    location "build/imgui"

    filter {"system:windows", "action:vs*"}
        systemversion(os.winSdkVersion() .. ".0")
	filter "*"

    local imgui_source_files  = { "external/imgui-1.65/*.cpp" }
    local imgui_include_files = { "external/imgui-1.65/*.h" }

    filter "system:linux"
        includedirs { }
    filter "system:windows"
        includedirs { "external/win/glfw-3.2.1/include" }
	filter "*"

    filter "system:linux"
        includedirs {  }
    filter "system:windows"
        includedirs { "external/win/glew-2.1.0/include" }
	filter "*"

	files { imgui_source_files, imgui_include_files }


project( "meshoptimizer" )
    kind "StaticLib"
    location "build/meshoptimizer"

    filter {"system:windows", "action:vs*"}
        systemversion(os.winSdkVersion() .. ".0")
	filter "*"

    local meshopt_source_files  = { "external/meshoptimizer/src/*.cpp" }
    local meshopt_include_files = { "external/meshoptimizer/src/*.h" }
    files { meshopt_source_files, meshopt_include_files}

 
project( "Compressed shadow maps" )
    kind "ConsoleApp"
    location "build/csm"

    filter "system:windows"
        debugdir "%{wks.location}/.." 
		
    
    filter {"system:windows", "action:vs*"}
        systemversion(os.winSdkVersion() .. ".0")

		
	luadir = '%{wks.location}/..'
		
	local source_files = { "src/*.cpp" }
	local include_files = { "include/*.h", "include/*.inl" }
	
	local file_dirs = { "helpers", "managers", "gui", "techniques", "techniques/rendersteps", "csm" } 

	for key,value in pairs(file_dirs) do
		table.insert(include_files, "include/"..value.."/*.h")
		table.insert(include_files, "include/"..value.."/*.inl")
		table.insert(source_files , "src/"    ..value.."/*.cpp")
	end

	local cuda_source_files  = { "src/cuda/*.cu", "src/cuda/csm/*.cu"}
	local cuda_include_files = { "include/cuda/*.cuh", "include/cuda/*.inl" }

    files { source_files, include_files, cuda_source_files, cuda_include_files }

    local include_dirs = { "include" }

	local external_include_dirs = { 
									"external/glm-0.9.9.2/include",
									"external/imgui-1.65",
									"external/tinyobjloader",
									"external/meshoptimizer/src",
									"external/json",
								  }
								  
    includedirs { include_dirs, external_include_dirs }
    links { "imgui", "meshoptimizer" }
									
	local shader_dirs  = { "shaders" }
	local shader_files = { "shaders/*.frag", "shaders/*.vert", "shaders/*.geom", "shaders/*.comp", "shaders/*.glsl", "shaders/Reusable/*.glsl" }

	files  { shader_files } 


	---------------------------------- GLFW
    filter "system:linux"
        libdirs     {}
		includedirs {}
        links       { os.findlib("glfw3") }
    filter "system:windows"
        libdirs     { "external/win/glfw-3.2.1/lib-vc2015" }
        includedirs { "external/win/glfw-3.2.1/include" }
        links       { "glfw3dll" }
        postbuildcommands { '{copy} "%{wks.location}/../external/win/glfw-3.2.1/lib-vc2015/glfw3.dll"  "%{cfg.targetdir}"'}
	filter "*"

    
	---------------------------------- GLEW
    filter "system:linux"
        libdirs     {}
		includedirs {}
        links       { os.findlib("glew") }
    filter "system:windows"
        libdirs     { "external/win/glew-2.1.0/lib/Release/x64" }
        includedirs { "external/win/glew-2.1.0/include" }
        links       { "glew32" }
		postbuildcommands { '{copy} "%{wks.location}/../external/win/glew-2.1.0/bin/Release/x64/glew32.dll"   "%{cfg.targetdir}"' }
	filter "*"

    
	---------------------------------- FREEIMAGE
    filter "system:linux"
        libdirs     {}
		includedirs {}
        links       { os.findlib("freeimage") }
    filter "system:windows"
        libdirs     { "external/win/FreeImage-3.18.0/Dist/x64" }
		includedirs { "external/win/FreeImage-3.18.0/Dist/x64" }
        links       { "FreeImage" }
		postbuildcommands { '{copy} "%{wks.location}/../external/win/FreeImage-3.18.0/Dist/x64/FreeImage.dll" "%{cfg.targetdir}"' }
	filter "*"
	

	---------------------------------- CUDA
	cuda_path = os.getenv("CUDA_PATH")
    filter "system:linux"
        libdirs     {}
		includedirs {}
        links       { os.findlib("cuda") }
    filter "system:windows"
        libdirs     { cuda_path .. "/lib/x64" }
		includedirs { cuda_path .. "/include" }
        links       { "cuda", "cudart", "cublas", "curand" }
	filter "*"

		
	CUDA_CC = "nvcc"
	CUDA_FLAGS = "--use_fast_math"
	CUDA_SOURCE_DIR  = os.getcwd().."/src/cuda"
		

	CUDA_WIN_FLAGS_RELEASE  = '-Xcompiler "/MD"'
	CUDA_WIN_FLAGS_DEBUG    = '-Xcompiler "/MDd"'

	------------- Custom build commands for cuda files (debug)
	filter { 'files:src/cuda/**.cu' }
	   -- A message to display while this build step is running (optional)
	   buildmessage 'Compiling %{file.relpath}'

	   -- One or more commands to run (required)
	   filter {'configurations:release', 'files:src/cuda/**.cu'}
		   buildcommands { '%{CUDA_CC} %{CUDA_FLAGS} %{CUDA_WIN_FLAGS_RELEASE} -c -I"%{cuda_path}/include" -I"%{wks.location}/../include" -I"%{wks.location}/external/glm-0.9.9.2/include" "%{file.path}"  -o "%{cfg.objdir}/%{file.basename}.cu.obj"' }
	   filter {'configurations:debug', 'files:src/cuda/**.cu'}
		   buildcommands { '%{CUDA_CC} %{CUDA_FLAGS} %{CUDA_WIN_FLAGS_DEBUG} -c -I"%{cuda_path}/include" -I"%{wks.location}/../include" -I"%{wks.location}/external/glm-0.9.9.2/include" "%{file.path}"  -o "%{cfg.objdir}/%{file.basename}.cu.obj"' }
	   filter {'files:src/cuda/**.cu'}

	   -- One or more outputs resulting from the build (required)
	   buildoutputs { '%{cfg.objdir}/%{file.basename}.cu.obj' }

	   -- One or more additional dependencies for this build command (optional)
	   buildinputs { '%{wks.location}/../include/cuda/*.cuh' }
    filter "*"
	
	
	files {cuda_source_files, cuda_include_files }
    
--EOF
