#include "common.h"

#include "helpers/OpenGLHelpers.h"
#include "helpers/FPCamera.h"
#include "helpers/CameraRecord.h"
#include "helpers/MeshData.h"
#include "helpers/LightData.h"
#include "helpers/ScopeTimer.h"
#include "helpers/Ini.h"

#include "gui/MeshGUI.h"
#include "gui/LightGUI.h"
#include "gui/FPCameraGUI.h"
#include "gui/RenderGUI.h"

#include "csm/CSMTechnique.h"


#include "managers/GLStateManager.h"
#include "managers/TextureManager.h"
#include "managers/GUIManager.h"
#include "managers/GLShapeManager.h"


#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <GLFW/glfw3.h>
#include <freeimage.h>
#include <tiny_obj_loader.h>

#include <iostream>
#include <random>

using namespace glm;

std::vector<Technique*> techniques = { new CSMTechnique };

bool display_gui = true;
bool full_screen = false;

RenderStats renderstats;

#define INIT_WIN_WIDTH  1280
#define INIT_WIN_HEIGHT 720

FPCamera fpcamera;
glm::vec3 fpcamera_speed(0.f, 0.f, 0.f);
glm::vec2 fpcamera_rotational_speed(0.f, 0.f);
bool cursor_lock = false;

using namespace GLHelpers;

static void resize_callback(GLFWwindow* window, int width, int height)
{
	DefaultRenderOptions().window_size = size2D(width, height);
	DefaultRenderOptions().output_size = size2D(width, height);
	DefaultRenderOptions().current_technique()->output_resize_callback(size2D(width, height));

	fpcamera.aspect = DefaultRenderOptions().output_size.width / float(DefaultRenderOptions().output_size.height);
	fpcamera.jitter_plane = glm::ivec2(DefaultRenderOptions().output_size.width, DefaultRenderOptions().output_size.height);
}

static void camera_key_callback(int key, int scancode, int action, int mods)
{
	if (action != GLFW_PRESS && action != GLFW_RELEASE) return;

	if (key == GLFW_KEY_W) fpcamera_speed.z = action == GLFW_PRESS ?  1.0f : 0.0f;
	if (key == GLFW_KEY_S) fpcamera_speed.z = action == GLFW_PRESS ? -1.0f : 0.0f;
	if (key == GLFW_KEY_D) fpcamera_speed.x = action == GLFW_PRESS ?  1.0f : 0.0f;
	if (key == GLFW_KEY_A) fpcamera_speed.x = action == GLFW_PRESS ? -1.0f : 0.0f;
	if (key == GLFW_KEY_E) fpcamera_speed.y = action == GLFW_PRESS ?  1.0f : 0.0f;
	if (key == GLFW_KEY_Q) fpcamera_speed.y = action == GLFW_PRESS ? -1.0f : 0.0f;

}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	size2D& window_size = DefaultRenderOptions().window_size;

	if (GUIManager::instance().key_callback(key, scancode, action, mods)) return; // If GUIManager uses the input then skip passing it to the rest of the systems

	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GLFW_TRUE);

	if (key == GLFW_KEY_F) {
		if (action == GLFW_PRESS) cursor_lock = !cursor_lock;
		
		if (cursor_lock) {
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN | GLFW_CURSOR_DISABLED);
			glfwSetCursorPos(window, window_size.width/2.0, window_size.height/2.0);
		} else {
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
	}

	if (key == GLFW_KEY_M && action == GLFW_PRESS) {
		display_gui = !display_gui;
	}

	camera_key_callback(key, scancode, action, mods);
} 

static void char_callback(GLFWwindow* window, unsigned int c)
{
	if (GUIManager::instance().char_callback(c)) return;
}

static void cursor_callback(GLFWwindow* window, double xpos, double ypos)
{
	size2D& window_size = DefaultRenderOptions().window_size;

	double half_width  = window_size.width  / 2.0;
	double half_height = window_size.height / 2.0;

	if (!cursor_lock) return;

	if (xpos == half_width && ypos == half_height) return;

	fpcamera_rotational_speed = glm::vec2(float(xpos - half_width), float(ypos - half_height));
	glfwSetCursorPos(window, half_width, half_height);

	return;
}

int main()
{
	//dl_main();
	//return 0;

	//////////////// 
	FreeImage_Initialise();

	////////////////
	if (!glfwInit())
	{
		std::cerr << "Error initializing GLFW." << std::endl;
		return -1;
	}
	std::cout << "Initialized glfw." << std::endl;
	glfwSetErrorCallback(errorCallbackGLFW);

	////////////////

	////////////////
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
	glfwWindowHint(GLFW_DOUBLEBUFFER, true);
	glfwWindowHint(GLFW_DEPTH_BITS, 0);

	GLFWwindow* window = glfwCreateWindow(INIT_WIN_WIDTH, INIT_WIN_HEIGHT, "Compressed shadow maps demo", NULL, NULL);
	if (!window)
	{
		std::cerr << "Error creating window." << std::endl;
		glfwTerminate();
		return -1;
	}
	std::cout << "Created window." << std::endl;
	size2D& output_size = DefaultRenderOptions().output_size;
	size2D& window_size = DefaultRenderOptions().window_size;

	output_size = size2D(INIT_WIN_WIDTH, INIT_WIN_HEIGHT);
	window_size = size2D(INIT_WIN_WIDTH, INIT_WIN_HEIGHT);

	////////////////
	glfwSetWindowSizeCallback(window, resize_callback);
	glfwSetKeyCallback(window, key_callback);
	glfwSetCharCallback(window, char_callback);
	glfwSetCursorPosCallback(window, cursor_callback);
	glfwMakeContextCurrent(window);
	glfwSwapInterval(0);

	////////////////
	GLenum glewInitResult = glewInit();
	if (glewInitResult != GL_NO_ERROR) {
		std::cerr << "Error creating window." << std::endl;
		glfwTerminate();
		return -1;
	}
	std::cout << "Initialized glew." << std::endl;


	//////////////// Default render options
	for (auto& o : DefaultRenderOptions().debug_var) o = 0.5f;

	////////////////
	// During init, enable debug output
#ifdef _DEBUG
	glEnable(GL_DEBUG_OUTPUT);
	glDebugMessageCallback(DebugCallbackGL, 0);
#endif

	////////////////
	MeshData  mesh;
	LightData lights;

	fpcamera.aspect = output_size.width / float(output_size.height);
	fpcamera.jitter_plane = ivec2(output_size.width, output_size.height);
	fpcamera.mov_speed = 0.25f;
	fpcamera.rot_speed = 0.001f;

	Ini ini;
	std::string tecname;
	ini.load("data/inis/default.ini", IniInfo(&mesh, &fpcamera, &lights, &tecname));

	////////////////
	GLStateManager::instance().resetState(true);

	for (int i = 0; i < techniques.size(); ++i)
	{
		DefaultRenderOptions().add_technique(techniques[i]);
	}

	////////////////
	GUIManager::instance().initialize(window);
	GUIManager::instance().addGUI("1_rendergui", std::shared_ptr<BaseGUI>(new RenderGUI(&DefaultRenderOptions(), &renderstats)));
	GUIManager::instance().addGUI("2_camfpgui", std::shared_ptr<BaseGUI>(new FPCameraGUI(&fpcamera)));
	GUIManager::instance().addGUI("3_meshgui", std::shared_ptr<BaseGUI>(new MeshGUI(&mesh)));

	////////////////
	GLShapeManager::instance().initialize();

	bool first_frame = true;

	Camera* camera = &fpcamera;

	////////////////
	std::cout << "Starting main loop." << std::endl;
	check_opengl();
	while (!glfwWindowShouldClose(window))
	{
		PROFILE_SCOPE("Main loop")

		glfwMakeContextCurrent(window);

		lights.updateGLResources();
		mesh.materialdata.updateGLResources();

		DefaultRenderOptions().new_frame();

		if (first_frame) 
		{
			DefaultRenderOptions().request_technique(techniques[0]->name());
			first_frame = false;
		}

		if (DefaultRenderOptions().get_flag(RenderOptions::FLAG_DIRTY_TECHNIQUE)) 
		{
			Technique::SetupData ts;
			ts.output_size = DefaultRenderOptions().output_size;
			ts.mesh      = &mesh;
			ts.lightdata = &lights;
			ts.camera    = camera;
			DefaultRenderOptions().switch_to_requested_technique();
			DefaultRenderOptions().initialize_technique(ts);
			GUIManager::instance().setTechnique(DefaultRenderOptions().current_technique());
			DefaultRenderOptions().reset_flag(RenderOptions::FLAG_DIRTY_TECHNIQUE);
		}

		//// Update vsync options
		glfwSwapInterval(DefaultRenderOptions().vsync ? 1 : 0);

		//// Swap current and previous depth tex
		//std::swap(depthtex, prev_depthtex);
		//fbo.AttachDepthTexture(depthtex);

		{ PROFILE_SCOPE("Render")
			DefaultRenderOptions().current_technique()->output_resize_callback(output_size);
			DefaultRenderOptions().current_technique()->frame_prologue();
			DefaultRenderOptions().current_technique()->frame_render();
			DefaultRenderOptions().current_technique()->frame_epilogue();
		}

		//// Draw GUI
		fpcamera.copied = false;
		if (display_gui) GUIManager::instance().draw();

		//// Swap buffers
		glfwSwapBuffers(window);

		//// Update timings
		TimingManager::instance().endFrame();

		TimingStats s = TimingManager::instance().getTimingStats("Render");
		renderstats.time_cpu_ms  = s.cpu_time;
		renderstats.time_gl_ms   = s.gl_time;
		renderstats.time_cuda_ms = s.cuda_time;

		//// Get input 
		glfwPollEvents();

		//// Update camera
		s = TimingManager::instance().getTimingStats("Main loop");
		float time = std::max(s.gl_time, std::max(s.cpu_time, s.cuda_time));
		
		if (fpcamera_rotational_speed != vec2(0.f, 0.f) || fpcamera_speed != vec3(0.f, 0.f, 0.f)) {
			fpcamera.update(fpcamera_rotational_speed, fpcamera_speed * time * 0.001f);
			fpcamera_rotational_speed = vec2(0.f, 0.f);
			fpcamera.moved = true;
		} else {
			fpcamera.update(fpcamera_rotational_speed, fpcamera_speed);
			fpcamera.moved = false;
		}



	}

	for (auto t : techniques) t->destroy();

	// Clear textures before leaving context
	TextureManager::instance().clear();
	GUIManager::instance().finalize(window);

	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}

