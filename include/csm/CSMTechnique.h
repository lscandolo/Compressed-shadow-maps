#pragma once

#include "common.h"
#include "techniques/Technique.h"
#include "techniques/rendersteps/OpenGLChannelCopier.h"
#include "techniques/rendersteps/OpenGLBackgroundRenderer.h"
#include "helpers/OpenGLHelpers.h"

#include "csm/QuadtreeGPUCompression.h"
#include "csm/BoundsRenderer.h"
#include "csm/Tiler.h"

class CSMTechnique : public Technique
{
public:

	CSMTechnique();
	
	virtual int  initial_setup(const SetupData& ts)   override;
	virtual void output_resize_callback(size2D window_size) override;
	virtual void destroy()           override;

	virtual void frame_prologue()    override;
	virtual void frame_render()      override;
	virtual void frame_epilogue()    override;

	virtual void reload_programs()   override;
	virtual std::string name()       override;

	virtual void draw_gui(int order) override;

private:

	MeshData * mesh;
	Camera   * cam;
	GLHelpers::FramebufferObject fbo;
	GLHelpers::TextureObject2D   colortex, depthtex;
	GLHelpers::ProgramObject     render_depth_program, gbuffer_program, shade_program, eval_shadow_program;
	bool dirty_programs;
	bool dirty_csm;

	int compression_option;

	BoundsRenderer boundsrenderer;
	QuadtreeGPUCompression csm;
	OpenGLBackgroundRenderer backgroundrenderer;

	OpenGLChannelCopier copier;

	int tilesize;
	int csmsize;

	size2D size;

	void recreate_textures(size2D window_size);
	void create_csm();
	void evaluate_shadow_cuda();
	void evaluate_shadow_opengl();

	bool merge_csm;
	bool relative_csm;
	float light_near;
	float light_far;
	bool light_is_dir;
	glm::vec3 light_dir;
	glm::vec3 light_spotpos;
	float     light_spotangle;
	float     light_intensity;
	glm::vec3 light_color;

	int      eval_pcf_size;
	bool     eval_hierarchical;
	bool     eval_cuda;

	bool outdatedCSM;

	CompressionConfig     csmconfig;
	Tiler::Input          tinput;
	Tiler::TileParameters tparam;
	CompressionStats      cstats;

	GLHelpers::TextureObject2D shadowtex, wpostex, wdpostex, wnortex, tcmattex;
	CudaOpenGLResource shadowtex_cuda, wpostex_cuda, wdpostex_cuda, wnortex_cuda;


};