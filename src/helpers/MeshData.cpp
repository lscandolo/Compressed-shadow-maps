#include "common.h"
#include "helpers/MeshData.h"
#include "managers/TextureManager.h"

#include "meshoptimizer.h"

#include <iostream>
#include <algorithm>


#ifdef _WIN32

#else
#include <libgen.h>
#endif

using namespace GLHelpers;

struct index_info_t
{
	int position_index;
	int normal_index;
	int texcoord_index;
	int material_index;
};

bool compareIndices(const index_info_t& a, const index_info_t& b)
{
	if (a.normal_index     != b.normal_index)     return a.normal_index   < b.normal_index;
	if (a.texcoord_index   != b.texcoord_index)   return a.texcoord_index < b.texcoord_index;
	if (a.position_index   != b.position_index)   return a.position_index < b.position_index;
											      return a.material_index < b.material_index;
}

static Material translateMaterial(tinyobj::material_t tm, std::string filepath)
{
	Material m;

	m.name = tm.name;

	std::vector< std::reference_wrapper<std::string>> texnames {
	  m.alpha_texname,     m.ambient_texname,      m.bump_texname,
	  m.diffuse_texname,   m.displacement_texname, m.emissive_texname,
	  m.metallic_texname,  m.normal_texname,       m.reflection_texname,
	  m.roughness_texname, m.sheen_texname,        m.specular_highlight_texname };

	std::vector< std::reference_wrapper<glm::uvec2>> texhandles {
	  m.gl.alpha_texhandle,     m.gl.ambient_texhandle,      m.gl.bump_texhandle,
	  m.gl.diffuse_texhandle,   m.gl.displacement_texhandle, m.gl.emissive_texhandle,
	  m.gl.metallic_texhandle,  m.gl.normal_texhandle,       m.gl.reflection_texhandle,
	  m.gl.roughness_texhandle, m.gl.sheen_texhandle,        m.gl.specular_highlight_texhandle };

	std::vector< std::reference_wrapper<std::string>> tinyobj_texnames{
	  tm.alpha_texname,     tm.ambient_texname,      tm.bump_texname,
	  tm.diffuse_texname,   tm.displacement_texname, tm.emissive_texname,
	  tm.metallic_texname,  tm.normal_texname,     	 tm.reflection_texname,
	  tm.roughness_texname, tm.sheen_texname,        tm.specular_highlight_texname };

	std::set<std::string> materialLoadedTextures;

	for (int i = 0; i < texnames.size(); ++i)
	{
		std::string& texname = texnames[i];
		glm::uvec2&  texhandle = texhandles[i];
		std::string& tinyobj_texname = tinyobj_texnames[i];
	
		texhandle = glm::uvec2(0, 0);
		
		if (tinyobj_texname.empty()) continue;

		texname = filepath + tinyobj_texname;

		if ( TextureManager::instance().loadTexture(texname, true) != SUCCESS ) {
			std::cerr << "Unable to load texture " << texname << std::endl;
			texname.clear();
			continue;
		}

		// Make texture resident
		//TextureData& t = TextureManager::instance().getTexture(texname);
		//if (!t.gltex->isResident()) t.gltex->makeResident();
		//GLuint64 h = t.gltex->getHandle();
		//texhandle = reinterpret_cast<glm::uvec2&>(h);
		//if (!h && !materialLoadedTextures.count(texname)) std::cerr << "Unable to make texture resident " << texname << std::endl;

		materialLoadedTextures.insert(texname);
	}

	for (int i = 0; i < 3; ++i) {
		m.gl.ambient[i]       = tm.ambient[i];
		m.gl.diffuse[i]       = tm.diffuse[i];
		m.gl.specular[i]      = tm.specular[i];
		m.gl.transmittance[i] = tm.transmittance[i];
		m.gl.emission[i]      = tm.emission[i];
	}

	m.gl.shininess = tm.shininess;
	m.gl.metallic  = tm.metallic;
	m.gl.roughness = tm.roughness;
	m.gl.sheen     = tm.sheen;

	return m;
}

static void computeTangents(const std::vector<glm::vec3>& vertex, const std::vector<glm::vec3>& normal,
	const std::vector<glm::vec2>& texcoord, const std::vector<unsigned int>& indices, std::vector<glm::vec4>& tangent)
{

	std::vector<glm::vec3> tan1(vertex.size(), glm::vec3(0,0,0));
	std::vector<glm::vec3> tan2(vertex.size(), glm::vec3(0,0,0));

	tangent.resize(vertex.size());

	if (texcoord.size() < vertex.size()) return;

	for (size_t a = 0; a < indices.size()/3; a++)
	{
		const unsigned int i1 = indices[3*a+0];
		const unsigned int i2 = indices[3*a+1];
		const unsigned int i3 = indices[3*a+2];

		const glm::vec3& v1 = vertex[i1];
		const glm::vec3& v2 = vertex[i2];
		const glm::vec3& v3 = vertex[i3];

		const glm::vec2& w1 = texcoord[i1];
		const glm::vec2& w2 = texcoord[i2];
		const glm::vec2& w3 = texcoord[i3];

		const float x1 = v2.x - v1.x;
		const float x2 = v3.x - v1.x;
		const float y1 = v2.y - v1.y;
		const float y2 = v3.y - v1.y;
		const float z1 = v2.z - v1.z;
		const float z2 = v3.z - v1.z;

		const float s1 = w2.x - w1.x;
		const float s2 = w3.x - w1.x;
		const float t1 = w2.y - w1.y;
		const float t2 = w3.y - w1.y;

		float r = 1.0F / (s1 * t2 - s2 * t1);

		if (s1 * t2 - s2 * t1 == 0.f) r = 1.f;
		
		glm::vec3 sdir( (t2 * x1 - t1 * x2) * r, 
						(t2 * y1 - t1 * y2) * r,
						(t2 * z1 - t1 * z2) * r);
		
		glm::vec3 tdir( (s1 * x2 - s2 * x1) * r, 
					    (s1 * y2 - s2 * y1) * r,
			            (s1 * z2 - s2 * z1) * r);

		tan1[i1] += sdir;
		tan1[i2] += sdir;
		tan1[i3] += sdir;

		tan2[i1] += tdir;
		tan2[i2] += tdir;
		tan2[i3] += tdir;

	}

	for (size_t a = 0; a < vertex.size(); a++)
	{

		const glm::vec3& n = normal[a];
		const glm::vec3& t = tan1[a];

		// Gram-Schmidt orthogonalize
		glm::vec3 tanxyz = glm::length(t) < 1e-6f ? glm::vec3(0.f, 0.f, 0.f) : glm::normalize(t - n * glm::dot(n, t));

		// Calculate handedness
		float tanw = (glm::dot(glm::cross(n, t), tan2[a]) < 0.0F) ? -1.0F : 1.0F;

		tangent[a] = glm::vec4(tanxyz, tanw);
	}

}

MeshDataGL::MeshDataGL() :
	vao(create_object<VertexArrayObject>()),
	posbuffer(create_object<BufferObject>()),
	normalbuffer(create_object<BufferObject>()),
	texcoordbuffer(create_object<BufferObject>()),
	tangentbuffer(create_object<BufferObject>()),
	colorbuffer(create_object<BufferObject>()),
	matidbuffer(create_object<BufferObject>()),
	indexbuffer(create_object<BufferObject>())
{
}

void MeshData::makeGLTexturesResident()
{
	for (Material& m : materialdata.materials) {
	
		std::vector< std::reference_wrapper<std::string>> texnames {
			 m.alpha_texname,     m.ambient_texname,      m.bump_texname,
			 m.diffuse_texname,   m.displacement_texname, m.emissive_texname,
			 m.metallic_texname,  m.normal_texname,       m.reflection_texname,
			 m.roughness_texname, m.sheen_texname,        m.specular_highlight_texname };

		std::vector< std::reference_wrapper<glm::uvec2>> texhandles{
			m.gl.alpha_texhandle,     m.gl.ambient_texhandle,      m.gl.bump_texhandle,
			m.gl.diffuse_texhandle,   m.gl.displacement_texhandle, m.gl.emissive_texhandle,
			m.gl.metallic_texhandle,  m.gl.normal_texhandle,       m.gl.reflection_texhandle,
			m.gl.roughness_texhandle, m.gl.sheen_texhandle,        m.gl.specular_highlight_texhandle };

		for (int i = 0; i < texnames.size(); ++i)
		{
			std::string& texname   = texnames[i];
			glm::uvec2&  texhandle = texhandles[i];

			if (texname.empty() || !TextureManager::instance().hasTexture(texname)) continue;

			TextureData& t = TextureManager::instance().getTexture(texname);

			if (!t.gltex->isResident()) t.gltex->makeResident();

			if (t.gltex->isResident()) {
				GLuint64 h = t.gltex->getHandle();
				texhandle = reinterpret_cast<glm::uvec2&>(h);
				materialdata.setDirty();
			}
		}
	}
}

void MeshData::makeGLTexturesNotResident()
{
	for (Material& m : materialdata.materials) {

		std::vector< std::reference_wrapper<std::string>> texnames{
			 m.alpha_texname,     m.ambient_texname,      m.bump_texname,
			 m.diffuse_texname,   m.displacement_texname, m.emissive_texname,
			 m.metallic_texname,  m.normal_texname,       m.reflection_texname,
			 m.roughness_texname, m.sheen_texname,        m.specular_highlight_texname };

		std::vector< std::reference_wrapper<glm::uvec2>> texhandles{
			m.gl.alpha_texhandle,     m.gl.ambient_texhandle,      m.gl.bump_texhandle,
			m.gl.diffuse_texhandle,   m.gl.displacement_texhandle, m.gl.emissive_texhandle,
			m.gl.metallic_texhandle,  m.gl.normal_texhandle,       m.gl.reflection_texhandle,
			m.gl.roughness_texhandle, m.gl.sheen_texhandle,        m.gl.specular_highlight_texhandle };

		for (int i = 0; i < texnames.size(); ++i)
		{
			std::string& texname = texnames[i];
			glm::uvec2&  texhandle = texhandles[i];
			if (texname.empty() || !TextureManager::instance().hasTexture(texname)) continue;

			TextureData& t = TextureManager::instance().getTexture(texname);
			if (t.gltex->isResident()) t.gltex->makeNotResident();

			if (!t.gltex->isResident()) {
				texhandle = glm::uvec2(0, 0);
				materialdata.setDirty();
			}
		}
	}
}


int MeshData::load_obj(std::string fullfilename, float scale, bool optimize)
{
	////////////////////// Clear previous resources
	clear();

	source_filename = fullfilename;

	////////////////////// Load mesh file to CPU
	std::string warnings;
	std::string errors;

	std::string filename;
	std::string filepath;



#ifdef _WIN32
	char fdir[1024];
	char fname[128];
	char fext[128];
	_splitpath_s(fullfilename.c_str(), nullptr, 0, fdir, 1024, fname, 128, fext, 128);
	filename = std::string(fname) + std::string(fext);
	filepath = std::string(fdir);
#else
	filename = std::string(basename(fullfilename.c_str()));
	filepath = std::string(dirname(fullfilename.c_str())) + std::string("/");
#endif

	std::vector<tinyobj::shape_t>    tinyobj_shapes;
	std::vector<tinyobj::material_t> tinyobj_materials;
	tinyobj::attrib_t                tinyobj_attributes;


	bool success = tinyobj::LoadObj(&tinyobj_attributes, &tinyobj_shapes, &tinyobj_materials, &warnings, &errors, fullfilename.c_str(), filepath.c_str(), true);

	if (!success)
	{
		std::cerr << "OBJ Loader errors: \n" << errors << std::endl;
		return ERROR_READING_FILE;
	}

#ifdef _DEBUG
	std::cerr << "OBJ Loader warnings: \n" << warnings << std::endl;
#endif


	////////////////////// Copy materials
	std::vector<Material>& materials = materialdata.materials;
	
	if (tinyobj_materials.size() == 0)
	{
		tinyobj::material_t m;
		for (int i = 0; i < 3; ++i) {
			m.ambient[i] = 1.0f;
			m.diffuse[i] = 1.0f;
			m.emission[i] = 0.f;
			m.specular[i] = 0.f;
			m.transmittance[i] = 0.f;
		}
		m.shininess = 0.f;
		m.ior = 1.f;
		m.dissolve = 1.f;

		m.roughness = m.metallic = m.sheen = m.clearcoat_roughness = m.clearcoat_thickness = m.anisotropy = m.anisotropy_rotation = 0.f;
		tinyobj_materials.push_back(m);
	}
	
	
	materialdata.materials.resize(tinyobj_materials.size());

	for (size_t i = 0; i < materials.size(); ++i)
	{
		materials[i] = translateMaterial(tinyobj_materials[i], filepath);
	}
	materialdata.setDirty();

	////////////////////// Flatten vertices
	int currentIndex = 0;
	std::map<index_info_t, int, bool(*)(const index_info_t&, const index_info_t&)> indexMap(compareIndices);
	 
	bbox.clear();
	bsphere.clear();

	index_info_t index_info;

	submeshes.resize(tinyobj_shapes.size());

	for (size_t shape_idx = 0; shape_idx < tinyobj_shapes.size(); ++shape_idx)
	{
		Submesh& smesh = submeshes[shape_idx];
		tinyobj::shape_t& s = tinyobj_shapes[shape_idx];
		smesh.name = s.name;
		smesh.start_index = static_cast<int>(indices.size());
		smesh.lowest_index_value  = INT_MAX;
		smesh.largest_index_value = 0;

		for (size_t index_idx = 0; index_idx < s.mesh.indices.size(); index_idx++) {
			const tinyobj::index_t& tindex = s.mesh.indices[index_idx];

			index_info.material_index = index_idx/3 < s.mesh.material_ids.size() ? std::max(0, s.mesh.material_ids[index_idx/3]) : 0;
			index_info.position_index = tindex.vertex_index;
			index_info.normal_index   = tindex.normal_index;
			index_info.texcoord_index = tindex.texcoord_index;

			auto& it = indexMap.find(index_info);
			int flat_index = 0;
			if (it == indexMap.end()) {
				indexMap[index_info] = currentIndex;
				flat_index = currentIndex;
				positions.push_back(glm::vec3(tinyobj_attributes.vertices [tindex.vertex_index   * 3 + 0], tinyobj_attributes.vertices [tindex.vertex_index   * 3 + 1], tinyobj_attributes.vertices[tindex.vertex_index * 3 + 2]));
				if (tindex.normal_index >= 0)   normals.push_back  (glm::vec3(tinyobj_attributes.normals  [tindex.normal_index   * 3 + 0], tinyobj_attributes.normals  [tindex.normal_index   * 3 + 1], tinyobj_attributes.normals [tindex.normal_index * 3 + 2]));
				if (tindex.texcoord_index >= 0) texcoords.push_back(glm::vec2(tinyobj_attributes.texcoords[tindex.texcoord_index * 2 + 0], tinyobj_attributes.texcoords[tindex.texcoord_index * 2 + 1]));
				colors.push_back   (glm::vec3(tinyobj_attributes.colors   [tindex.vertex_index   * 3 + 0], tinyobj_attributes.colors   [tindex.vertex_index   * 3 + 1], tinyobj_attributes.colors  [tindex.vertex_index * 3 + 2]));
				matIds.push_back(index_info.material_index);
				bbox.extend(positions.back());
				currentIndex++;
			} else {
				flat_index = it->second;
			}
			
			indices.push_back(flat_index);
			smesh.largest_index_value = std::max(flat_index, smesh.largest_index_value);
			smesh.lowest_index_value = std::min(flat_index, smesh.lowest_index_value);
		}

		smesh.default_material_index = s.mesh.material_ids.empty() ? 0 : s.mesh.material_ids[0];
		smesh.end_index = static_cast<int>(indices.size());
	}

	////////////////// Rescale positions if requested
	if (scale > 0.f) 
	{
		glm::vec3 e = bbox.extent();
		float scale_mult = scale / std::max(std::max(e[0], e[1]), e[2]);
		original_scale = 1.f / scale_mult;
		for (glm::vec3& v : positions) v *= scale_mult;
		bbox.max *= scale_mult;
		bbox.min *= scale_mult;
	} else {
		original_scale = 1.f;
	}

	bsphere.from_bbox(bbox);
	

	////////////////// Add a minimum value if attributes are not available to avoid empty buffers
	if (normals.empty())   normals.push_back(glm::vec3(0, 0, 0));
	if (texcoords.empty()) texcoords.push_back(glm::vec2(0, 0));
	if (colors.empty())    colors.push_back(glm::vec3(0, 0, 0));
	if (matIds.empty())    matIds.push_back(0);

	////////////////// Optimize vertex positions
	if (optimize && false) 
	{
		meshopt_optimizeVertexCache(indices.data(), indices.data(), indices.size(), positions.size());
		std::vector<unsigned int> remap(positions.size());
		meshopt_optimizeVertexFetchRemap(remap.data(), indices.data(), indices.size(), positions.size());
		meshopt_remapIndexBuffer(indices.data(), indices.data(), indices.size(), remap.data());
		meshopt_remapVertexBuffer(positions.data(), positions.data(), positions.size(), sizeof(glm::vec3), remap.data());
		meshopt_remapVertexBuffer(normals.data(), normals.data(), normals.size(), sizeof(glm::vec3), remap.data());
		meshopt_remapVertexBuffer(texcoords.data(), texcoords.data(), texcoords.size(), sizeof(glm::vec2), remap.data());
		meshopt_remapVertexBuffer(matIds.data(), matIds.data(), matIds.size(), sizeof(GLshort), remap.data());
		meshopt_remapVertexBuffer(colors.data(), colors.data(), colors.size(), sizeof(glm::vec3), remap.data());
	}

	////////////////// Compute tangents
	computeTangents(positions, normals, texcoords, indices, tangents);

	reloadGPU();


	return SUCCESS;
}

void MeshData::reloadGPU()
{
	clearGPU();

	////////////////// Upload to GPU
	gl.posbuffer->Create(GL_ARRAY_BUFFER);
	gl.posbuffer->UploadData(positions.size() * sizeof(glm::vec3), GL_STATIC_DRAW, positions.data());

	gl.normalbuffer->Create(GL_ARRAY_BUFFER);
	gl.normalbuffer->UploadData(normals.size() * sizeof(glm::vec3), GL_STATIC_DRAW, normals.data());

	gl.texcoordbuffer->Create(GL_ARRAY_BUFFER);
	gl.texcoordbuffer->UploadData(texcoords.size() * sizeof(glm::vec2), GL_STATIC_DRAW, texcoords.data());

	gl.tangentbuffer->Create(GL_ARRAY_BUFFER);
	gl.tangentbuffer->UploadData(tangents.size() * sizeof(glm::vec4), GL_STATIC_DRAW, tangents.data());

	gl.colorbuffer->Create(GL_ARRAY_BUFFER);
	gl.colorbuffer->UploadData(colors.size() * sizeof(glm::vec3), GL_STATIC_DRAW, colors.data());

	gl.matidbuffer->Create(GL_ARRAY_BUFFER);
	gl.matidbuffer->UploadData(matIds.size() * sizeof(GLshort), GL_STATIC_DRAW, matIds.data());

	gl.vao->Create();
	gl.vao->SetAttributeBufferSource(gl.posbuffer, 0, 3, GL_FLOAT, GL_FALSE);
	gl.vao->SetAttributeBufferSource(gl.normalbuffer, 1, 3, GL_FLOAT, GL_TRUE);
	gl.vao->SetAttributeBufferSource(gl.texcoordbuffer->bytesize ? gl.texcoordbuffer : gl.normalbuffer, 2, 2, GL_FLOAT, GL_FALSE);
	gl.vao->SetAttributeBufferSource(gl.tangentbuffer, 3, 4, GL_FLOAT, GL_FALSE);
	gl.vao->SetAttributeBufferSource(gl.colorbuffer, 4, 3, GL_FLOAT, GL_FALSE);
	gl.vao->SetAttributeBufferSource(gl.matidbuffer, 5, 1, GL_SHORT, GL_FALSE);


	if (positions.size() >= (1 << 16)) {
		gl.indexbuffer->Create(GL_ELEMENT_ARRAY_BUFFER);
		gl.indexbuffer->UploadData(indices.size() * sizeof(GLuint), GL_STATIC_DRAW, indices.data());
		gl.indexbuffertype = GL_UNSIGNED_INT;
	}
	else {
		gl.indexbuffer->Create(GL_ELEMENT_ARRAY_BUFFER);
		std::vector<GLushort> shortIndices(indices.size());
		for (size_t i = 0; i < indices.size(); ++i) shortIndices[i] = indices[i];
		gl.indexbuffer->UploadData(shortIndices.size() * sizeof(GLushort), GL_STATIC_DRAW, shortIndices.data());
		gl.indexbuffertype = GL_UNSIGNED_SHORT;
	}

	/////////////////////// Create materials GL buffer
	materialdata.updateGLResources();

	////////////////////// Make textures resident
	makeGLTexturesResident();

}


void MeshData::clear()
{
	clearCPU();
	clearGPU();
	source_filename.clear();
}

void MeshData::clearGPU()
{
	gl.vao->Release();
	gl.posbuffer->Release();
	gl.normalbuffer->Release();
	gl.texcoordbuffer->Release();
	gl.tangentbuffer->Release();
	gl.colorbuffer->Release();
	gl.matidbuffer->Release();
	gl.indexbuffer->Release();

	materialdata.clearGLResources();
}

void MaterialData::setDirty()
{
	gldirty = true;
}

MaterialData::MaterialData()
{
	materialListBuffer = create_object<BufferObject>();
}

int MaterialData::updateGLResources()
{
	if (!materialListBuffer->id)
	{
		materialListBuffer->Create(GL_SHADER_STORAGE_BUFFER);
	}

	if (!gldirty) return SUCCESS;

	std::vector<MaterialStructGL> materialListGL(materials.size());
	for (size_t i = 0; i < materials.size(); ++i) materialListGL[i] = materials[i].gl;
	materialListBuffer->UploadData(materialListGL.size() * sizeof(MaterialStructGL), GL_STATIC_READ, materialListGL.data());
	gldirty = false;

	std::cout << "Updated Material GL data" << std::endl;//!!

	return SUCCESS;
}

void MaterialData::clearGLResources()
{
	materialListBuffer->Release();
}

void MeshData::clearCPU()
{
	positions.clear(); positions.shrink_to_fit();
	normals.clear();   normals.shrink_to_fit();
	texcoords.clear(); texcoords.shrink_to_fit();
	tangents.clear();  tangents.shrink_to_fit();
	colors.clear();    colors.shrink_to_fit();
	matIds.clear();    matIds.shrink_to_fit();
	indices.clear();   indices.shrink_to_fit();
	submeshes.clear(); submeshes.shrink_to_fit();

	materialdata.materials.clear(); materialdata.materials.shrink_to_fit();
	materialdata.setDirty();
}