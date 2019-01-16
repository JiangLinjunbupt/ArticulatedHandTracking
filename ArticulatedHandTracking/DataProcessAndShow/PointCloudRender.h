#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "Camera.h"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>


struct Grid {
	std::vector<unsigned int> indices;
	std::vector<float> vertices;
	std::vector<float> texcoords;

	Grid(int grid_width, int grid_height) {
		///--- So that we don't have to bother with connectivity data structure!!
		int primitive_restart_idx = 0xffffffff;
		glPrimitiveRestartIndex(primitive_restart_idx);
		glEnable(GL_PRIMITIVE_RESTART);

		int skip = 1;

		///--- Vertices
		for (int row = 0; row < grid_height; row += skip) {
			for (int col = 0; col < grid_width; col += skip) {
				float x = col;
				float y = row;
				vertices.push_back(x); /// i [0...width]
				vertices.push_back(y); /// y [0...height]
			}
		}

		///--- TexCoords
		for (int row = 0; row < grid_height; row += skip) {
			for (int col = 0; col < grid_width; col += skip) {
				float x = col / ((float)grid_width);
				float y = row / ((float)grid_height);
				texcoords.push_back(x); /// u [0,1]
				texcoords.push_back(y); /// v [0,1]
			}
		}

		///--- Faces
		for (int row = 0; row < grid_height - 1; row += skip) {
			for (int col = 0; col < grid_width; col += skip) {
				indices.push_back((row + 1) * grid_width + col);
				indices.push_back(row * grid_width + col);
			}
			indices.push_back(primitive_restart_idx);
		}
	}
};


class PointCloudRender
{
public:
	unsigned int ID;
	unsigned int vao;
	unsigned int vbo;
	unsigned int ebo;
	unsigned int texture_id_cmap = 0;
	unsigned int texture_id_color = 0;
	unsigned int texture_id_depth = 0;

	int num_indexes = 0;
	int num_vertices = 0;
	float alpha = 0.87f;

	Camera* _camera = NULL;
	// constructor generates the shader on the fly
	// ------------------------------------------------------------------------
	PointCloudRender(Camera* camera, const char* vertexPath, const char* fragmentPath, unsigned int depth_tex_id, unsigned int color_tex_id);


	void LoadandCompileShader(const char* vertexPath, const char* fragmentPath);
	void drawPointCloud();


	// activate the shader
	// ------------------------------------------------------------------------
	void use()const;
	void unuse() const;


	// utility uniform functions
	// ------------------------------------------------------------------------
	void setBool(const std::string &name, bool value) const;
	// ------------------------------------------------------------------------
	void setInt(const std::string &name, int value) const;
	// ------------------------------------------------------------------------
	void setFloat(const std::string &name, float value) const;
	// ------------------------------------------------------------------------
	void setVec2(const std::string &name, const glm::vec2 &value) const;
	void setVec2(const std::string &name, float x, float y) const;
	// ------------------------------------------------------------------------
	void setVec3(const std::string &name, const glm::vec3 &value) const;
	void setVec3(const std::string &name, float x, float y, float z) const;
	// ------------------------------------------------------------------------
	void setVec4(const std::string &name, const glm::vec4 &value) const;
	void setVec4(const std::string &name, float x, float y, float z, float w) const;
	// ------------------------------------------------------------------------
	void setMat2(const std::string &name, const glm::mat2 &mat) const;
	// ------------------------------------------------------------------------
	void setMat3(const std::string &name, const glm::mat3 &mat) const;
	void setMat3(const std::string &name, const Eigen::Matrix3f& mat) const;
	// ------------------------------------------------------------------------
	void setMat4(const std::string &name, const glm::mat4 &mat) const;


private:
	// utility function for checking shader compilation/linking errors.
	// ------------------------------------------------------------------------
	void checkCompileErrors(GLuint shader, std::string type);
};


