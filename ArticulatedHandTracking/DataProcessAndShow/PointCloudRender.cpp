#include"PointCloudRender.h"


// constructor generates the shader on the fly
// ------------------------------------------------------------------------
PointCloudRender::PointCloudRender(Camera* camera, const char* vertexPath, const char* fragmentPath, unsigned int depth_tex_id, unsigned int color_tex_id)
{
	this->_camera = camera;
	LoadandCompileShader(vertexPath, fragmentPath);

	Grid grid(_camera->width(), _camera->height());

	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);
	glGenBuffers(1, &ebo);

	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glBufferData(GL_ARRAY_BUFFER, (grid.vertices.size() + grid.texcoords.size()) * sizeof(float), NULL, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, grid.vertices.size() * sizeof(float), grid.vertices.data());
	glBufferSubData(GL_ARRAY_BUFFER, grid.vertices.size() * sizeof(float), grid.texcoords.size() * sizeof(float), grid.texcoords.data());


	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)(grid.vertices.size() * sizeof(float)));
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, grid.indices.size() * sizeof(unsigned int), grid.indices.data(), GL_STATIC_DRAW);


	this->use();  //设置shader 里面的uniform值的时候，需要先绑定着色器程序
	this->texture_id_depth = depth_tex_id;
	this->texture_id_color = color_tex_id;
	this->setInt("tex_depth", 1);
	this->setInt("tex_color", 0);

	const int sz = 3; GLfloat tex[3 * sz] = {/*red*/ 202.0 / 300, 86.0 / 300, 122.0 / 300, /*magenta*/  202.0 / 300, 86.0 / 300, 122.0 / 300, /*blue*/ 0.40, 0.0, 0.7 };
	glActiveTexture(GL_TEXTURE2);
	glGenTextures(1, &texture_id_cmap);
	glBindTexture(GL_TEXTURE_1D, texture_id_cmap);
	glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB, sz, 0, GL_RGB, GL_FLOAT, tex);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	this->setInt("fix_colormap", 2);




	this->setFloat("CameraCenterX", camera->cameraCenterX());
	this->setFloat("CameraCenterY", camera->cameraCenterY());
	this->setFloat("focal_length_x", camera->focal_length_x());
	this->setFloat("focal_length_y", camera->focal_length_y());
	this->setFloat("zNear", camera->zNear());
	this->setFloat("zFar", camera->zFar());
	this->setFloat("alpha", this->alpha);
	this->setFloat("enable_fix_colormap", 0.0f);


	num_indexes = grid.indices.size();
	num_vertices = grid.vertices.size();

	this->unuse();
	glBindVertexArray(0);
}


void PointCloudRender::LoadandCompileShader(const char* vertexPath, const char* fragmentPath)
{
	// 1. retrieve the vertex/fragment source code from filePath
	std::string vertexCode;
	std::string fragmentCode;
	std::ifstream vShaderFile;
	std::ifstream fShaderFile;
	// ensure ifstream objects can throw exceptions:
	vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	try
	{
		// open files
		vShaderFile.open(vertexPath);
		fShaderFile.open(fragmentPath);
		std::stringstream vShaderStream, fShaderStream;
		// read file's buffer contents into streams
		vShaderStream << vShaderFile.rdbuf();
		fShaderStream << fShaderFile.rdbuf();
		// close file handlers
		vShaderFile.close();
		fShaderFile.close();
		// convert stream into string
		vertexCode = vShaderStream.str();
		fragmentCode = fShaderStream.str();
	}
	catch (std::ifstream::failure e)
	{
		std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
	}
	const char* vShaderCode = vertexCode.c_str();
	const char * fShaderCode = fragmentCode.c_str();
	// 2. compile shaders
	unsigned int vertex, fragment;
	// vertex shader
	vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 1, &vShaderCode, NULL);
	glCompileShader(vertex);
	checkCompileErrors(vertex, "VERTEX");
	// fragment Shader
	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &fShaderCode, NULL);
	glCompileShader(fragment);
	checkCompileErrors(fragment, "FRAGMENT");
	// shader Program
	ID = glCreateProgram();
	glAttachShader(ID, vertex);
	glAttachShader(ID, fragment);
	glLinkProgram(ID);
	checkCompileErrors(ID, "PROGRAM");
	// delete the shaders as they're linked into our program now and no longer necessery
	glDeleteShader(vertex);
	glDeleteShader(fragment);
}
// activate the shader
// ------------------------------------------------------------------------
void PointCloudRender::drawPointCloud()
{
	if (texture_id_depth == 0)
	{
		std::cout << " texture_id_depth == 0\n";
		return;
	}

	if (alpha < 1.0) {
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}
	else {
		glDisable(GL_BLEND);
	}

	glBindVertexArray(vao);
	this->use();
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture_id_color);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, texture_id_depth);

	glEnable(GL_PROGRAM_POINT_SIZE);
	glDrawArrays(GL_POINTS, 0, num_vertices);
	glDisable(GL_PROGRAM_POINT_SIZE);

	this->unuse();
	glBindVertexArray(0);

}

void PointCloudRender::use()const { glUseProgram(ID); }
void PointCloudRender::unuse() const { glUseProgram(0); }


#pragma region SetFunctions
// utility uniform functions
// ------------------------------------------------------------------------
void PointCloudRender::setBool(const std::string &name, bool value) const
{
	int id = glGetUniformLocation(ID, name.c_str());
	if (id == -1)
		printf("!!!WARNING: shader '%d' does not contain uniform variable '%s'\n", ID, name);
	glUniform1i(id, (int)value);
}
// ------------------------------------------------------------------------
void PointCloudRender::setInt(const std::string &name, int value) const
{
	int id = glGetUniformLocation(ID, name.c_str());
	if (id == -1)
		printf("!!!WARNING: shader '%d' does not contain uniform variable '%s'\n", ID, name);
	glUniform1i(id, (int)value);
}
// ------------------------------------------------------------------------
void PointCloudRender::setFloat(const std::string &name, float value) const
{
	int id = glGetUniformLocation(ID, name.c_str());
	if (id == -1)
		printf("!!!WARNING: shader '%d' does not contain uniform variable '%s'\n", ID, name);
	glUniform1f(id, value);
}
// ------------------------------------------------------------------------
void PointCloudRender::setVec2(const std::string &name, const glm::vec2 &value) const
{
	int id = glGetUniformLocation(ID, name.c_str());
	if (id == -1)
		printf("!!!WARNING: shader '%d' does not contain uniform variable '%s'\n", ID, name);
	glUniform2fv(id, 1, &value[0]);
}
void PointCloudRender::setVec2(const std::string &name, float x, float y) const
{
	int id = glGetUniformLocation(ID, name.c_str());
	if (id == -1)
		printf("!!!WARNING: shader '%d' does not contain uniform variable '%s'\n", ID, name);
	glUniform2f(id, x, y);
}
// ------------------------------------------------------------------------
void PointCloudRender::setVec3(const std::string &name, const glm::vec3 &value) const
{
	int id = glGetUniformLocation(ID, name.c_str());
	if (id == -1)
		printf("!!!WARNING: shader '%d' does not contain uniform variable '%s'\n", ID, name);
	glUniform3fv(id, 1, &value[0]);
}
void PointCloudRender::setVec3(const std::string &name, float x, float y, float z) const
{
	int id = glGetUniformLocation(ID, name.c_str());
	if (id == -1)
		printf("!!!WARNING: shader '%d' does not contain uniform variable '%s'\n", ID, name);
	glUniform3f(id, x, y, z);
}
// ------------------------------------------------------------------------
void PointCloudRender::setVec4(const std::string &name, const glm::vec4 &value) const
{
	int id = glGetUniformLocation(ID, name.c_str());
	if (id == -1)
		printf("!!!WARNING: shader '%d' does not contain uniform variable '%s'\n", ID, name);
	glUniform4fv(id, 1, &value[0]);
}
void PointCloudRender::setVec4(const std::string &name, float x, float y, float z, float w) const
{
	int id = glGetUniformLocation(ID, name.c_str());
	if (id == -1)
		printf("!!!WARNING: shader '%d' does not contain uniform variable '%s'\n", ID, name);
	glUniform4f(id, x, y, z, w);
}
// ------------------------------------------------------------------------
void PointCloudRender::setMat2(const std::string &name, const glm::mat2 &mat) const
{
	int id = glGetUniformLocation(ID, name.c_str());
	if (id == -1)
		printf("!!!WARNING: shader '%d' does not contain uniform variable '%s'\n", ID, name);
	glUniformMatrix2fv(id, 1, GL_FALSE, &mat[0][0]);
}
// ------------------------------------------------------------------------
void PointCloudRender::setMat3(const std::string &name, const glm::mat3 &mat) const
{
	int id = glGetUniformLocation(ID, name.c_str());
	if (id == -1)
		printf("!!!WARNING: shader '%d' does not contain uniform variable '%s'\n", ID, name);
	glUniformMatrix3fv(id, 1, GL_FALSE, &mat[0][0]);
}

void PointCloudRender::setMat3(const std::string &name, const Eigen::Matrix3f& mat) const
{
	int id = glGetUniformLocation(ID, name.c_str());
	if (id == -1)
		printf("!!!WARNING: shader '%d' does not contain uniform variable '%s'\n", ID, name);
	glUniformMatrix3fv(id, 1, GL_FALSE, mat.data());
}

// ------------------------------------------------------------------------
void PointCloudRender::setMat4(const std::string &name, const glm::mat4 &mat) const
{
	int id = glGetUniformLocation(ID, name.c_str());
	if (id == -1)
		printf("!!!WARNING: shader '%d' does not contain uniform variable '%s'\n", ID, name);
	glUniformMatrix4fv(id, 1, GL_FALSE, &mat[0][0]);
}

#pragma endregion SetFunctions


// utility function for checking shader compilation/linking errors.
// ------------------------------------------------------------------------
void PointCloudRender::checkCompileErrors(GLuint shader, std::string type)
{
	GLint success;
	GLchar infoLog[1024];
	if (type != "PROGRAM")
	{
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
	else
	{
		glGetProgramiv(shader, GL_LINK_STATUS, &success);
		if (!success)
		{
			glGetProgramInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
}