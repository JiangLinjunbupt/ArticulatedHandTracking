#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>

class ColorTexture8UC3 {
private:
	GLuint texture = 0;
	int width, height;
	bool _is_init = false;
	int fid = -1; ///< loaded frame data (avoid unnecessary loads)
public:
	GLuint texid() {
		if (!_is_init)
		{
			std::cout << "ColorTexture8UC3 does not init!!!\n";
			exit(0);
		}

		return texture;
	}

	ColorTexture8UC3(int width, int height) {
		this->width = width;
		this->height = height;
		if (_is_init) return;
		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		std::vector<unsigned char> data(width*height * 3, 0); ///< initialization
		/// @see Table.2 https://www.khronos.org/opengles/sdk/docs/man3/docbook4/xhtml/glTexImage2D.xml
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, data.data());
		glBindTexture(GL_TEXTURE_2D, 0);
		_is_init = true;
	}

	/// @note -1 forces refresh
	void load(GLvoid* pixels, int fid = -1) {
		if (this->fid != fid || fid == -1) {
			glBindTexture(GL_TEXTURE_2D, texture);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, pixels);
			glBindTexture(GL_TEXTURE_2D, 0);
			this->fid = fid;
		}
	}
};

