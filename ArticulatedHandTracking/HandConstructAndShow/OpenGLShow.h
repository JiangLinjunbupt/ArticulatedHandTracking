#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include"HandModel.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "OpenGLCamera.h"
#include "shader.h"
#include <iostream>
using namespace std;


namespace DisPlay_ReSult
{
	void framebuffer_size_callback(GLFWwindow* window, int width, int height);
	void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
	void mouse_callback(GLFWwindow* window, double xpos, double ypos);
	void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
	void processInput(GLFWwindow *window);
	unsigned int loadTexture(char const * path);

	// settings
	unsigned int SCR_WIDTH = 640 * 2;
	unsigned int SCR_HEIGHT = 480 * 2;

	// camera
	OpenGLCamera opengl_camera(glm::vec3(0.0f, 0.0f, 3.0f));
	float lastX = SCR_WIDTH / 2.0f;
	float lastY = SCR_HEIGHT / 2.0f;
	bool firstMouse = true;
	bool mouse_leftbutton_press = false;

	// timing
	float deltaTime = 0.0f;	// time between current frame and last frame
	float lastFrame = 0.0f;

	HandModel * handmodel;
	unsigned int vao, vbo, ebo;

	unsigned int Axis_vao, Axis_vbo;

	Shader *ourShader;
	Shader *AxisShader;
	unsigned int Material_texture;

	GLFWwindow* window;


	float Axis_vertex[] = {
		0.0f , 0.0f , 0.0f ,
		5.0f, 0.0f, 0.0f,

		0.0f , 0.0f , 0.0f ,
		0.0f,5.0f,0.0f,

		0.0f , 0.0f , 0.0f ,
		0.0f,0.0f,5.0f
	};

	int init()
	{
		// glfw: initialize and configure
		// ------------------------------
		glfwInit();
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

		// glfw window creation
		// --------------------
		window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "ArticulatedHandTracking", NULL, NULL);
		if (window == NULL)
		{
			std::cout << "Failed to create GLFW window" << std::endl;
			glfwTerminate();
			return -1;
		}
		glfwMakeContextCurrent(window);
		glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
		glfwSetMouseButtonCallback(window, mouse_button_callback);
		glfwSetCursorPosCallback(window, mouse_callback);
		glfwSetScrollCallback(window, scroll_callback);

		// tell GLFW to capture our mouse
		//glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

		// glad: load all OpenGL function pointers
		// ---------------------------------------
		if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
		{
			std::cout << "Failed to initialize GLAD" << std::endl;
			return -1;
		}
	}

	void init_BUFFER()
	{
		glEnable(GL_DEPTH_TEST);

		glGenVertexArrays(1, &vao);
		glGenBuffers(1, &vbo);
		glGenBuffers(1, &ebo);

		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float)*handmodel->Vertex_num * 6, NULL, GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int)*handmodel->Face_num*3, handmodel->F_array, GL_STATIC_DRAW);

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(sizeof(float)*handmodel->Vertex_num*3));
		glEnableVertexAttribArray(1);


		glGenVertexArrays(1, &Axis_vao);
		glGenBuffers(1, &Axis_vbo);

		glBindVertexArray(Axis_vao);
		glBindBuffer(GL_ARRAY_BUFFER, Axis_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Axis_vertex), Axis_vertex, GL_STATIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, FALSE, 3 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);


		Material_texture = loadTexture("skin.png");

		ourShader = new Shader("vertex.glsl", "fragment.glsl");
		ourShader->use();
		ourShader->setInt("material.Material_texture", 0);
		ourShader->setFloat("material.shininess", 2.0f);

		ourShader->setVec3("dirLight.direction", -0.2f, -1.0f, -0.3f);
		ourShader->setVec3("dirLight.ambient", 0.5f, 0.5f, 0.5f);
		ourShader->setVec3("dirLight.diffuse", 0.7f, 0.7f, 0.7f);
		ourShader->setVec3("dirLight.specular", 0.2f, 0.2f, 0.2f);


		AxisShader = new Shader("Axis_vertex.glsl", "Axis_fragment.glsl");
	}
	int Display()
	{
		// render loop
		// -----------
		while (!glfwWindowShouldClose(window))
		{
			// per-frame time logic
			// --------------------
			float currentFrame = glfwGetTime();
			deltaTime = currentFrame - lastFrame;
			lastFrame = currentFrame;

			// input
			// -----
			processInput(window);

			// render
			// ------
			glClearColor(1.0,1.0,1.0,1.0);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			// activate shader
			ourShader->use();
			glActiveTexture(0);
			glBindTexture(GL_TEXTURE_2D, Material_texture);
			ourShader->setVec3("viewPos", opengl_camera.Position);

			// pass projection matrix to shader (note that in this case it could change every frame)
			glm::mat4 projection = glm::perspective(glm::radians(opengl_camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
			ourShader->setMat4("projection", projection);

			// camera/view transformation
			glm::mat4 view = opengl_camera.GetViewMatrix();
			ourShader->setMat4("view", view);

			handmodel->serializeModel();

			glBindVertexArray(vao);
			glBindBuffer(GL_ARRAY_BUFFER,vbo);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float)*handmodel->Vertex_num * 3, handmodel->V_Final_array);
			glBufferSubData(GL_ARRAY_BUFFER, sizeof(float)*handmodel->Vertex_num * 3, sizeof(float)*handmodel->Vertex_num * 3, handmodel->Normal_Final_array);
			//glDrawArrays(GL_POINTS, 0, handmodel->Vertex_num);
			
			//问题参见：https://www.opengl.org/discussion_boards/showthread.php/141929-problem-w-glDrawElements
			//这里有个巨大的坑：只有GL_UNSIGNED_INT,	GL_UNSIGNED_BYTE 或者  GL_UNSIGNED_SHORT才能被允许，我最开始使用GL_INT一直画不出来图
			glDrawElements(GL_TRIANGLES, handmodel->Face_num*3, GL_UNSIGNED_INT,0);

			AxisShader->use();
			AxisShader->setMat4("projection", projection);
			AxisShader->setMat4("view", view);
			glBindVertexArray(Axis_vao);
			glLineWidth(5);
			glDrawArrays(GL_LINES, 0, 6);

			// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
			// -------------------------------------------------------------------------------
			glfwSwapBuffers(window);
			glfwPollEvents();
		}

		// glfw: terminate, clearing all previously allocated GLFW resources.
		// ------------------------------------------------------------------
		glfwTerminate();
		return 0;
	}


	// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
	// ---------------------------------------------------------------------------------------------------------
	void processInput(GLFWwindow *window)
	{
		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
			glfwSetWindowShouldClose(window, true);

		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
			opengl_camera.ProcessKeyboard(FORWARD, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
			opengl_camera.ProcessKeyboard(BACKWARD, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
			opengl_camera.ProcessKeyboard(LEFT, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
			opengl_camera.ProcessKeyboard(RIGHT, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
			opengl_camera.ProcessKeyboard(UP, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
			opengl_camera.ProcessKeyboard(DOWN, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
			opengl_camera.ResetPostion();
	}

	// glfw: whenever the window size changed (by OS or user resize) this callback function executes
	// ---------------------------------------------------------------------------------------------
	void framebuffer_size_callback(GLFWwindow* window, int width, int height)
	{
		// make sure the viewport matches the new window dimensions; note that width and 
		// height will be significantly larger than specified on retina displays.
		glViewport(0, 0, width, height);

		SCR_WIDTH = width;
		SCR_HEIGHT = height;
	}

	void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
	{
		if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
			mouse_leftbutton_press = true;
		if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
		{
			mouse_leftbutton_press = false;
			firstMouse = true;
		}
	}
	// glfw: whenever the mouse moves, this callback is called
	// -------------------------------------------------------
	void mouse_callback(GLFWwindow* window, double xpos, double ypos)
	{
		if (mouse_leftbutton_press)
		{
			if (firstMouse)
			{
				lastX = xpos;
				lastY = ypos;
				firstMouse = false;
			}

			float xoffset = xpos - lastX;
			float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

			lastX = xpos;
			lastY = ypos;

			opengl_camera.ProcessMouseMovement(xoffset, yoffset);
		}
	}

	// glfw: whenever the mouse scroll wheel scrolls, this callback is called
	// ----------------------------------------------------------------------
	void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
	{
		opengl_camera.ProcessMouseScroll(yoffset);
	}


	// utility function for loading a 2D texture from file
	// ---------------------------------------------------
	unsigned int loadTexture(char const * path)
	{
		unsigned int textureID;
		glGenTextures(1, &textureID);

		int width, height, nrComponents;
		unsigned char *data = stbi_load(path, &width, &height, &nrComponents, 0);
		if (data)
		{
			GLenum format;
			if (nrComponents == 1)
				format = GL_RED;
			else if (nrComponents == 3)
				format = GL_RGB;
			else if (nrComponents == 4)
				format = GL_RGBA;

			glBindTexture(GL_TEXTURE_2D, textureID);
			glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
			glGenerateMipmap(GL_TEXTURE_2D);

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

			stbi_image_free(data);
		}
		else
		{
			std::cout << "Texture failed to load at path: " << path << std::endl;
			stbi_image_free(data);
		}

		return textureID;
	}
}