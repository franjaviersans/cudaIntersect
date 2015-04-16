#pragma comment (lib,"lib/AntTweakBar.lib")
#pragma comment(lib, "lib/glfw3dll.lib")
#pragma comment(lib, "lib/glew32.lib")
#pragma comment(lib, "opengl32.lib")



#include "CPUtimer.h"
#include "Definitions.h"

#include "include/GLFW/glfw3.h"
#include "include/AntTweakBar/AntTweakBar.h"
#include "GLSLProgram.h"
#include "Transformation.h"
#include "3DModel.h"
#include "ArcBall.h"
#include "FBOQuad.h"
#include <stdlib.h>
#include <string>
#include <iostream>
#include <vector>



using namespace std;

///< Only wrapping the glfw functions
namespace glfwFunc
{

	#ifdef CUDA_CODE
		//Create a class with the CUDA wraper
		CUDA c;
	#endif
	GLFWwindow* glfwWindow;
	const unsigned int WINDOW_WIDTH = 1024;
	const unsigned int WINDOW_HEIGHT = 650;
	unsigned int Q;
	unsigned int M;
	unsigned int N;
	unsigned int gridX;
	unsigned int gridY = 5;
	unsigned int gridZ = 5;
	unsigned int blockX = 512;
	unsigned int muestras;
	float tiempo_total;
	float tiempo_computo;
	float tiempo_promedio;
	unsigned int hilos, hilosxbloque;
	const float NCP = 0.01f;
	const float FCP = 52.f;
	const float fAngle = 45.f;
	FBOQuad * quad;
	string strNameWindow = "Hello GLFW";
	glm::vec3 eye = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 lightdir = glm::vec3(3.0f, 3.0f, 3.0f);
	glm::vec4 BLACK = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
	glm::vec4 GREEN = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
	glm::vec4 RED = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
	//glm::vec4 m_vec4ColorAB = glm::vec4(0.5f, 0.64f, 0.26f, 0.3f);
	//glm::vec4 m_vec4ColorC = glm::vec4(0.67f, 0.28f, 0.31f, 1.0f);
	glm::vec4 m_vec4ColorAB = glm::vec4(17.0f/256.0f, 164.0f/256.0f, 2.0f/256.0f, 0.3f);
	glm::vec4 m_vec4ColorC = glm::vec4(255.0f/256.0f, 76.0f/256.0f, 76.0f/256.0f, 1.0f);
	glm::mat4 m_matOrtho = glm::ortho(-0.5f, 0.5f, -0.5f, 0.5f, -0.5f, 0.5f);
	UINT32 iteration = 0;
	C3DModel m_model, m_cone;
	CArcBall m_arcball;
	TwBar *myBar;
	bool m_bLeftButton;
	bool m_bFlag;
	glm::ivec2 m_MousePoint(0);
	Transformation m_trans;
	vector<Transformation> m_vTransformation;


	CGLSLProgram m_program, m_bgprogram;
	glm::mat4x4 mProjMatrix, mModelViewMatrix, mCTransfor;


	void TW_CALL pressExit(void *clientData)
	{ 
		TwTerminate();
		exit(0);
	}

	void TW_CALL pressStart(void *clientData)
	{ 
		unsigned int i = 0;
		m_vTransformation.clear();		
		#ifdef CUDA_CODE
			float total_time = 0;

			//A timer to know how much the CPU is
			CPUTimer timer;

			//Start CPU timer
			timer.StartCounter();

			
			bool finish = true;

			unsigned int intersections[MAX_N];
			float scalars[MAX_N];

			//Do M iterations to test the data
			for(i = 0;i < M /*&& finish*/;++i)
			{
				Transformation t;
				
				
				finish = c.CudaIntercept(total_time, scalars, intersections, N, t, gridX, gridY, gridZ, blockX);

				t.iteration = i;
				t.intersection = finish;

				if(Q < M && i % Q == 0) m_vTransformation.push_back(t);
				else if(Q >= M && i == M - 1 ) m_vTransformation.push_back(t);				
			}

			//Print the total GPUtime of execution
			printf("Total time: %f msecs.\n", total_time);

			//Print the average GPU time per iteration
			printf("Average: %f msecs.\n", total_time / i);

			//Total execution time
			cout << "Tiempo: "<< timer.GetCounter() <<" microseconds\n";


			tiempo_total = float(timer.GetCounter());
			tiempo_computo = total_time;
			tiempo_promedio = total_time / i;
		#endif

		//Set the min and the max in the range of the iteration number
		iteration = 0;
		TwSetParam(myBar, "Muestra", "min", TW_PARAM_INT32, 1, &iteration);
		iteration = m_vTransformation.size();
		TwSetParam(myBar, "Muestra", "max", TW_PARAM_INT32, 1, &iteration);

		iteration = m_vTransformation.size();
		if(i > 0) m_trans = m_vTransformation[iteration - 1];
		else{
			m_trans.m_fRotationAngle = 0.0f;
			m_trans.m_fRotationVectorx = 0.0f;
			m_trans.m_fRotationVectory = 0.0f;
			m_trans.m_fRotationVectorz = 0.0f;
			m_trans.m_fScalar = 0.0f;
			m_trans.m_fTranslationx = 0.0f;
			m_trans.m_fTranslationy = 0.0f;
			m_trans.m_fTranslationz = 0.0f;
			m_trans.intersection = false;
			m_trans.iteration = 0;
			tiempo_total = 0;
			tiempo_computo = 0;
			tiempo_promedio = 0;
		}
	}

	//Reset GUI
	void TW_CALL pressClear(void *clientData)
	{ 
		tiempo_total = 0;
		tiempo_computo = 0;
		tiempo_promedio = 0;
		iteration = 0;
		m_vTransformation.clear();
		m_trans.m_fRotationAngle = 0.0f;
		m_trans.m_fRotationVectorx = 0.0f;
		m_trans.m_fRotationVectory = 0.0f;
		m_trans.m_fRotationVectorz = 0.0f;
		m_trans.m_fScalar = 0.0f;
		m_trans.m_fTranslationx = 0.0f;
		m_trans.m_fTranslationy = 0.0f;
		m_trans.m_fTranslationz = 0.0f;
	}

	inline int TwEventMouseWheelGLFW3(GLFWwindow* window, double xoffset, double yoffset)
	{return TwEventMouseWheelGLFW((int)yoffset);}
	inline int TwEventCharGLFW3(GLFWwindow* window, int codepoint)
	{return TwEventCharGLFW(codepoint, GLFW_PRESS);}
	inline int TwWindowSizeGLFW3(GLFWwindow* window, int width, int height)
	{return TwWindowSize(width, height);}


	//Con esta funcion se puede obtener el valor 
	void TW_CALL SetVarCallback(const void *value, void *clientData)
	{
		iteration = ((const int *)value)[0]; 
		if(iteration != 0)
		{
			m_trans = m_vTransformation[iteration - 1];
		}
		else
		{
			m_trans.m_fRotationAngle = 0.0f;
			m_trans.m_fRotationVectorx = 0.0f;
			m_trans.m_fRotationVectory = 0.0f;
			m_trans.m_fRotationVectorz = 0.0f;
			m_trans.m_fScalar = 0.0f;
			m_trans.m_fTranslationx = 0.0f;
			m_trans.m_fTranslationy = 0.0f;
			m_trans.m_fTranslationz = 0.0f;
			m_trans.intersection = false;
			m_trans.iteration = 0;
		}
	}

	void TW_CALL GetVarCallback(void *value, void *clientData)
	{
		((int*) value)[0] = iteration;
	}

	//Con esta funcion se puede obtener el valor 
	void TW_CALL SetVarCallback2(const void *value, void *clientData)
	{
		blockX = ((const int *)value)[0]; 
		gridX = (m_model.GetPointerMesh()->size() + blockX)/blockX;
	}

	void TW_CALL GetVarCallback2(void *value, void *clientData)
	{
		((int*) value)[0] = blockX;
	}

	///< Callback function used by GLFW to capture some possible error.
	void errorCB(int error, const char* description)
	{
		cout << description << endl;
	}

	void onMouseMove(GLFWwindow *window, double xpos, double ypos)
	{
		TwMouseMotion(int(xpos), int(ypos));
		if (m_bLeftButton)
		{
			if (!m_bFlag)
			{
				m_bFlag = true;
				m_arcball.OnMouseDown(glm::ivec2(xpos, ypos));
			}
			else
			{
				m_MousePoint.x = int(xpos);
				m_MousePoint.y = int(ypos);
				m_arcball.OnMouseMove(m_MousePoint, ROTATE);
			}
		}
	}

	void onMouseDown(GLFWwindow* window, int button, int action, int mods)
	{

		if(m_bLeftButton)
		{
			m_bLeftButton = false;
					m_bFlag = false;
					m_arcball.OnMouseUp(m_MousePoint);
		}

		if(!TwEventMouseButtonGLFW(button, action))
		{

			if (action == GLFW_PRESS)
			{
				if (button == GLFW_MOUSE_BUTTON_LEFT)
				{
					m_bLeftButton = true;
				}
			}
			else if (action == GLFW_RELEASE)
			{
				if (button == GLFW_MOUSE_BUTTON_LEFT)
				{
				
				}
			}
		}
	}

	///
	/// The keyboard function call back
	/// @param window id of the window that received the event
	/// @param iKey the key pressed or released
	/// @param iScancode the system-specific scancode of the key.
	/// @param iAction can be GLFW_PRESS, GLFW_RELEASE or GLFW_REPEAT
	/// @param iMods Bit field describing which modifier keys were held down (Shift, Alt, & so on)
	///
	void keyboardCB(GLFWwindow* window, int iKey, int iScancode, int iAction, int iMods)
	{
		if(!TwEventKeyGLFW(iKey, iAction))
		{
			if (iAction == GLFW_PRESS)
			{
				switch (iKey)
				{
				case GLFW_KEY_ESCAPE:
				case GLFW_KEY_Q:
					glfwSetWindowShouldClose(window, GL_TRUE);
					break;
				}
			}
		}
	}

	///< The resizing function
	void resizeCB(GLFWwindow* window, int iWidth, int iHeight)
	{
		if (iHeight == 0) iHeight = 1;
		float ratio = iWidth / float(iHeight);
		glViewport(0, 0, iWidth, iHeight);
		mProjMatrix = glm::perspective(fAngle, ratio, NCP, FCP);
		m_arcball.Resize(float(iWidth), float(iHeight));

		TwWindowSizeGLFW3(window, iWidth, iHeight);

	}


	///
	/// Init all data and variables.
	/// @return true if everything is ok, false otherwise
	///
	bool initialize()
	{
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);
		glPolygonOffset(1, 1);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glewExperimental = GL_TRUE;
		m_bLeftButton = m_bFlag = false;
		if (glewInit() != GLEW_OK)
		{
			cout << "- glew Init failed :(" << endl;
			return false;
		}
		std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
		std::cout << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
		std::cout << "Vendor: " << glGetString(GL_VENDOR) << std::endl;
		std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
		//load the shaders
		m_program.loadShader("shaders/basic.vert", CGLSLProgram::VERTEX);
		m_program.loadShader("shaders/basic.frag", CGLSLProgram::FRAGMENT);
		m_program.create_link();
		m_program.enable();
		m_program.addAttribute("vVertex");
		m_program.addAttribute("vNormal");
		m_program.addUniform("mProjection");
		m_program.addUniform("mModelView");
		m_program.addUniform("vec4Color");
		m_program.addUniform("vec3Eye");
		m_program.addUniform("vec3Lightdir");
		m_program.disable();
		

		//Load the bgShader
		m_bgprogram.loadShader("shaders/background.vert", CGLSLProgram::VERTEX);
		m_bgprogram.loadShader("shaders/background.frag", CGLSLProgram::FRAGMENT);
		m_bgprogram.create_link();
		m_bgprogram.enable();
		m_bgprogram.addAttribute("vVertex");
		m_bgprogram.addAttribute("vColor");
		m_bgprogram.addUniform("mProjection");
		m_bgprogram.disable();


		//Load Geometry
		m_model.load("geometry/surfaceAB.ply");
		m_cone.load("geometry/surfaceC.ply");


		TwInit(TW_OPENGL_CORE, NULL);
		TwWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);


		quad = FBOQuad::Instance();


		//Declare the bar
		myBar = TwNewBar("Menu");

		
		Q = 10;
		M = 2000;

		#ifdef CUDA_CODE
			//Pass the data to GPU
			c.Init((float3 *)((m_model.GetPointerData())->data()), 
									(uint3* )((m_model.GetPointerMesh())->data()), 
									(float4 *)((m_model.GetPointerNormal())->data()),
									(float3 *)((m_cone.GetPointerData())->data()),
									(m_model.GetPointerData())->size(), 
									(m_model.GetPointerMesh())->size(),
									(m_model.GetPointerNormal())->size(),
									(m_cone.GetPointerData())->size());
		#endif

		tiempo_total = 0.0f;
		tiempo_computo =  0.0f;
		tiempo_promedio = 0.0f;

		//Set the varaibles to initiate a run in the code
		TwAddVarRW(myBar,"Numero de iteraciones (M)",TW_TYPE_UINT32, &M, "label='Numero de iteraciones (M)' group='Opciones de corrida'");
		TwAddVarRW(myBar,"Numero de transformaciones (N)",TW_TYPE_UINT32, &N, "label='Numero de transformaciones (N)' group='Opciones de corrida'");
		TwAddVarRW(myBar,"Intervalo de muestreo (Q)",TW_TYPE_UINT32, &Q, "label='Intervalo de muestreo (Q)' group='Opciones de corrida'");
		TwAddVarCB(myBar,"Block Dimension X",TW_TYPE_UINT32, SetVarCallback2, GetVarCallback2, &blockX, "label='Block Dimension X' group='Opciones de corrida'");
		gridX = (m_model.GetPointerMesh()->size() + blockX)/blockX;
		TwAddVarRO(myBar,"Grid Dimension X",TW_TYPE_UINT32, &gridX, "label='Grid Dimension X' group='Opciones de corrida'");
		TwAddVarRW(myBar,"Grid Dimension Y",TW_TYPE_UINT32, &gridY, "label='Grid Dimension Y' group='Opciones de corrida'");
		TwAddVarRW(myBar,"Grid Dimension Z",TW_TYPE_UINT32, &gridZ, "label='Grid Dimension Z' group='Opciones de corrida'");
		TwAddButton(myBar,"Start", pressStart,NULL,"label='Start' group='Opciones de corrida'");
		TwAddButton(myBar,"Reset", pressClear,NULL,"label='Reset' group='Opciones de corrida'");

		//Set a new variable for the iterations
		TwAddVarCB(myBar,"Muestra",TW_TYPE_UINT32, SetVarCallback, GetVarCallback, &iteration, "label='Muestra' group=Transformacion");

		//Set new variables for transormations
		TwAddVarRO(myBar,"Iteracion",TW_TYPE_UINT32, &m_trans.iteration, "label='Iteracion' group=Transformacion");
		TwAddVarRO(myBar,"Solucion",TW_TYPE_BOOLCPP, &m_trans.intersection, "label='Solucion' group=Transformacion");
		TwAddVarRO(myBar,"Translacion en X",TW_TYPE_FLOAT, &m_trans.m_fTranslationx, "label='Translacion en X' group=Transformacion");
		TwAddVarRO(myBar,"Translacion en Y",TW_TYPE_FLOAT, &m_trans.m_fTranslationy, "label='Translacion en Y' group=Transformacion");
		TwAddVarRO(myBar,"Translacion en Z",TW_TYPE_FLOAT, &m_trans.m_fTranslationz, "label='Translacion en Z' group=Transformacion");
		TwAddVarRO(myBar,"Escalamiento",TW_TYPE_FLOAT, &m_trans.m_fScalar, "label='Escalamiento' group=Transformacion");
		TwAddVarRO(myBar,"Angulo de rotacion",TW_TYPE_FLOAT, &m_trans.m_fRotationAngle, "label='Angulo de Rotacion' group=Transformacion");
		TwAddVarRO(myBar,"Eje de rotacion",TW_TYPE_DIR3F, &m_trans.m_fRotationVectorx, "label='Eje de rotacion' group=Transformacion");

		//Set the variables for the execution time
		TwAddVarRO(myBar,"Total",TW_TYPE_FLOAT, &tiempo_total, "label='Total' group='Tiempo de ejecucion'");
		TwAddVarRO(myBar,"Computo",TW_TYPE_FLOAT, &tiempo_computo, "label='Computo' group='Tiempo de ejecucion'");
		TwAddVarRO(myBar,"Por iteracion",TW_TYPE_FLOAT, &tiempo_promedio, "label='Por iteracion' group='Tiempo de ejecucion'");

		//Define a exit button
		TwAddButton(myBar,"Salir", pressExit,NULL,"label='Salir' group=Archivo");

		//Set the min and the max in the range of the iteration number
		iteration = 0;
		TwSetParam(myBar, "Muestra", "min", TW_PARAM_INT32, 1, &iteration);
		TwSetParam(myBar, "Muestra", "max", TW_PARAM_INT32, 1, &iteration);

		M = 0;
		TwSetParam(myBar, "Numero de iteraciones (M)", "min", TW_PARAM_INT32, 1, &M);
		M = MAX_M;
		TwSetParam(myBar, "Numero de iteraciones (M)", "max", TW_PARAM_INT32, 1, &M);
		M = 2000;

		N = 0;
		TwSetParam(myBar, "Numero de transformaciones (N)", "min", TW_PARAM_INT32, 1, &N);
		N = MAX_N;
		TwSetParam(myBar, "Numero de transformaciones (N)", "max", TW_PARAM_INT32, 1, &N);
		N = 500;
		
		//Set the callbacks!!!
		glfwSetKeyCallback(glfwFunc::glfwWindow, glfwFunc::keyboardCB);
		glfwSetWindowSizeCallback(glfwFunc::glfwWindow, glfwFunc::resizeCB);
		glfwSetMouseButtonCallback(glfwFunc::glfwWindow, glfwFunc::onMouseDown);
		glfwSetCursorPosCallback(glfwFunc::glfwWindow, glfwFunc::onMouseMove);
		glfwSetScrollCallback(glfwWindow, (GLFWscrollfun)TwEventMouseWheelGLFW3);
		glfwSetCharCallback(glfwWindow, (GLFWcharfun)TwEventCharGLFW3);


		return true;
	}

	///< The main rendering function.
	void draw()
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		//glClearColor(0.94f, 0.94f, 0.94f, 0.f);
		//glClearColor(0.25f, 0.25f, 0.25f, 0.f);
		glClearColor(0.54f, 0.54f, 0.54f, 0.f);
		mModelViewMatrix = glm::translate(glm::mat4(), glm::vec3(0, 0, -6.f)) * m_arcball.GetTransformation();

		mCTransfor = glm::mat4();

		if(m_vTransformation.size() != 0 && iteration != 0 && iteration <= m_vTransformation .size())
		{
		
			//Generate quaternion
			glm::vec3 rotation_angle = glm::normalize(glm::vec3(m_vTransformation[iteration - 1].m_fRotationVectorx,
																m_vTransformation[iteration - 1].m_fRotationVectory, 
																m_vTransformation[iteration - 1].m_fRotationVectorz));
			glm::quat quater = glm::quat(m_vTransformation[iteration - 1].m_fRotationAngle, glm::normalize(glm::vec3(rotation_angle)));

			//Generate rotation matrix
			glm::mat4 RotationMat = glm::mat4_cast(glm::normalize(quater));

			//Generate Transformation
			mCTransfor =	glm::translate(glm::mat4(), glm::vec3(  m_vTransformation[iteration - 1].m_fTranslationx,
																	m_vTransformation[iteration - 1].m_fTranslationy, 
																	m_vTransformation[iteration - 1].m_fTranslationz)) * 
							RotationMat * 
							glm::scale(glm::mat4(), glm::vec3(m_vTransformation[iteration - 1].m_fScalar)) * 
							glm::mat4();
		}
		

		m_bgprogram.enable();
		{
			glDisable(GL_DEPTH_TEST);
			glUniformMatrix4fv(m_bgprogram.getLocation("mProjection"), 1, GL_FALSE, glm::value_ptr(m_matOrtho));
			quad->Draw();
			glEnable(GL_DEPTH_TEST);
		}
		m_bgprogram.disable();

		m_program.enable();
		{
			glUniformMatrix4fv(m_program.getLocation("mProjection"), 1, GL_FALSE, glm::value_ptr(mProjMatrix));
			mModelViewMatrix = glm::translate(glm::mat4(), glm::vec3(0, 0, -6.f)) * glm::scale(glm::mat4(), glm::vec3(0.6, 0.6, 0.6)) * m_arcball.GetTransformation();
			//glUniformMatrix4fv(m_program.getLocation("mModelView"), 1, GL_FALSE, glm::value_ptr(mModelViewMatrix));

			/*if(m_vTransformation.size() != 0 && iteration != 0 && iteration <= m_vTransformation .size() && m_vTransformation[iteration - 1].intersection)
			{
				glDisable(GL_CULL_FACE);
				glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
				glLineWidth(1.0);
				glUniform4fv(m_program.getLocation("vec4Color"), 1, glm::value_ptr(GREEN));
				m_model.drawTriangleObject(m_vTransformation[iteration - 1].triID);
				cout<<m_vTransformation[iteration - 1].triID<<endl;
				glEnable(GL_CULL_FACE);
			}*/
			
			//ModelView for the cone
			glm::mat4 mataux = mModelViewMatrix * mCTransfor;
			
			glUniformMatrix4fv(m_program.getLocation("mModelView"), 1, GL_FALSE, glm::value_ptr(mataux));
			mataux = glm::inverse(mataux);
			glm::vec4 auxEye = mataux * glm::vec4(eye, 1.0f), auxLightdir = mataux * glm::vec4(lightdir, 0.0f);

			glUniform4fv(m_program.getLocation("vec4Color"), 1, glm::value_ptr(m_vec4ColorC));
			glUniform3fv(m_program.getLocation("vec3Eye"), 1, glm::value_ptr(auxEye));
			glUniform3fv(m_program.getLocation("vec3Lightdir"), 1, glm::value_ptr(auxLightdir));
			m_cone.drawObject();
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			glPolygonOffset(1, 1);
			glLineWidth(3.0f);
			glUniform4fv(m_program.getLocation("vec4Color"), 1, glm::value_ptr(BLACK));
			m_cone.drawObject();
			
			//Reset modelview
			glUniformMatrix4fv(m_program.getLocation("mModelView"), 1, GL_FALSE, glm::value_ptr(mModelViewMatrix));
			mataux = glm::inverse(mModelViewMatrix);
			auxEye = mataux * glm::vec4(eye, 1.0f);
			auxLightdir = mataux * glm::vec4(lightdir, 0.0f);

			glUniform4fv(m_program.getLocation("vec4Color"), 1, glm::value_ptr(m_vec4ColorAB));
			glUniform3fv(m_program.getLocation("vec3Eye"), 1, glm::value_ptr(auxEye));
			glUniform3fv(m_program.getLocation("vec3Lightdir"), 1, glm::value_ptr(auxLightdir));

			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			glLineWidth(1.0);
			m_model.drawObject();	
		}
		m_program.disable();


		//Draw the AntTweakBar
		TwDraw();

		glfwSwapBuffers(glfwFunc::glfwWindow);
	}

	/// Here all data must be destroyed + glfwTerminate
	void destroy()
	{
		c.Destroy();
		m_model.deleteBuffers();
		m_cone.deleteBuffers();
		glfwDestroyWindow(glfwFunc::glfwWindow);
		glfwTerminate();
		delete quad;
	}
};

int main(int argc, char** argv)
{
	glfwSetErrorCallback(glfwFunc::errorCB);
	if (!glfwInit())	exit(EXIT_FAILURE);
	glfwFunc::glfwWindow = glfwCreateWindow(glfwFunc::WINDOW_WIDTH, glfwFunc::WINDOW_HEIGHT, glfwFunc::strNameWindow.c_str(), NULL, NULL);
	if (!glfwFunc::glfwWindow)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}
	glfwMakeContextCurrent(glfwFunc::glfwWindow);
	if (!glfwFunc::initialize()) exit(EXIT_FAILURE);
	glfwFunc::resizeCB(glfwFunc::glfwWindow, glfwFunc::WINDOW_WIDTH, glfwFunc::WINDOW_HEIGHT);	//just the 1st time
	
	// main loop!
	while (!glfwWindowShouldClose(glfwFunc::glfwWindow))
	{
		glfwFunc::draw();
		glfwPollEvents();	//or glfwWaitEvents()
	}
	glfwFunc::destroy();
	return EXIT_SUCCESS;
}
