#ifndef OGL_BASIC
#define OGL_BASIC

typedef unsigned char MOUSE_OP;
const MOUSE_OP ROTATE = 0;
const MOUSE_OP TRANSLATE = 1;
const MOUSE_OP SCALE = 2;
class COGLBasic
{
public:
	static COGLBasic& getInstance()
	{
		static COGLBasic    instance; // Guaranteed to be destroyed.
		// Instantiated on first use.
		return instance;
	}
	float fNCP;
	float fFCP;
	float fAngle;
private:
	COGLBasic() {fNCP = 0.01f; fFCP = 500.f; fAngle = 45.f;};                   // Constructor? (the {} brackets) are needed here.

	// C++ 11
	// =======
	// We can use the better technique of deleting the methods
	// we don't want.
	COGLBasic(COGLBasic const&);
	void operator=(COGLBasic const&);
};

#endif