#ifndef __Transformation__  
#define __Transformation__


class Transformation
{
	public:
		float m_fScalar;
		float m_fTranslationx, m_fTranslationy, m_fTranslationz;
		float m_fRotationAngle;
		float m_fRotationVectorx, m_fRotationVectory, m_fRotationVectorz;
		bool intersection;
		int iteration;
		unsigned int triID;
		unsigned int originID;
		unsigned int destID;
};

#endif