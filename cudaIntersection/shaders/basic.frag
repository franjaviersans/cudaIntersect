#version 440

uniform vec4 vec4Color;
uniform vec3 vec3Eye;
uniform vec3 vec3Lightdir;

layout(location = 0) out vec4 vFragColor;


in vec4 vert;
in vec4 norm;


void main(void)
{

	// fragment shader with Oren-Nayar reflectance model

	// set value depending on material
    float roughness = .10;
    vec3 lightColor = vec3(1.0, 1.0, 1.0), ambient = vec3(0.25, 0.25, 0.25);
    
    const float PI = 3.14159;

    // interpolating normals will change the length of the normal, so renormalize the normal.
    vec3 normal = normalize(norm.xyz);
    vec3 eyeDir = normalize(vec3Eye - vert.xyz);
	vec3 lightDir = normalize(vec3Lightdir - vert.xyz);
    
	float intensity = max(dot(normal,lightDir), 0.0);

    // calculate intermediary values
    float NdotL = dot(normal, lightDir);
    float NdotV = dot(normal, eyeDir); 

    float angleVN = acos(NdotV);
    float angleLN = acos(NdotL);
    
    float alpha = max(angleVN, angleLN);
    float beta = min(angleVN, angleLN);
    float gamma = dot(eyeDir - normal * dot(eyeDir, normal), lightDir - normal * dot(lightDir, normal));
    
    float roughnessSquared = roughness * roughness;
    
    // calculate A and B
    float A = 1.0f - 0.5f * (roughnessSquared / (roughnessSquared + 0.57f));

    float B = 0.45f * (roughnessSquared / (roughnessSquared + 0.09f));
 
    float C = sin(alpha) * tan(beta);
    
    // put it all together
    float L1 = max(0.0f, NdotL) * (A + B * max(0.0f, gamma) * C);
    
    // get the final color 
    vec3 finalValue = clamp(lightColor * L1 * vec4Color.xyz + ambient * vec4Color.xyz, vec3(0.0f, 0.0f, 0.0f), vec3(1.0f, 1.0f, 1.0f));
    vFragColor = vec4(finalValue, vec4Color.w);
}