#version 410

layout(location = 0) in vec3 VerPos;

out vec3 Color;

uniform mat4 Mvp;
uniform mat4 ModelMat;

void main()
{
    Color = VerPos;
    gl_Position = ModelMat * Mvp * vec4(VerPos, 1.0);
}