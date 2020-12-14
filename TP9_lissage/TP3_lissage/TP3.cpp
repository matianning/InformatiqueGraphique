// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <float.h>

// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>
GLFWwindow* window;

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
using namespace glm;

#include <common/shader.hpp>
#include <common/texture.hpp>
#include <common/objloader.hpp>
#include <common/vboindexer.hpp>

#include <algorithm>

/******************************************************************************/
/***************           Fonctions à completer         **********************/
/******************************************************************************/
int maxIter = 20;
float rotation = 0.0;
bool automatique = true;
#define PI 3.1415926

/******************************************************************************/

void compute_triangle_normals (const std::vector<glm::vec3> & vertices,
                               const std::vector<std::vector<unsigned short> > & triangles,
                               std::vector<glm::vec3> & triangle_normals){

    for (unsigned int i = 0 ; i < triangles.size() ; i ++) {
        const std::vector<unsigned short> & ti = triangles[i];
        glm::vec3 e01 = vertices[ti[1]] - vertices[ti[0]];
        glm::vec3 e02 = vertices[ti[2]] - vertices[ti[0]];
        glm::vec3 n = glm::cross(e01, e02);
        n = glm::normalize (n);
        triangle_normals.push_back (n);
    }

}

void compute_smooth_vertex_normals (const std::vector<glm::vec3> & vertices,
                                    const std::vector<std::vector<unsigned short> > & triangles,
                                    unsigned int weight_type,
                                    std::vector<glm::vec3> & vertex_normals){

    vertex_normals.clear();
    for ( unsigned int i=0; i < vertices.size(); i++){
        vertex_normals.push_back(glm::vec3(0., 0., 0.));
    }
    std::vector<glm::vec3> triangle_normals;
    compute_triangle_normals (vertices, triangles, triangle_normals);

    for ( unsigned int t_id = 0 ; t_id < triangles.size(); t_id++){
        const glm::vec3 &t_normal = triangle_normals[t_id];
        const std::vector<unsigned short> &t = triangles[t_id];
        for (unsigned int  j = 0; j < 3; j++) {
            const glm::vec3 &vj_pos = vertices[ t[j] ];
            unsigned short vj = t[j];
            float w = 1.0; // uniform weights
            glm::vec3 e0 = vertices[t[(j+1)%3]] - vj_pos;
            glm::vec3 e1 = vertices[t[(j+2)%3]] - vj_pos;
            if (weight_type == 1) { // area weight
                w = glm::length(glm::cross (e0, e1))/2.;
            } else if (weight_type == 2) { // angle weight
                e0 = glm::normalize(e0);
                e1 = glm::normalize(e1);
                w = (2.0 - (glm::dot (e0, e1) + 1.0)) / 2.0;
            }
            if (w <= 0.0)
                continue;
            vertex_normals[vj] =vertex_normals[vj] + t_normal * w;
        }


    }

    for ( unsigned int i=0; i < vertex_normals.size(); i++){
        vertex_normals[i] = glm::normalize(vertex_normals[i]);
    }
}

void collect_one_ring (const std::vector<glm::vec3> & vertices,
                       const std::vector<std::vector<unsigned short> > & triangles,
                       std::vector<std::vector<unsigned short> > & one_ring) {

    one_ring.resize (vertices.size ());
    for (unsigned int i = 0; i < triangles.size (); i++) {
        const std::vector<unsigned short> & ti = triangles[i];
        for (unsigned int j = 0; j < 3; j++) {
            unsigned short vj = ti[j];
            for (unsigned int k = 1; k < 3; k++) {
                unsigned int vk = ti[(j+k)%3];
                if (std::find (one_ring[vj].begin (), one_ring[vj].end (), vk) == one_ring[vj].end ())
                    one_ring[vj].push_back (vk);
            }
        }
    }
}


void compute_vertex_valences (const std::vector<glm::vec3> & vertices,
                              const std::vector<std::vector<unsigned short> > & triangles,
                              std::vector<unsigned int> & valences ) {

    std::vector<std::vector<unsigned short> > one_ring;

    collect_one_ring( vertices, triangles, one_ring );

    valences.clear();
    valences.resize(vertices.size());

    for( unsigned int i = 0 ; i < vertices.size() ; i++ )
        valences[i] = one_ring[i].size();

}

/******************************** Lissage laplacian uniform ******************************/

//Fonction calculer l'opérateur uniforme de laplace 
std::vector<glm::vec3> calc_uniform_mean_curvature(const std::vector<glm::vec3> & vertices,
                                                   const std::vector<std::vector<unsigned short>> & triangles){
    std::vector<glm::vec3> result;
    std::vector<std::vector<unsigned short> > one_ring;
    collect_one_ring( vertices, triangles, one_ring );

    for(unsigned int i = 0; i < vertices.size(); i++){
        glm::vec3 Lu(0.0,0.0,0.0);

        for(unsigned int j = 0; j < one_ring[i].size(); j++){
            Lu+=vertices[one_ring[i][j]];
        }
        if(one_ring[i].size()!=0)
            Lu /=one_ring[i].size();

        Lu -= vertices[i];

        result.push_back(Lu);
    }

    return result;
}

/********************************  Laplace-Beltrami  ***********************************/
float norm(glm::vec3 v){ 
    float x = v.x * v.x;
    float y = v.y * v.y;
    float z = v.z * v.z;
    return sqrt(x+y+z);
}
float min(float a, float b, float c ){
    if(a <= b && a <= c) return a;
    else if(b < a && b <= c) return b;
    else return c;
}

std::vector<float> calc_triangle_quality(const std::vector<glm::vec3> & vertices,
                                         const std::vector<std::vector<unsigned short>> & triangles){

    std::vector<float> result;
    for(unsigned int i = 0; i < triangles.size(); i++){
        float a(0.0), b(0.0), c(0.0);

        a = norm(vertices[triangles[i][0]]);
        b = norm(vertices[triangles[i][1]]);
        c = norm(vertices[triangles[i][2]]);
        glm::vec3 tmp = glm::cross(vertices[triangles[i][0]], vertices[triangles[i][1]]);

        float nominator = a * b * c * 2.0;
        float denominator = 4 * norm(tmp);
        float r = nominator / denominator;
        //********* Stabilité numérique **********
        //éliminer le cas où le dénominateur est petit ou négatif
        if(denominator <= 0.1){ r = 10;}
        float min_longueur = min(a,b,c);
        float rapport = r / min_longueur;

        result.push_back(rapport);
    }
    return result;
}


float calc_weights(const glm::vec3 & v, const glm::vec3 & vi){
    float w(0.0);
    
    float cotAlpha =  glm::dot(v,vi) / norm(glm::cross(v,vi));
    float cotBeta =   glm::dot(vi,v) / norm(glm::cross(vi,v));


    w = 0.5 * (cotAlpha + cotBeta);

    return w;
}


std::vector<glm::vec3> calc_mean_curvature(const std::vector<glm::vec3> & vertices,
                                           const std::vector<std::vector<unsigned short>> & triangles){
    std::vector<glm::vec3> result;
    std::vector<std::vector<unsigned short> > one_ring;
    collect_one_ring( vertices, triangles, one_ring );

    for(unsigned int i = 0; i < vertices.size(); i++){
        glm::vec3 L(0.0,0.0,0.0);
        float total_weight(0.0);
        

        for(unsigned int j = 0; j < one_ring[i].size(); j++){
            float current_weight(0.0);
            current_weight = calc_weights(vertices[i], vertices[one_ring[i][j]]);
            if(current_weight > 30.0) current_weight = 30.0;
            L += current_weight * (vertices[one_ring[i][j]] - vertices[i]);
            total_weight += current_weight;

        }
        
        L /= total_weight;
        result.push_back(L);
    }
    return result;
}

/*************************Gaussian Curvature*********************************/
std::vector<float> calc_gauss_curvature(const std::vector<glm::vec3> & vertices,
                                        const std::vector<std::vector<unsigned short>> & triangles){
    std::vector<float> result;
    std::vector<std::vector<unsigned short> > one_ring;
    collect_one_ring( vertices, triangles, one_ring );

    for(unsigned int i = 0; i < vertices.size(); i++){
        float angle_total(0.0);
        for(unsigned int j = 0; j < one_ring[i].size() - 1; j++){
            float angle(0.0);

            glm::vec3 OA = vertices[one_ring[i][j]] - vertices[i];
            glm::vec3 OB = vertices[one_ring[i][j+1]] - vertices[i];
            float nominator = glm::dot(OA, OB);
            float denominator = norm(OA) * norm(OB);
            angle = acos(nominator / denominator) * 180.0 / PI;
            std::cout<<angle<<" ";
            angle_total += angle;
            
        }
        float res = 360.0 - angle_total;
        result.push_back(res);
    }


    return result;
}

/****************************************************************************/
bool LineMode = false;
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_P && action == GLFW_PRESS){
        if(!LineMode){
            LineMode = true;
            glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
        }
        else{
            LineMode = false;
            glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
        }
    }

    if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS){
        if(!automatique)
            rotation += 0.05f;
    }

    if (key == GLFW_KEY_LEFT && action == GLFW_PRESS){
        if(!automatique)
            rotation -= 0.05f;
    }

    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS){
        if(automatique){
            automatique = false;
            rotation = 0.0f;
        }
        else{
            automatique = true;
        }
        
    }
}

int main( void )
{
    
    // Initialise GLFW
    if( !glfwInit() )
    {
        fprintf( stderr, "Failed to initialize GLFW\n" );
        getchar();
        return -1;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Open a window and create its OpenGL context
    window = glfwCreateWindow( 1024, 768, "TP3 - Lissage", NULL, NULL);
    if( window == NULL ){
        fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n" );
        getchar();
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    //key
    glfwSetKeyCallback(window, key_callback);

    // Initialize GLEW
    glewExperimental = true; // Needed for core profile
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        getchar();
        glfwTerminate();
        return -1;
    }

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    // Hide the mouse and enable unlimited mouvement
    //glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Set the mouse at the center of the screen
    glfwPollEvents();
    glfwSetCursorPos(window, 1024/2, 768/2);

    // Dark blue background
    glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

    // Enable depth test
    glEnable(GL_DEPTH_TEST);
    // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS);

    // Cull triangles which normal is not towards the camera
    glDisable(GL_CULL_FACE);

    GLuint VertexArrayID;
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);

    // Create and compile our GLSL program from the shaders
    GLuint programID = LoadShaders( "vertex_shader.glsl", "fragment_shader.glsl" );

    // Get a handle for our "MVP" uniform
    GLuint MatrixID = glGetUniformLocation(programID, "MVP");
    GLuint ViewMatrixID = glGetUniformLocation(programID, "V");
    GLuint ModelMatrixID = glGetUniformLocation(programID, "M");

    // Load the texture
    GLuint Texture = loadDDS("uvmap.DDS");

    // Get a handle for our "myTextureSampler" uniform
    GLuint TextureID  = glGetUniformLocation(programID, "myTextureSampler");

    std::vector<unsigned short> indices; //Triangles concaténés dans une liste
    std::vector<std::vector<unsigned short> > triangles;
    std::vector<glm::vec3> indexed_vertices;
    std::vector<glm::vec2> indexed_uvs;
    std::vector<glm::vec3> indexed_normals;

    //Chargement du fichier de maillage
    std::string filename("bunny.off");
    loadOFF(filename, indexed_vertices, indices, triangles );
    indexed_uvs.resize(indexed_vertices.size(), glm::vec2(1.)); //List vide de UV

    glm::vec3 bb_min( FLT_MAX, FLT_MAX, FLT_MAX );
    glm::vec3 bb_max( FLT_MIN, FLT_MIN, FLT_MIN );

    //Calcul de la boite englobante du modèle
    for( unsigned int i = 0 ; i < indexed_vertices.size() ; i++ ){
        bb_min = glm::min(bb_min, indexed_vertices[i]);
        bb_max = glm::max(bb_max, indexed_vertices[i]);
    }

    glm::vec3 size = bb_max - bb_min;
    glm::vec3 center = glm::vec3(bb_min.x + size.x/2, bb_min.y + size.y/2 , bb_min.z + size.z/2 );
    float model_scale = 2.0/std::max( std::max(size.x, size.y), size.z );


    //******************** Lissage Laplacien Uniforme***********************/
/*
    for(unsigned int iter = 0; iter < maxIter; iter++){
        std::vector<glm::vec3> uniform_mean_curvature = calc_uniform_mean_curvature(indexed_vertices,triangles);
        std::vector<glm::vec3> indexed_vertices_lisse_uniform;
        for(unsigned int i = 0; i < indexed_vertices.size(); i++){
        uniform_mean_curvature[i].x *= 0.5; 
        uniform_mean_curvature[i].y *= 0.5; 
        uniform_mean_curvature[i].z *= 0.5; 
        indexed_vertices[i] += uniform_mean_curvature[i];
        }
    }

    //**************Calculer la qualité des triangles (Laplacien uniform)**************
    
    std::vector<float> triangle_quality = calc_triangle_quality(indexed_vertices, triangles);
    float mean_quality(0.0);

    for(unsigned int i = 0; i < triangle_quality.size(); i++){
        mean_quality += triangle_quality[i];
    }
    std::cout<<"Qualité triangle Lissage Laplace-uniforme : "<<mean_quality<<std::endl;
    mean_quality /= triangle_quality.size();
    std::cout<<"Qualité moyenne par triangle Laplace-uniforme : "<<mean_quality<<std::endl;
*/

    //******************** Lissage Laplace-Beltrami ***********************/

    for(unsigned int iter = 0; iter < maxIter; iter++){
        std::vector<glm::vec3> mean_curvature = calc_mean_curvature(indexed_vertices,triangles);
        for(unsigned int i = 0; i < indexed_vertices.size(); i++){
        mean_curvature[i].x *= 0.5; 
        mean_curvature[i].y *= 0.5; 
        mean_curvature[i].z *= 0.5; 
        indexed_vertices[i] += mean_curvature[i];
        }
    }

    std::vector<float> triangle_quality_cot = calc_triangle_quality(indexed_vertices, triangles);
    double mean_quality_cot(0.0);
    for(unsigned int i = 0; i < triangle_quality_cot.size(); i++){
        mean_quality_cot +=  triangle_quality_cot[i];
    }

    std::cout<<"Qualité triangle Lissage Laplace-Beltrami : "<<mean_quality_cot<<std::endl;
    mean_quality_cot /= triangle_quality_cot.size();
    std::cout<<"Qualité moyenne par triangle Lissage Laplace-Beltrami : "<<mean_quality_cot<<std::endl;

    //****************************************************************/

    //******************Calculer les normales par sommet*****************
    // indexed_normals.resize(indexed_vertices.size(), glm::vec3(1.));
    compute_smooth_vertex_normals(indexed_vertices, triangles, 2, indexed_normals);

    std::vector<unsigned int> valences;
    compute_vertex_valences( indexed_vertices, triangles, valences );

    unsigned int max_valence = 0;
    for( unsigned int i = 0 ; i < valences.size() ; i++ ){
        max_valence = std::max(max_valence, valences[i]);
    }


    std::vector<float> valence_field(valences.size());

    for( unsigned int i = 0 ; i < valences.size() ; i++ ){
        valence_field[i] = valences[i]/float(max_valence);
    }

    //*****************Calcul Curvature Gaussienne**************
    /*
    float max_curv,min_curv;
    std::vector<float> curvature_gaussienne = calc_gauss_curvature(indexed_vertices, triangles);
    
    for(unsigned int i = 0; i < curvature_gaussienne.size(); i++){
        //std::cout<<i<<":"<<curvature_gaussienne[i]<<std::endl;
        max_curv = std::max(max_curv, curvature_gaussienne[i]);
        min_curv = std::min(min_curv, curvature_gaussienne[i]);
    }
    for(unsigned int i = 0; i < curvature_gaussienne.size(); i++){
        curvature_gaussienne[i] += abs(min_curv);
        curvature_gaussienne[i] /= max_curv;
        std::cout<<curvature_gaussienne[i]<<" ";
    }
*/
   

    //****************************************************************
    // Load it into a VBO

    GLuint vertexbuffer;
    glGenBuffers(1, &vertexbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    glBufferData(GL_ARRAY_BUFFER, indexed_vertices.size() * sizeof(glm::vec3), &indexed_vertices[0], GL_STATIC_DRAW);

    GLuint uvbuffer;
    glGenBuffers(1, &uvbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
    glBufferData(GL_ARRAY_BUFFER, indexed_uvs.size() * sizeof(glm::vec2), &indexed_uvs[0], GL_STATIC_DRAW);

    GLuint normalbuffer;
    glGenBuffers(1, &normalbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
    glBufferData(GL_ARRAY_BUFFER, indexed_normals.size() * sizeof(glm::vec3), &indexed_normals[0], GL_STATIC_DRAW);

    GLuint valencesbuffer;
    glGenBuffers(1, &valencesbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, valencesbuffer);
    glBufferData(GL_ARRAY_BUFFER, valence_field.size() * sizeof(float), &valence_field[0], GL_STATIC_DRAW);

    // Generate a buffer for the indices as well
    GLuint elementbuffer;
    glGenBuffers(1, &elementbuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned short), &indices[0] , GL_STATIC_DRAW);

    // Get a handle for our "LightPosition" uniform
    glUseProgram(programID);
    GLuint LightID = glGetUniformLocation(programID, "LightPosition_worldspace");

    // For speed computation
    double lastTime = glfwGetTime();
    int nbFrames = 0;

    do{



        // Measure speed
        double currentTime = glfwGetTime();
        nbFrames++;
        if ( currentTime - lastTime >= 1.0 ){ // If last prinf() was more than 1sec ago
            // printf and reset
            printf("%f ms/frame\n", 1000.0/double(nbFrames));
            nbFrames = 0;
            lastTime += 1.0;
        }

        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Use our shader
        glUseProgram(programID);

        // Projection matrix : 45 Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
        glm::mat4 ProjectionMatrix = glm::perspective(glm::radians(45.0f), 4.0f / 3.0f, 0.1f, 100.0f);
        // Camera matrix
        glm::mat4 ViewMatrix       = glm::lookAt(
                    glm::vec3(0,0,3), // Camera is at (4,3,3), in World Space
                    glm::vec3(0,0,0), // and looks at the origin
                    glm::vec3(0,1,0)  // Head is up (set to 0,-1,0 to look upside-down)
                    );
        // Model matrix : an identity matrix (model will be at the origin)
        glm::mat4 ModelMatrix      = glm::scale(glm::mat4(1.0f), glm::vec3(model_scale))*glm::translate(glm::mat4(1.0f), glm::vec3(-center.x, -center.y, -center.z));
        if(automatique) rotation+=0.001f;
        ModelMatrix = glm::rotate(ModelMatrix, rotation, glm::vec3(0.0,1.0,0.0));

        glm::mat4 MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;

        // Send our transformation to the currently bound shader,
        // in the "MVP" uniform
        glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);
        glUniformMatrix4fv(ModelMatrixID, 1, GL_FALSE, &ModelMatrix[0][0]);
        glUniformMatrix4fv(ViewMatrixID, 1, GL_FALSE, &ViewMatrix[0][0]);

        glm::vec3 lightPos = glm::vec3(4,4,4);
        glUniform3f(LightID, lightPos.x, lightPos.y, lightPos.z);

        // Bind our texture in Texture Unit 0
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, Texture);
        // Set our "myTextureSampler" sampler to use Texture Unit 0
        glUniform1i(TextureID, 0);

        // 1rst attribute buffer : vertices
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
        glVertexAttribPointer(
                    0,                  // attribute
                    3,                  // size
                    GL_FLOAT,           // type
                    GL_FALSE,           // normalized?
                    0,                  // stride
                    (void*)0            // array buffer offset
                    );

        // 2nd attribute buffer : UVs
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
        glVertexAttribPointer(
                    1,                                // attribute
                    2,                                // size
                    GL_FLOAT,                         // type
                    GL_FALSE,                         // normalized?
                    0,                                // stride
                    (void*)0                          // array buffer offset
                    );

        // 3rd attribute buffer : normals
        glEnableVertexAttribArray(2);
        glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
        glVertexAttribPointer(
                    2,                                // attribute
                    3,                                // size
                    GL_FLOAT,                         // type
                    GL_FALSE,                         // normalized?
                    0,                                // stride
                    (void*)0                          // array buffer offset
                    );

        // 4th attribute buffer : valences
        glEnableVertexAttribArray(3);
        glBindBuffer(GL_ARRAY_BUFFER, valencesbuffer);
        glVertexAttribPointer(
                    3,                                // attribute
                    1,                                // size
                    GL_FLOAT,                         // type
                    GL_FALSE,                         // normalized?
                    0,                                // stride
                    (void*)0                          // array buffer offset
                    );

        // Index buffer
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);


        // Draw the triangles !
        glDrawElements(
                    GL_TRIANGLES,      // mode
                    indices.size(),    // count
                    GL_UNSIGNED_SHORT,   // type
                    (void*)0           // element array buffer offset
                    );

        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glDisableVertexAttribArray(2);



        // Swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();

    } // Check if the ESC key was pressed or the window was closed
    while( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
           glfwWindowShouldClose(window) == 0 );

    // Cleanup VBO and shader
    glDeleteBuffers(1, &vertexbuffer);
    glDeleteBuffers(1, &uvbuffer);
    glDeleteBuffers(1, &normalbuffer);
    glDeleteBuffers(1, &valencesbuffer);
    glDeleteBuffers(1, &elementbuffer);
    glDeleteProgram(programID);
    glDeleteTextures(1, &Texture);
    glDeleteVertexArrays(1, &VertexArrayID);

    // Close OpenGL window and terminate GLFW
    glfwTerminate();

    return 0;
}


