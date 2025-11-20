// "C:\Users\pedro\AppData\Local\Programs\Python\Python311\include\Python.h"
#include <Python.h>
#include <stdio.h>


int main() {
    Py_Initialize(); 

    PyObject *name, *load_module,*func,*callfunc, *args;
    
    name = PyUnicode_FromString("py_example_script" );  
    load_module = PyImport_Import(name);
          
    func= PyObject_GetAttrString(load_module,(char *)"fun1");
    callfunc=PyObject_CallObject(func,NULL);
    double f1output = PyFloat_AsDouble(callfunc);
    
    func= PyObject_GetAttrString(load_module,(char *)"fun2");
    args = PyTuple_Pack(1,PyFloat_FromDouble(13));
    callfunc=PyObject_CallObject(func,args);
    double f2output = PyFloat_AsDouble(callfunc);
    
    Py_Finalize();
    
    printf("%f\n",f1output);
    
    int low_val = (int)f2output;

    FILE *fp = fopen("model.xml", "w");
    if (!fp) {
        perror("Failed to create XML file");
        return 1;
    }

    fprintf(fp, "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n");
    fprintf(fp, "<!DOCTYPE nta PUBLIC '-//Uppaal Team//DTD Flat System 1.6//EN' 'http://www.it.uu.se/research/group/darts/uppaal/flat-1_6.dtd'>\n");
    fprintf(fp, "<nta>\n");
    fprintf(fp, "\t<declaration>// global declarations\n");
    fprintf(fp, "int x = 20;\n");
    fprintf(fp, "const int low = %d;\n", low_val);
    fprintf(fp, "const int high = 28;\n");
    fprintf(fp, "bool actuator = false;</declaration>\n");

    fprintf(fp, "\t<template>\n");
    fprintf(fp, "\t\t<name>SimpleAutomaton</name>\n");
    fprintf(fp, "\t\t<declaration>// local declarations</declaration>\n");
    fprintf(fp, "\t\t<location id=\"id0\" x=\"-8\" y=\"0\">\n");
    fprintf(fp, "\t\t\t<name x=\"-18\" y=\"-34\">Off</name>\n");
    fprintf(fp, "\t\t</location>\n");
    fprintf(fp, "\t\t<location id=\"id1\" x=\"408\" y=\"0\">\n");
    fprintf(fp, "\t\t\t<name x=\"398\" y=\"-34\">On</name>\n");
    fprintf(fp, "\t\t</location>\n");
    fprintf(fp, "\t\t<init ref=\"id0\"/>\n");

    // transitions
    fprintf(fp, "\t\t<transition id=\"id2\">\n");
    fprintf(fp, "\t\t\t<source ref=\"id1\"/>\n");
    fprintf(fp, "\t\t\t<target ref=\"id0\"/>\n");
    fprintf(fp, "\t\t\t<label kind=\"guard\" x=\"153\" y=\"-144\">x &gt;= high</label>\n");
    fprintf(fp, "\t\t\t<label kind=\"assignment\" x=\"119\" y=\"-110\">actuator = false</label>\n");
    fprintf(fp, "\t\t\t<nail x=\"280\" y=\"-119\"/>\n");
    fprintf(fp, "\t\t\t<nail x=\"93\" y=\"-119\"/>\n");
    fprintf(fp, "\t\t</transition>\n");

    fprintf(fp, "\t\t<transition id=\"id3\">\n");
    fprintf(fp, "\t\t\t<source ref=\"id1\"/>\n");
    fprintf(fp, "\t\t\t<target ref=\"id1\"/>\n");
    fprintf(fp, "\t\t\t<label kind=\"assignment\" x=\"383\" y=\"59\">x = x + 1</label>\n");
    fprintf(fp, "\t\t\t<nail x=\"442\" y=\"59\"/>\n");
    fprintf(fp, "\t\t\t<nail x=\"365\" y=\"59\"/>\n");
    fprintf(fp, "\t\t</transition>\n");

    fprintf(fp, "\t\t<transition id=\"id4\">\n");
    fprintf(fp, "\t\t\t<source ref=\"id0\"/>\n");
    fprintf(fp, "\t\t\t<target ref=\"id0\"/>\n");
    fprintf(fp, "\t\t\t<label kind=\"assignment\" x=\"-16\" y=\"59\">x = x - 1</label>\n");
    fprintf(fp, "\t\t\t<nail x=\"34\" y=\"59\"/>\n");
    fprintf(fp, "\t\t\t<nail x=\"-34\" y=\"59\"/>\n");
    fprintf(fp, "\t\t</transition>\n");

    fprintf(fp, "\t\t<transition id=\"id5\">\n");
    fprintf(fp, "\t\t\t<source ref=\"id0\"/>\n");
    fprintf(fp, "\t\t\t<target ref=\"id1\"/>\n");
    fprintf(fp, "\t\t\t<label kind=\"guard\" x=\"153\" y=\"-34\">x &lt;= low</label>\n");
    fprintf(fp, "\t\t\t<label kind=\"assignment\" x=\"119\" y=\"8\">actuator = true</label>\n");
    fprintf(fp, "\t\t</transition>\n");

    fprintf(fp, "\t</template>\n");
    fprintf(fp, "\t<system>Process = SimpleAutomaton();\nsystem SimpleAutomaton;</system>\n");
    fprintf(fp, "\t<queries>\n\t\t<query>\n\t\t\t<formula/>\n\t\t\t<comment/>\n\t\t</query>\n\t</queries>\n");
    fprintf(fp, "</nta>\n");

    fclose(fp);
    printf("UPPAAL XML file created with low = %d\n", low_val);


    return 0;
}
