#include <stdio.h>
#include <Python/Python.h>


void module_test(){

	printf("Hello world");

}

int main(){

	return 0;

}

/* Method mapping table */
static PyMethodDef Model_Methods[] = {
    {"module_test", module_test, METH_VARARGS, "Python interface for fputs C library function"},
    {NULL, NULL, 0, NULL}
};

/* Module definition */
static struct PyModuleDef models = {
    PyModuleDef_HEAD_INIT,
    "models",
    "C implementation of AP models",
    -1,
    Model_Methods
};

/*
PyMODINIT_FUNC PyInit_models(void) {
    return PyModule_Create(&models);
}
*/
