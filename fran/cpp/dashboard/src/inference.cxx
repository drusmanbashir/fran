#include "inference.h"
#include <iostream>


Inference::Inference () {
    m_cascade_mod = py::module_::import("fran.inference.cascade");
    m_base_mod = py::module_::import("fran.inference.base");
    std::cout<<"Inference initialized\n";


}


