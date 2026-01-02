
#pragma once
#include "franproject.h"
#include <QString>
#include <QVariantMap>


struct InferenceArgs{

};


class Inference{

public: 
  Inference () ;

private:
  py::object m_cascade_mod;
  py::object m_base_mod;
};
