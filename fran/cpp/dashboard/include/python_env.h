#pragma once
#ifdef slots
#undef slots
#endif

#include <pybind11/embed.h>

namespace py = pybind11;


class PythonEnv {
public:
  static PythonEnv &instance();
  py::object fn_infer;

private:
  py::scoped_interpreter guard_;
  py::object m_app_;

  PythonEnv();
  ~PythonEnv() = default;
  PythonEnv(const PythonEnv &) = delete;
  PythonEnv &operator=(const PythonEnv &) = delete;
};
