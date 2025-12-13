#pragma once
#ifdef slots
#undef slots
#endif

#include <pybind11/embed.h>

namespace py = pybind11;

class PlansDF {
public:
  PlansDF() : m_plans_py(py::none()) {};
  PlansDF(py::object plans) : m_plans_py(plans) {} // here a python dateframe plans is passed to constructor
  explicit operator bool() const { return !m_plans_py.is_none(); };
  std::vector<std::string> operator[](size_t i) const;
  std::vector<std::string> operator [] (std::string colname) const ;
  std::vector<std::string> header() const;
  size_t rowCount() const ;
  size_t columnCount() const ;
  bool has_column(const std::string& name) const ;
private:
  py::object m_plans_py;
};

class FranProject {
public:
  static FranProject &instance();
  void loadProject(std::string project_name, std::string mnemonic = "");
  std::string parseDict();
  bool plan_loaded();
  PlansDF& getPlansDF() { return m_plansDF; }
  const PlansDF& getPlansDF () const{return m_plansDF;};

private:
  py::scoped_interpreter guard_;
  py::object m_proj_mod;
  py::dict m_global_properties;
  PlansDF m_plansDF;
  std::string m_project_title;

  FranProject();
  ~FranProject() = default;
  FranProject(const FranProject &) = delete;
  FranProject &operator=(const FranProject &) = delete;
};
