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

  py::object operator()(size_t i, std::string colname) const;
  py::object operator()(size_t i, size_t j) const;
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
  void run_analyze_resample(int plan_id, int n_procs, bool overwrite = false);
  std::string parseDict();
  bool plan_loaded();
  PlansDF& getPlansDF() { return m_plansDF; }
  const PlansDF& getPlansDF () const{return m_plansDF;};
  const std::string project_title() const { return m_project_title; }

private:
  py::scoped_interpreter guard_;
  py::object m_proj_mod;
  py::module  m_analyze_resample;
  py::dict m_global_properties;
  PlansDF m_plansDF;
  std::string m_project_title;

  FranProject();
  ~FranProject() = default;
  FranProject(const FranProject &) = delete;
  FranProject &operator=(const FranProject &) = delete;
};
